# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
from typing import List, Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from mmcv.parallel import is_module_wrapper
from mmdet.utils import get_root_logger
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def single_gpu_fewshot_test(model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Optional[str] = None,
                    show_score_thr: float = 0.3) -> List:
    """Test model with single gpu for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (DataLoader): Pytorch data loader.
        show (bool): Whether to show the image. Default: False.
        out_dir (str | None): The directory to write the image. Default: None.
        show_score_thr (float): Minimum score of bboxes to be shown.
            Default: 0.3.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `test` mode
            result = model(mode='test', rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            # make sure each time only one image to be shown
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                if is_module_wrapper(model):
                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
                else:
                    model.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        results.extend(result)

        prog_bar.update(batch_size)
    return results


def multi_gpu_fewshot_test(model: nn.Module,
                   data_loader: DataLoader,
                   tmpdir: str = None,
                   gpu_collect: bool = False) -> List:
    """Test model with multiple gpus for meta-learning based detector.

    The model forward function requires `mode`, while in mmdet it requires
    `return_loss`. And the `encode_mask_results` is removed.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `test` mode
            result = model(mode='test', rescale=True, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            prog_bar.update(batch_size * world_size)

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    logger = get_root_logger()
    logger.info('starting model initialization...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `model_init` mode
            result = model(mode='model_init', **data)
        results.append(result)
        prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # `model_init` will process the forward features saved in model.
    if is_module_wrapper(model):
        model.module.model_init()
    else:
        model.model_init()
    logger.info('model initialization done.')

    return results


def multi_gpu_model_init(model: nn.Module, data_loader: DataLoader) -> List:
    """Forward support images for meta-learning based detector initialization.

    The function usually will be called before `single_gpu_test` in
    `QuerySupportEvalHook`. It firstly forwards support images with
    `mode=model_init` and the features will be saved in the model.
    Then it will call `:func:model_init` to process the extracted features
    of support images to finish the model initialization.

    Noted that the `data_loader` should NOT use distributed sampler, all the
    models in different gpus should be initialized with same images.

    Args:
        model (nn.Module): Model used for extracting support template features.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[Tensor]: Extracted support template features.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, _ = get_dist_info()
    if rank == 0:
        logger = get_root_logger()
        logger.info('starting model initialization...')
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    # the model_init dataloader do not use distributed sampler to make sure
    # all of the gpus get the same initialization
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # forward in `model_init` mode
            result = model(mode='model_init', **data)
        results.append(result)
        if rank == 0:
            prog_bar.update(num_tasks=len(data['img_metas'].data[0]))
    # model_init function will process the forward features saved in model.
    if is_module_wrapper(model):
        model.module.model_init()
    else:
        model.model_init()
    if rank == 0:
        logger.info('model initialization done.')
    return results
