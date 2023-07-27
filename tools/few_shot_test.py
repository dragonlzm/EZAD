# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.datasets.utils import get_copy_dataset_type
from mmdet.models import build_detector
from mmdet.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/show the '
            'results / save the results) with the argument "--out", "--eval"',
            '"--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    cfg = compat_cfg(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    # update overall dataloader(for train, val and test) setting
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # currently only support single images testing
    assert test_loader_cfg['samples_per_gpu'] == 1, \
        'currently only support single images testing'
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # for meta-learning methods which require support template dataset
    # for model initialization.
    if cfg.data.get('model_init', None) is not None:
        origin_copy_from_train_dataset = cfg.data.model_init.pop('copy_from_train_dataset')
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        # try to load the ann_cfg from the checkpoint
        if cfg.data.model_init.get('ann_cfg', None) is None:
            assert checkpoint['meta'].get('model_init_ann_cfg',
                                          None) is not None
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = \
                checkpoint['meta']['model_init_ann_cfg']
        # if the check point in the model does not contain the "data_infos"
        # usually it's the checkpoint before finetuneing
        # we should get hte ann_cfg from the training data
        ## check the data_infos
        mark_contain_data_infos = True
        if isinstance(cfg.data.model_init.ann_cfg, dict):
            if cfg.data.model_init.ann_cfg.get('data_infos', None) is None:
                mark_contain_data_infos = False
        elif isinstance(cfg.data.model_init.ann_cfg, list):
            for ann_cfg_ in cfg.data.model_init.ann_cfg:
                if ann_cfg_.get('data_infos', None) is None:
                    mark_contain_data_infos = False
                    break
        
        # update the ann_cfg if data_infos not in data_infos:
        if not mark_contain_data_infos:
            if origin_copy_from_train_dataset:
                if cfg.data.model_init.ann_cfg is not None:
                    warnings.warn(
                        'model_init dataset will copy support '
                        'dataset used for training and original '
                        'ann_cfg will be discarded', UserWarning)
                train_dataset = build_dataset(cfg.data.train)
                # 200 annotation for novel cates, 10 anno for each cate
                # cfg.data.model_init.ann_cfg [{'data_infos': [{'id': 386164, 'filename': 'train2017/000000386164.jpg', 
                # 'width': 480, 'height': 640, 'ann': {'bboxes': array([[  7.19,  35.96, 478.92, 624.18]], dtype=float32), 
                # 'labels': array([18])}}, {'id': 51191, 'filename': 'train2017/000000051191.jpg', 
                # 'width': 628, 'height': 640, 'ann': {'bboxes': array([[373.06, 129.63, 481.23, 340.27]], 
                # dtype=float32), 'labels': array([17])}}, ···
                
                cfg.data.model_init.ann_cfg = [
                    dict(data_infos=train_dataset.get_support_data_infos())
                ]
                
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)

    if not distributed:
        # Please use MMCV >= 1.4.4 for CPU testing!
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        show_kwargs = dict(show_score_thr=args.show_score_thr)
        if cfg.data.get('model_init', None) is not None:
            from mmdet.apis import (single_gpu_model_init,
                                                  single_gpu_fewshot_test)
            single_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import single_gpu_fewshot_test
        outputs = single_gpu_fewshot_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if cfg.data.get('model_init', None) is not None:
            from mmdet.apis import (multi_gpu_model_init,
                                                  multi_gpu_fewshot_test)
            multi_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import multi_gpu_fewshot_test
        outputs = multi_gpu_fewshot_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
