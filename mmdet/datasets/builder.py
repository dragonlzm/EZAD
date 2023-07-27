# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial
import os.path as osp
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, Sampler


import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, ConfigDict
from torch.utils.data import DataLoader

from .samplers import (DistributedGroupSampler, DistributedSampler, GroupSampler,
                       InfiniteSampler, InfiniteGroupSampler, DistributedInfiniteSampler, 
                       DistributedInfiniteGroupSampler)

from .collate import multi_pipeline_collate_fn

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg: ConfigDict,
                  default_args: Dict = None,
                  rank: Optional[int] = None,
                  work_dir: Optional[str] = None,
                  timestamp: Optional[str] = None) -> Dataset:
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                ClassBalancedDataset, MultiImageMixDataset,
                                QueryAwareDataset)
    from .utils import get_copy_dataset_type
    # If save_dataset is set to True, dataset will be saved into json.
    save_dataset = cfg.pop('save_dataset', False)

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'QueryAwareDataset':
        query_dataset = build_dataset(cfg['dataset'], default_args)
        # build support dataset
        if cfg.get('support_dataset', None) is not None:
            # if `copy_from_query_dataset` is True, copy and update config
            # from query_dataset and copy `data_infos` by using copy dataset
            # to avoid reproducing random sampling.
            if cfg['support_dataset'].pop('copy_from_query_dataset', False):
                support_dataset_cfg = copy.deepcopy(cfg['dataset'])
                support_dataset_cfg.update(cfg['support_dataset'])
                support_dataset_cfg['type'] = get_copy_dataset_type(
                    cfg['dataset']['type'])
                support_dataset_cfg['ann_cfg'] = [
                    dict(data_infos=copy.deepcopy(query_dataset.data_infos))
                ]
                cfg['support_dataset'] = support_dataset_cfg
            support_dataset = build_dataset(cfg['support_dataset'],
                                            default_args)
        # support dataset will be a copy of query dataset in QueryAwareDataset
        else:
            support_dataset = None

        dataset = QueryAwareDataset(
            query_dataset,
            support_dataset,
            num_support_ways=cfg['num_support_ways'],
            num_support_shots=cfg['num_support_shots'],
            repeat_times=cfg.get('repeat_times', 1))
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    # save dataset for the reproducibility
    if rank == 0 and save_dataset:
        save_dataset_path = osp.join(work_dir, f'{timestamp}_saved_data.json')
        if hasattr(dataset, 'save_data_infos'):
            dataset.save_data_infos(save_dataset_path)
        else:
            raise AttributeError(
                f'`save_data_infos` is not implemented in {type(dataset)}.')

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = True,
                     shuffle: bool = True,
                     seed: Optional[int] = None,
                     data_cfg: Optional[Dict] = None,
                     use_infinite_sampler: bool = False,
                     **kwargs) -> DataLoader:
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
            Default:1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int): Random seed. Default:None.
        data_cfg (dict | None): Dict of data configure. Default: None.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                            ClassBalancedDataset, MultiImageMixDataset,
                            QueryAwareDataset)
    rank, world_size = get_dist_info()
    (sampler, batch_size, num_workers) = build_sampler(
        dist=dist,
        shuffle=shuffle,
        dataset=dataset,
        num_gpus=num_gpus,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        seed=seed,
        use_infinite_sampler=use_infinite_sampler)
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    if isinstance(dataset, QueryAwareDataset):
        # `QueryAwareDataset` will return a list of DataContainer
        # `multi_pipeline_collate_fn` are designed to handle
        # the data with list[list[DataContainer]]
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(
                multi_pipeline_collate_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_sampler(
        dist: bool,
        shuffle: bool,
        dataset: Dataset,
        num_gpus: int,
        samples_per_gpu: int,
        workers_per_gpu: int,
        seed: int,
        use_infinite_sampler: bool = False) -> Tuple[Sampler, int, int]:
    """Build pytorch sampler for dataLoader.

    Args:
        dist (bool): Distributed training/test or not.
        shuffle (bool): Whether to shuffle the data at every epoch.
        dataset (Dataset): A PyTorch dataset.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        seed (int): Random seed.
        use_infinite_sampler (bool): Whether to use infinite sampler.
            Noted that infinite sampler will keep iterator of dataloader
            running forever, which can avoid the overhead of worker
            initialization between epochs. Default: False.

    Returns:
        tuple: Contains corresponding sampler and arguments

            - sampler(:obj:`sampler`) : Corresponding sampler
              used in dataloader.
            - batch_size(int): Batch size of dataloader.
            - num_works(int): The number of processes loading data in the
                data loader.
    """

    rank, world_size = get_dist_info()
    if dist:
        # Infinite sampler will return a infinite stream of index. But,
        # the length of infinite sampler is set to the actual length of
        # dataset, thus the length of dataloader is still determined
        # by the dataset.

        if shuffle:
            if use_infinite_sampler:
                sampler = DistributedInfiniteGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                # DistributedGroupSampler will definitely shuffle the data to
                # satisfy that images on each GPU are in the same group
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            if use_infinite_sampler:
                sampler = DistributedInfiniteSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if use_infinite_sampler:
            sampler = InfiniteGroupSampler(
                dataset, samples_per_gpu, seed=seed, shuffle=shuffle)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) \
                if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    return sampler, batch_size, num_workers
