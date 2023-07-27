# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
import math
from collections import defaultdict

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES
from .coco import CocoDataset
from .few_shot_base import BaseFewShotDataset
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings


@DATASETS.register_module()
class QueryAwareDataset:
    """A wrapper of QueryAwareDataset.

    Building QueryAwareDataset requires query and support dataset.
    Every call of `__getitem__` will firstly sample a query image and its
    annotations. Then it will use the query annotations to sample a batch
    of positive and negative support images and annotations. The positive
    images share same classes with query, while the annotations of negative
    images don't have any category from query.

    Args:
        query_dataset (:obj:`BaseFewShotDataset`):
            Query dataset to be wrapped.
        support_dataset (:obj:`BaseFewShotDataset` | None):
            Support dataset to be wrapped. If support dataset is None,
            support dataset will copy from query dataset.
        num_support_ways (int): Number of classes for support in
            mini-batch, the first one always be the positive class.
        num_support_shots (int): Number of support shots for each
            class in mini-batch, the first K shots always from positive class.
        repeat_times (int): The length of repeated dataset will be `times`
            larger than the original dataset. Default: 1.
    """

    def __init__(self,
                 query_dataset: BaseFewShotDataset,
                 support_dataset: Optional[BaseFewShotDataset],
                 num_support_ways: int,
                 num_support_shots: int,
                 repeat_times: int = 1) -> None:
        self.query_dataset = query_dataset
        if support_dataset is None:
            self.support_dataset = self.query_dataset
        else:
            self.support_dataset = support_dataset
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        self.CLASSES = self.query_dataset.CLASSES
        self.repeat_times = repeat_times
        assert self.num_support_ways <= len(
            self.CLASSES
        ), 'Please set `num_support_ways` smaller than the number of classes.'
        # build data index (idx, gt_idx) by class.
        self.data_infos_by_class = {i: [] for i in range(len(self.CLASSES))}
        # counting max number of annotations in one image for each class,
        # which will decide whether sample repeated instance or not.
        self.max_anns_num_one_image = [0 for _ in range(len(self.CLASSES))]
        # count image for each class annotation when novel class only
        # has one image, the positive support is allowed sampled from itself.
        self.num_image_by_class = [0 for _ in range(len(self.CLASSES))]

        for idx in range(len(self.support_dataset)):
            labels = self.support_dataset.get_ann_info(idx)['labels']
            class_count = [0 for _ in range(len(self.CLASSES))]
            for gt_idx, gt in enumerate(labels):
                self.data_infos_by_class[gt].append((idx, gt_idx))
                class_count[gt] += 1
            for i in range(len(self.CLASSES)):
                # number of images for each class
                if class_count[i] > 0:
                    self.num_image_by_class[i] += 1
                # max number of one class annotations in one image
                if class_count[i] > self.max_anns_num_one_image[i]:
                    self.max_anns_num_one_image[i] = class_count[i]

        for i in range(len(self.CLASSES)):
            assert len(self.data_infos_by_class[i]
                       ) > 0, f'Class {self.CLASSES[i]} has zero annotation'
            if len(
                    self.data_infos_by_class[i]
            ) <= self.num_support_shots - self.max_anns_num_one_image[i]:
                warnings.warn(
                    f'During training, instances of class {self.CLASSES[i]} '
                    f'may smaller than the number of support shots which '
                    f'causes some instance will be sampled multiple times')
            if self.num_image_by_class[i] == 1:
                warnings.warn(f'Class {self.CLASSES[i]} only have one '
                              f'image, query and support will sample '
                              f'from instance of same image')

        # Disable the group sampler, because in few shot setting,
        # one group may only has two or three images.
        if hasattr(self.query_dataset, 'flag'):
            self.flag = np.zeros(
                len(self.query_dataset) * self.repeat_times, dtype=np.uint8)

        self._ori_len = len(self.query_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Return query image and support images at the same time.

        For query aware dataset, this function would return one query image
        and num_support_ways * num_support_shots support images. The support
        images are sampled according to the selected query image. There should
        be no intersection between the classes of instances in query data and
        in support data.

        Args:
            idx (int): the index of data.

        Returns:
            dict: A dict contains query data and support data, it
            usually contains two fields.

                - query_data: A dict of single query data information.
                - support_data: A list of dict, has
                  num_support_ways * num_support_shots support images
                  and corresponding annotations.
        """
        idx %= self._ori_len
        # sample query data
        try_time = 0
        while True:
            try_time += 1
            cat_ids = self.query_dataset.get_cat_ids(idx)
            # query image have too many classes, can not find enough
            # negative support classes.
            if len(self.CLASSES) - len(cat_ids) >= self.num_support_ways - 1:
                break
            else:
                idx = self._rand_another(idx) % self._ori_len
            assert try_time < 100, \
                'Not enough negative support classes for ' \
                'query image, please try a smaller support way.'

        query_class = np.random.choice(cat_ids)
        query_gt_idx = [
            i for i in range(len(cat_ids)) if cat_ids[i] == query_class
        ]
        query_data = self.query_dataset.prepare_train_img(
            idx, 'query', query_gt_idx)
        query_data['query_class'] = [query_class]

        # sample negative support classes, which not appear in query image
        support_class = [
            i for i in range(len(self.CLASSES)) if i not in cat_ids
        ]
        support_class = np.random.choice(
            support_class,
            min(self.num_support_ways - 1, len(support_class)),
            replace=False)
        support_idxes = self.generate_support(idx, query_class, support_class)
        support_data = [
            self.support_dataset.prepare_train_img(idx, 'support', [gt_idx])
            for (idx, gt_idx) in support_idxes
        ]
        return {'query_data': query_data, 'support_data': support_data}

    def __len__(self) -> int:
        """Length after repetition."""
        return len(self.query_dataset) * self.repeat_times

    def _rand_another(self, idx: int) -> int:
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def generate_support(self, idx: int, query_class: int,
                         support_classes: List[int]) -> List[Tuple[int]]:
        """Generate support indices of query images.

        Args:
            idx (int): Index of query data.
            query_class (int): Query class.
            support_classes (list[int]): Classes of support data.

        Returns:
            list[tuple(int)]: A mini-batch (num_support_ways *
                num_support_shots) of support data (idx, gt_idx).
        """
        support_idxes = []
        if self.num_image_by_class[query_class] == 1:
            # only have one image, instance will sample from same image
            pos_support_idxes = self.sample_support_shots(
                idx, query_class, allow_same_image=True)
        else:
            # instance will sample from different image from query image
            pos_support_idxes = self.sample_support_shots(idx, query_class)
        support_idxes.extend(pos_support_idxes)
        for support_class in support_classes:
            neg_support_idxes = self.sample_support_shots(idx, support_class)
            support_idxes.extend(neg_support_idxes)
        return support_idxes

    def sample_support_shots(
            self,
            idx: int,
            class_id: int,
            allow_same_image: bool = False) -> List[Tuple[int]]:
        """Generate support indices according to the class id.

        Args:
            idx (int): Index of query data.
            class_id (int): Support class.
            allow_same_image (bool): Allow instance sampled from same image
                as query image. Default: False.
        Returns:
            list[tuple[int]]: Support data (num_support_shots)
                of specific class.
        """
        support_idxes = []
        num_total_shots = len(self.data_infos_by_class[class_id])

        # count number of support instance in query image
        cat_ids = self.support_dataset.get_cat_ids(idx % self._ori_len)
        num_ignore_shots = len([1 for cat_id in cat_ids if cat_id == class_id])

        # set num_sample_shots for each time of sampling
        if num_total_shots - num_ignore_shots < self.num_support_shots:
            # if not have enough support data allow repeated data
            num_sample_shots = num_total_shots
            allow_repeat = True
        else:
            # if have enough support data not allow repeated data
            num_sample_shots = self.num_support_shots
            allow_repeat = False
        while len(support_idxes) < self.num_support_shots:
            selected_gt_idxes = np.random.choice(
                num_total_shots, num_sample_shots, replace=False)

            selected_gts = [
                self.data_infos_by_class[class_id][selected_gt_idx]
                for selected_gt_idx in selected_gt_idxes
            ]
            for selected_gt in selected_gts:
                # filter out query annotations
                if selected_gt[0] == idx:
                    if not allow_same_image:
                        continue
                if allow_repeat:
                    support_idxes.append(selected_gt)
                elif selected_gt not in support_idxes:
                    support_idxes.append(selected_gt)
                if len(support_idxes) == self.num_support_shots:
                    break
            # update the number of data for next time sample
            num_sample_shots = min(self.num_support_shots - len(support_idxes),
                                   num_sample_shots)
        return support_idxes

    def save_data_infos(self, output_path: str) -> None:
        """Save data_infos into json."""
        self.query_dataset.save_data_infos(output_path)
        # for query aware datasets support and query set use same data
        paths = output_path.split('.')
        self.support_dataset.save_data_infos(
            '.'.join(paths[:-1] + ['support_shot', paths[-1]]))

    def get_support_data_infos(self) -> List[Dict]:
        """Return data_infos of support dataset."""
        return copy.deepcopy(self.support_dataset.data_infos)


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.separate_eval = separate_eval
        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                if cat_id not in self.dataset.cat_ids:
                    continue
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids if cat_id in self.dataset.cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None):
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = dataset.flag
        self.num_samples = len(dataset)

        if dynamic_scale is not None:
            assert isinstance(dynamic_scale, tuple)
        self._dynamic_scale = dynamic_scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                results['mix_results'] = mix_results

            if self._dynamic_scale is not None:
                # Used for subsequent pipeline to automatically change
                # the output image size. E.g MixUp, Resize.
                results['scale'] = self._dynamic_scale

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

    def update_dynamic_scale(self, dynamic_scale):
        """Update dynamic_scale. It is called by an external hook.

        Args:
            dynamic_scale (tuple[int]): The image scale can be
               changed dynamically.
        """
        assert isinstance(dynamic_scale, tuple)
        self._dynamic_scale = dynamic_scale
