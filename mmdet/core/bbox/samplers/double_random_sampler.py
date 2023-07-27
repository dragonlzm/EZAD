# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler
from .sampling_result import SamplingResult

### in here we mainly rewrite the sample function in the class
@BBOX_SAMPLERS.register_module()
class DoubleRandomSampler(RandomSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 double_random_sample=True,
                 drop_ratio=0.3,
                 **kwargs):
        super(DoubleRandomSampler, self).__init__(**kwargs)
        self.double_random_sample = double_random_sample
        self.drop_ratio = drop_ratio

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        
        # we try to randomly drop some samples here
        if self.double_random_sample:
            # concat the pos_inds and neg_inds
            pos_and_neg_idxs = torch.cat([pos_inds, neg_inds], dim=0)
            # obtain the remained idx of the pos_and_neg_idxs
            sample_factor = 1 - np.random.rand(1)[0] * self.drop_ratio
            nms_topk = int(pos_and_neg_idxs.shape[0] * sample_factor)
            subsample_idxs = np.random.choice(pos_and_neg_idxs.shape[0], nms_topk, replace=False)
            subsample_idxs = torch.from_numpy(subsample_idxs).to(pos_and_neg_idxs.device)
            # sort the remained idx of the pos_and_neg_idxs
            #subsample_idxs, _ = torch.sort(subsample_idxs)
            pos_idx_max = pos_inds.shape[0]
            # split the idx into remained_idx of pos_inds and remained_idx of neg_inds
            pos_subsample_idx = subsample_idxs[subsample_idxs < pos_idx_max]
            neg_subsample_idx = subsample_idxs[subsample_idxs >= pos_idx_max]
            # print('before sample:', 'pos_inds:', pos_inds.shape, pos_inds,
            #     'neg_inds:', neg_inds.shape, neg_inds, 'subsample_idxs', subsample_idxs.shape, subsample_idxs,
            #     'pos_subsample_idx', pos_subsample_idx, 'neg_subsample_idx', neg_subsample_idx)
            
            # update the pos_inds, neg_inds
            pos_inds = pos_and_neg_idxs[pos_subsample_idx]
            neg_inds = pos_and_neg_idxs[neg_subsample_idx]
            # print('after sample:', 'pos_inds:', pos_inds.shape, pos_inds,
            #       'neg_inds', neg_inds.shape, neg_inds)

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

