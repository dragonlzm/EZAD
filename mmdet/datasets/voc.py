# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time

VOC_SPLIT1 = dict(
    ALL_CLASSES=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor'),
    NOVEL_CLASSES=('bird', 'bus', 'cow', 'motorbike', 'sofa'),
    BASE_CLASSES=('aeroplane', 'bicycle', 'boat', 'bottle', 'car',
                    'cat', 'chair', 'diningtable', 'dog', 'horse',
                    'person', 'pottedplant', 'sheep', 'train',
                    'tvmonitor'))

VOC_SPLIT2 = dict(
    ALL_CLASSES=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor'),
    NOVEL_CLASSES=('aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    BASE_CLASSES=('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                    'chair', 'diningtable', 'dog', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'train', 'tvmonitor'))

VOC_SPLIT3 = dict(
    ALL_CLASSES=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor'),
    NOVEL_CLASSES=('boat', 'cat', 'motorbike', 'sheep', 'sofa'),
    BASE_CLASSES=('aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
                    'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'person', 'pottedplant', 'train', 'tvmonitor'))


@DATASETS.register_module()
class VOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall', 'gt_acc']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        
        if metric == 'gt_acc':
            over_all_acc, s_acc, m_acc, l_acc, person_acc, overall_entropy, all_cos_score = self.calc_gt_acc(results)
            eval_results['over_all_acc'] = over_all_acc
            eval_results['s_acc'] = s_acc
            eval_results['m_acc'] = m_acc
            eval_results['l_acc'] = l_acc
            eval_results['person_acc'] = person_acc
            eval_results['overall_entropy'] = overall_entropy
            eval_results['all_cos_score'] = all_cos_score

            log_msg = f'\n over_all_acc\t{over_all_acc:.4f}' + \
                f'\n s_acc\t{s_acc:.4f}' + \
                f'\n m_acc\t{m_acc:.4f}' + \
                f'\n l_acc\t{l_acc:.4f}' + \
                f'\n person_acc\t{person_acc:.4f}' + \
                f'\n overall_entropy\t{overall_entropy:.4f}' + \
                f'\n all_cos_score\t{all_cos_score:.4f}'
            print_log(log_msg, logger=logger)
        
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                
                # calculate evaluate results of different class splits
                if self.eval_on_splits is not None:
                    if self.eval_on_splits == 'split1':
                        class_splits = VOC_SPLIT1
                    elif self.eval_on_splits == 'split2':
                        class_splits = VOC_SPLIT2
                    else:
                        class_splits = VOC_SPLIT3
                    class_splits_mean_aps = {k: [] for k in class_splits.keys()}
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)
                
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def calc_gt_acc(self, results):
        all_gts = 0
        correct_num = 0
        gt_num_over_scales = np.array([0,0,0])
        corr_num_over_scales = np.array([0,0,0])
        person_gt_num = 0
        person_correct_num = 0
        
        all_entropy = 0
        all_max_score = []
        
        all_cos_score = 0

        for ele in results:
            if torch.is_tensor(ele):
                continue
            pred_res = torch.from_numpy(ele[0])
            gt_res = torch.from_numpy(ele[1])
            scale_info = torch.from_numpy(ele[2])
            entro_result = torch.from_numpy(ele[3])
            max_score = torch.from_numpy(ele[4])
            cos_score = torch.from_numpy(ele[5])

            all_max_score.append(max_score)
            # if -1 in the gt_res it means, it using the random bbox for prediction
            gt_num = pred_res.shape[0]
            all_gts += gt_num
            if -1 not in gt_res:
                # calculate the acc over all scale
                matched_res = (pred_res == gt_res)
                correct_num += matched_res.sum().item()

                # calculate the acc over different scale 
                s_pred = pred_res[scale_info==0]
                s_gt = gt_res[scale_info==0]
                m_pred = pred_res[scale_info==1]
                m_gt = gt_res[scale_info==1]
                l_pred = pred_res[scale_info==2]
                l_gt = gt_res[scale_info==2]

                # deal with small, median and large
                gt_num_over_scales[0] += s_pred.shape[0]
                gt_num_over_scales[1] += m_pred.shape[0]
                gt_num_over_scales[2] += l_pred.shape[0]

                corr_num_over_scales[0] += (s_pred == s_gt).sum().item()
                corr_num_over_scales[1] += (m_pred == m_gt).sum().item()
                corr_num_over_scales[2] += (l_pred == l_gt).sum().item()

                # deal with the person categories
                person_pred = pred_res[gt_res==0]
                person_gt = gt_res[gt_res==0]
                person_gt_num += person_pred.shape[0]
                person_correct_num += (person_pred == person_gt).sum().item()

            # aggregate the entropy
            all_entropy += entro_result.sum().item()
            all_cos_score += cos_score.sum().item()

        if -1 not in gt_res:
            over_all_acc = correct_num / all_gts
            acc_over_scales = corr_num_over_scales / gt_num_over_scales
            s_acc = acc_over_scales[0]
            m_acc = acc_over_scales[1]
            l_acc = acc_over_scales[2]
            person_acc = person_correct_num / person_gt_num
        else:
            over_all_acc, s_acc, m_acc, l_acc, person_acc = 0, 0, 0, 0, 0

        all_entropy = all_entropy / all_gts
        all_cos_score = all_cos_score / all_gts

        # distri visualization
        all_max_score = torch.cat(all_max_score).cpu().numpy()
        #print(all_max_score.shape)
        if self.visualization_path != None:
            sns.displot(all_max_score, kde=True)
            #plt.show()
            file_name = str(time.time()) + '.png'
            # create path if path is not exist
            if not os.path.exists(self.visualization_path):
                os.makedirs(self.visualization_path)            
            plt.savefig(os.path.join(self.visualization_path, file_name))
        return over_all_acc, s_acc, m_acc, l_acc, person_acc, all_entropy, all_cos_score
