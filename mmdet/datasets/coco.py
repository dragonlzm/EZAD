# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

ZEROSHOT_COCO_SPLIT = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                    'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                    'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
                    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
                    'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
                    'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush'),
    NOVEL_CLASSES=('airplane', 'bus', 'cat', 'dog', 'cow', 
                    'elephant', 'umbrella', 'tie', 'snowboard', 
                    'skateboard', 'cup', 'knife', 'cake', 'couch', 
                    'keyboard', 'sink', 'scissors'),
    BASE_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'train', 
                    'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
                    'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
                    'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
                    'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
                    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
                    'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
                    'mouse', 'remote', 'microwave', 'oven', 'toaster', 
                    'refrigerator', 'book', 'clock', 'vase', 'toothbrush'))

FEWSHOT_COCO_SPLIT = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    NOVEL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 
                    'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
                    'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
                    'couch', 'potted plant', 'dining table', 'tv'),
    BASE_CLASSES=('truck', 'traffic light', 'fire hydrant', 'stop sign', 
                    'parking meter', 'bench', 'elephant', 'bear', 'zebra', 
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                    'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
                    'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                    'teddy bear', 'hair drier', 'toothbrush'))


@DATASETS.register_module()
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _patchacc2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['score'] = result
            json_results.append(data)
        return json_results

    def _gtacc2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['score'] = result
            json_results.append(data)
        return json_results

    def _clipproposal2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['score'] = result
            json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], np.ndarray) and (results[0].shape[0] in [2, 5, 6]):
            json_results = self._gtacc2json(results)
            result_files['gt_acc'] = f'{outfile_prefix}.gt_acc.json'
            mmcv.dump(json_results, result_files['gt_acc'])
        # elif isinstance(results[0], np.ndarray) and len(results[0]) > 5:
        #     json_results = self._patchacc2json(results)
        #     result_files['patch_acc'] = f'{outfile_prefix}.patch_acc.json'
        #     mmcv.dump(json_results, result_files['patch_acc'])
        elif isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def calc_patch_acc(self, results):
        aver_acc = 0
        for i in range(len(self.img_ids)):
            patches_gt = torch.from_numpy(self.patches_gt[i]).view(-1)
            # if the patches is not the bg, the position will become True
            patches_gt_sign = (patches_gt != 1.0000e+04)
            predict_result = torch.from_numpy(results[i])
            # if the predicted score higher than 0.5 regarded as fg, the position is True
            predict_result_sign = (predict_result >= 0.5) 
            img_acc = torch.sum(patches_gt_sign == predict_result_sign).float()
            img_acc /= predict_result.shape[0]
            aver_acc += img_acc
        aver_acc = aver_acc.item()
        aver_acc /= len(self.img_ids)
        return aver_acc

    def calc_gt_anchor_iou(self, results):
        total_iou = 0
        total_proposal = 0
        for i in range(len(results)):
            total_proposal += results[i].shape[0]
            total_iou += results[i].sum()
        aver_iou = total_iou / total_proposal
        return aver_iou

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

    def calc_proposal_selection_eval(self, results):
        loss = torch.nn.L1Loss()
        pred_val = torch.cat([torch.from_numpy(ele[0]) for ele in results],dim=0)
        pred_val_target = torch.cat([torch.from_numpy(ele[1]) for ele in results],dim=0)
        
        if self.seperate_base_and_novel:
            # select the base
            pred_val_base = torch.cat([torch.from_numpy(ele[0][..., self.base_idx]) for ele in results],dim=0)
            pred_val_target_base = torch.cat([torch.from_numpy(ele[1][..., self.base_idx]) for ele in results],dim=0)
            loss_val_base = loss(pred_val_base, pred_val_target_base)
            # select novel
            pred_val_novel = torch.cat([torch.from_numpy(ele[0][..., self.novel_idx]) for ele in results], dim=0)
            pred_val_target_novel = torch.cat([torch.from_numpy(ele[1][..., self.novel_idx]) for ele in results], dim=0)
            loss_val_novel = loss(pred_val_novel, pred_val_target_novel)
            return loss_val_base.item(), loss_val_novel.item()
        else:
            loss_val = loss(pred_val, pred_val_target)
            return loss_val.item()

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i], cat_ids=self.cat_ids)
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann.get('iscrowd', False):
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'patch_acc', 'gt_acc', 'gt_anchor_iou', 'proposal_selection']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            #if metric == 'clip_proposal':
            #    avg_proposal_num = self.calc_clip_proposal(results)
            #    eval_results['clip_proposal'] = avg_proposal_num
            #    log_msg = f'\navg_proposal_num\t{avg_proposal_num:.4f}'
            #    print_log(log_msg, logger=logger)
            #    continue
            if metric == 'gt_anchor_iou':
                acc = self.calc_gt_anchor_iou(results)
                eval_results['gt_anchor_iou'] = acc
                log_msg = f'\gt_anchor_iou\t{acc:.4f}'
                print_log(log_msg, logger=logger)
                continue                
            if metric == 'patch_acc':
                acc = self.calc_patch_acc(results)
                eval_results['patch_acc'] = acc
                log_msg = f'\nacc\t{acc:.4f}'
                print_log(log_msg, logger=logger)
                continue
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
                continue
            if metric == 'proposal_selection':
                if self.seperate_base_and_novel:
                    loss_base, loss_novel = self.calc_proposal_selection_eval(results)
                    eval_results['loss_base'] = loss_base
                    eval_results['loss_novel'] = loss_novel
                    log_msg = f'\n loss_base\t{loss_base:.4f}' +  f'\n loss_novel\t{loss_novel:.4f}'
                    print_log(log_msg, logger=logger)
                else:
                    loss = self.calc_proposal_selection_eval(results)
                    eval_results['loss'] = loss
                    log_msg = f'\n loss\t{loss:.4f}'
                    print_log(log_msg, logger=logger)
                continue
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            # eval each class splits
            if self.eval_on_splits is not None:
                if self.eval_on_splits == 'zeroshot':
                    class_splits = ZEROSHOT_COCO_SPLIT
                else:
                    class_splits = FEWSHOT_COCO_SPLIT
                for split_name in class_splits.keys():
                    split_cat_ids = [
                        self.cat_ids[i] for i in range(len(self.CLASSES))
                        if self.CLASSES[i] in class_splits[split_name]
                    ]
                    self._evaluate_by_class_split(
                        cocoGt,
                        cocoDt,
                        iou_type,
                        proposal_nums,
                        iou_thrs,
                        split_cat_ids,
                        metric,
                        metric_items,
                        eval_results,
                        False,
                        logger,
                        split_name=split_name + ' ')
            else:
                # eval all classes
                self._evaluate_by_class_split(cocoGt, cocoDt, iou_type,
                                            proposal_nums, iou_thrs,
                                            self.cat_ids, metric, metric_items,
                                            eval_results, classwise, logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _evaluate_by_class_split(self,
                                 cocoGt,
                                 cocoDt,
                                 iou_type,
                                 proposal_nums,
                                 iou_thrs,
                                 cat_ids,
                                 metric,
                                 metric_items,
                                 eval_results,
                                 classwise,
                                 logger,
                                 split_name = ''):
        """Evaluation a split of classes in COCO protocol.

        Args:
            cocoGt (object): coco object with ground truth annotations.
            cocoDt (object): coco object with detection results.
            iou_type (str): Type of IOU.
            proposal_nums (Sequence[int]): Number of proposals.
            iou_thrs (float | Sequence[float]): Thresholds of IoU.
            cat_ids (list[int]): Class ids of classes to be evaluated.
            metric (str): Metrics to be evaluated.
            metric_items (str | list[str]): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            eval_results (dict[str, float]): COCO style evaluation metric.
            classwise (bool): Whether to evaluating the AP for each class.
            split_name (str): Name of split. Default:''.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.params.catIds = cat_ids
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')
        if split_name is not None:
            print_log(f'\n evaluation of {split_name} class', logger=logger)
        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]

            for item in metric_items:
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[split_name + item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2], \
                    f'{self.cat_ids},{precisions.shape}'

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [split_name + 'category', split_name + 'AP'] * (
                    num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[split_name + key] = val
            ap = cocoEval.stats[:6]
            eval_results[split_name + f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            return eval_results
