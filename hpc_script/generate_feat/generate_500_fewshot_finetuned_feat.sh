#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

# generate the feature for COCO dataset (few-shot setting), using the CLIP porposal

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

## generate fewshot_finetuned feature
#1
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#2
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#3
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#4
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#5
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#6
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#7
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#8
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#9
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#10
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#11
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#12
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True 

#13
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#14
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

#15
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

# remain
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

# generate the gt only
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth 3 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=False model.test_cfg.generate_gt_feat=True \
model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/fewshot_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_fewshot \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True

