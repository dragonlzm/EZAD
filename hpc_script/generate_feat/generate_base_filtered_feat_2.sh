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

# this script aims to generate the CLIP feature(base finetuned feature), with CLIP proposal
# the proposal is filtered base on the base categories gt bboxes
# this procedure is distributed in two subtasks

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# spliting the generation into other section to accelerate the procedure
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/filter_base \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True
