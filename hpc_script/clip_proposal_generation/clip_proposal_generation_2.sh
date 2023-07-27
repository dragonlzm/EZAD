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

# This script aims for generate CLIP proposal for the COCO zero-shot settting (this procedure wil be distributed in 15 sub-tasks)

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco_base48.py"
# JSONFILE_PREFIX="data/test/cls_proposal_generator_coco/results_32_32_512_base48"
# BBOX_SAVE_PATH_ROOT="data/coco/clip_proposal/32_32_512_base48"

# for fixing the channel problem
CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py"
JSONFILE_PREFIX="data/test/cls_proposal_generator_coco/results_32_32_512_channel_corr"
BBOX_SAVE_PATH_ROOT="data/coco/clip_proposal/32_32_512_channel_corr"
CHECKPOINT="data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth"


# 1
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json 

# 2
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_2 \
    --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
    model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
    data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json 

# 3
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_3 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json 

# 4
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_4 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json 

# 5
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_5 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json 

# 6
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_6 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json 

# 7
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_7 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json 

# 8
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_8 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json 

# 9
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_9 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json 

# 10
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_10 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json 


# 11
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_11 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json 

# 12
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json 

# 13
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json 

# 14
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json 

# 15
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json 


# the remain
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_remain \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017.json 