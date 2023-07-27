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

# generating the CLIP proposal on LVIS dataset using the all LVIS name 
# (for our proposed LVIS zero-shot setting, Freq as base, comm + rare as novel,
# the model is finetuned only on freq, not freq + comm), this procedure will be distributed in 13 subtasks

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# this script is for generating the proposal for the freq and common


CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_lvis_fc866.py"
JSONFILE_PREFIX="data/test/cls_proposal_generator_coco/results_lvis_32_32_512"
BBOX_SAVE_PATH_ROOT="data/detection/lvis_v1/clip_proposal/lvis_fc866_32_32_512"
CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_freq405_train_gt_only_100_rand/epoch_18.pth"

# 1
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_0_8000.json 

# 2
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_8000_16000.json 

# 3
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_16000_24000.json 

# 4
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_24000_32000.json 

# 5
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_32000_40000.json 

# 6
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_40000_48000.json 

# 7
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
    --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
    model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
    data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_48000_56000.json 

# 8
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_56000_64000.json 

# 9
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_64000_72000.json 

# 10
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_72000_80000.json 

# 11
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_80000_88000.json 

# 12
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_88000_96000.json 

# 13
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_96000_104000.json 

# the remain
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT} \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json 