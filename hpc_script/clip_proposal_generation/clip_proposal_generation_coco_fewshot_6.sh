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

# This script aims for generate CLIP proposal for the COCO few-shot settting, 
# finetune the model using the 60 base and using all 80 cates for generating proposal (this procedure will be distributed in 15 subtasks)

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py"
CHECK_POINT="/project/nevatia_174/zhuoming/detection/cls_finetuner/cls_finetuner_clip_base60_all_train/epoch_12.pth"
JSONFILE_PREFIX="data/test/cls_proposal_generator_coco/results_32_32_512_fewshot"
BBOX_SAVE_PATH_ROOT="data/coco/clip_proposal/32_32_512_fewshot"

# # 1
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json 
		

# # 2
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_2 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json 


# # 3
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_3 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json 

# # 4
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_4 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json 

# # 5
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_5 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json 

# 6
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECK_POINT} 2 \
--eval=proposal_fast \
--options jsonfile_prefix=${JSONFILE_PREFIX}_6 \
--cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
    model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
    data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json 

# # 7
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_7 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json 

# # 8
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_8 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json 

# # 9
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_9 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json 

# # 10
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_10 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json 

# # 11
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_11 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json 

# # 12
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_12 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json 

# # 13
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_13 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json 

# # 14
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_14 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json 

# # 15
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_15 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json 

# # the remain
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECK_POINT} 2 \
# --eval=proposal_fast \
# --options jsonfile_prefix=${JSONFILE_PREFIX}_remain \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
#     data.test.ann_file=data/coco/annotations/instances_train2017.json 