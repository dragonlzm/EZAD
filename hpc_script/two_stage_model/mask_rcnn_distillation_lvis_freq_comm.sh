#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

# Mask R-CNN with Distillation experiment on LVIS freq(base)/comm+rare(novel)
cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection

# 48 epoch exp
ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
model.roi_head.bbox_head.learnable_temperature=True"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp_48e"
TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_48e.py"
TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"
#START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp/epoch_16.pth"

bash tools/new_dist_train.sh ${TRAIN_CONFIG} 2 \
${WORK_DIR} /data/zhuoming/detection \
--cfg-options ${ADDITIONAL_CONFIG} \
#--resume-from=${START_FROM}
#--resume-from=${WORK_DIR}/latest.pth



CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh ${TEST_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}

