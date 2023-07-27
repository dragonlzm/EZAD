#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

## training the distillation with COCO few-shot setting
cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection

# 301212 (learnable temperature)
WORK_DIR="data/grad_clip_check/mask_rcnn_distillation_r101_base60_fewshot_setting"
ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
model.roi_head.bbox_head.learnable_temperature=True"

bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_r101_base60_fewshot_setting.py 2 \
${WORK_DIR} /data/zhuoming/detection \
--cfg-options ${ADDITIONAL_CONFIG} \
#--resume-from=${WORK_DIR}/latest.pth

# test the model
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base60_tuned_clip_feat_r101_fpn_1x_coco.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options model.roi_head.bbox_head.reg_with_cls_embedding=True data.test.eval_on_splits='fewshot' \
model.test_cfg.rcnn.score_thr=0.0 model.test_cfg.rcnn.max_per_img=300 \
${ADDITIONAL_CONFIG}
