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

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

## this script is for extracting the CLIP feature for Freq/comm+rare LVIS setting
## using RPN proposal, the CLIP is only finetuned on the freq

# spliting the generation into other section to accelerate the procedure
#CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2/epoch_18.pth"
CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_freq405_train_gt_only_100_rand/epoch_18.pth"
CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py"
BBOX_SAVE_PATH_ROOT="data/detection/lvis_v1/clip_proposal/lvis_fc866_32_32_512"
FEAT_SAVE_PATH_ROOT="data/lvis_v1/clip_proposal_feat/lvis_freq_finetuned_vision_and_text"

#### update for using the best overall perf model to extract feature, no longer filter the base cate
# 1
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_0_8000.json 

# 2
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_8000_16000.json 

# 3
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_16000_24000.json 

# 4
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_24000_32000.json 

# 5
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_32000_40000.json 

# 6
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_40000_48000.json 

# 7
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_48000_56000.json 

# 8
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_56000_64000.json 

# 9
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_64000_72000.json

# 10
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_72000_80000.json 

# 11
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_80000_88000.json 

# 12
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_88000_96000.json 

# 13
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_96000_104000.json 

# remain
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json 