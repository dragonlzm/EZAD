#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

# finetuning the CLIP on LVIS base categories of LVIS zero-shot setting 
# (base categories includes the freq and comm categories total 866 categories)

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data


WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_freq405_train_lr0002"
bash tools/new_dist_train.sh configs/cls_finetuner/cls_finetuner_clip_lvis_freq405_train.py 2 \
${WORK_DIR} ./data \
--cfg-options runner.max_epochs=18 optimizer.lr=0.0002 \
#--resume-from=${WORK_DIR}/latest.pth


# for testing base
bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_freq405_train.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/base_results

# for testing novel
bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_comm461_train.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/novel_results

# for testing all
bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/all_results
