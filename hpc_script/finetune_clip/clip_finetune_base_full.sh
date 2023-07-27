#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=24:00:00
#SBATCH --account=nevatia_174

# finetuning the CLIP on COCO base categories (zero-shot, base 48)

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/cls_finetuner_clip_base48_all_train
    #--resume-from=/project/nevatia_174/zhuoming/detection/test/new_rpn_patches246_coco/latest.pth