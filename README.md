# Efficient Feature Distillation for Zero-shot Annotation Object Detection (EZAD)
This codebase hosts the project of Efficient Feature Distillation for Zero-shot Annotation Object Detection (EZAD), as presented in our paper:

    Efficient Feature Distillation for Zero-shot Annotation Object Detection;
    Zhuoming Liu, Xuefeng Hu, Ram Nevatia;
    The IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.
    arXiv preprint arXiv:2303.12145

The full paper is available at: [https://arxiv.org/abs/2303.12145](https://arxiv.org/abs/2303.12145). 
Implementation based on MMDetection is included in [MMDetection](https://github.com/open-mmlab/mmdetection).



## Highlights
- **Eliminating the Domain Gap:** For bridging the domain gap, we find that simply finetuning the layer normalization layers in the CLIP with the base category instances is effective. It significantly improves the classification accuracy on both base and novel of COCO dataset by 25% in general. It also improve the performance of zero-shot detector on novel categories of COCO dataset by 11.1% in AP.

- **CLIP proposals:** We use CLIP to select the distillation regions with the help of the novel category names. It improve the zero-shot detector's performance on novel by 5.8% in AP.

Our method uses the name of the novel categories, which combined with CLIP proposals and domain gap elimination can significantly improve the model performance and training efficiency.


## Required hardware
We use 2 NVIDIA V100 GPUs for experiments. 


## Prepare the Environment
Please refer the following commands to prepare the environment:

    conda create --name pyt
    conda activate pyt
    
    # install the pytorch
    conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
    
    # install the mmcv
    pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

    # install other packets
    pip install fiftyone tensorboard ftfy seaborn pycocotools terminaltables regex isort lvis opencv-python PyYAML
    conda install tensorflow-estimator==2.1.0

If your node could not connect to the internet, you should use run the `tools/mmcv_installation.sh` to install the MMCV. The `tools/mmcv_installation.sh` will use the pre-cloned MMCV codebase to build the MMCV. This scipt needs the GPU to bulid the MMCV.


## How to reproduce our results
To fully reproduce the EZSD, you have to run the following steps:
### 1. Finetuning the CLIP
For COCO dataset zero-shot setting, you can use the following command to finetune the CLIP:

    sbatch hpc_script/finetune_clip/clip_finetune_base_full.sh

For COCO dataset few-shot setting, you can use the following command to finetune the CLIP:

    sbatch hpc_script/finetune_clip/clip_finetune_base60_full.sh

For LVIS dataset, freq+comm(base)/rare(novel), you can use the following command to finetune the CLIP:

    sbatch hpc_script/finetune_clip/cls_finetuner_clip_lvis_base_train.sh

For LVIS dataset, freq(base)/comm+rare(novel), you can use the following command to finetune the CLIP:

    sbatch hpc_script/finetune_clip/cls_finetuner_clip_lvis_freq_train.sh

### 2. Generating the CLIP proposal
For COCO dataset zero-shot setting, you can use the following command to generate the CLIP proposals, this procedure is divided into 15 tasks:

    sbatch hpc_script/clip_proposal_generation/clip_proposal_generation_*.sh

For COCO dataset few-shot setting, you can use the following command to generate the CLIP proposals, this procedure is divided into 15 tasks:

    sbatch hpc_script/clip_proposal_generation/clip_proposal_generation_coco_fewshot_*.sh

For LVIS dataset, freq+comm(base)/rare(novel), you can use the following command to generate the CLIP proposals, this procedure is divided into 13 tasks:

    sbatch hpc_script/clip_proposal_generation/clip_proposal_generation_lvis_original_*.sh

For LVIS dataset, freq(base)/comm+rare(novel), you can use the following command to generate the CLIP proposals, this procedure is divided into 13 tasks. (Since we want the model performance on Rare to be the Open Vocabulary Detection, we do not use Rare name in generating the CLIP proposal):

    sbatch hpc_script/clip_proposal_generation/clip_proposal_generation_lvis_lvisfc866_*.sh


### 3. Extract the CLIP feature using the CLIP proposal
For COCO dataset zero-shot setting, you can use the following command to extract CLIP Proposal features, this procedure is divided into 2 tasks:

    sbatch hpc_script/generate_feat/generate_base_filtered_feat_1.sh
    sbatch hpc_script/generate_feat/generate_base_filtered_feat_2.sh

For COCO dataset few-shot setting, you can use the following command to extract CLIP Proposal features:

    sbatch hpc_script/generate_feat/generate_500_fewshot_finetuned_feat.sh

For LVIS dataset, freq+comm(base)/rare(novel), you can use the following command to extract CLIP Proposal features:

    sbatch hpc_script/generate_feat/generate_lvis_feat.sh

For LVIS dataset, freq(base)/comm+rare(novel), you can use the following command to extract CLIP Proposal features:

    sbatch hpc_script/generate_feat/generate_lvis_fc866_feat.sh

### 4. Train a detector with the CLIP feature
For COCO dataset zero-shot setting, you can use the following command to train the detector:

    sbatch hpc_script/two_stage_model/mask_rcnn_distillation_coco.sh

For COCO dataset few-shot setting, you can use the following command to train the detector:

    sbatch hpc_script/two_stage_model/mask_rcnn_distillation_coco_few_shot.sh

For LVIS dataset, freq+comm(base)/rare(novel), you can use the following command to train the detector:

    sbatch hpc_script/two_stage_model/mask_rcnn_distillation_lvis.sh

For LVIS dataset, freq(base)/comm+rare(novel), you can use the following command to train the detector:

    sbatch hpc_script/two_stage_model/mask_rcnn_distillation_lvis_freq_comm.sh


## Acknowledgement 
This material is based on research sponsored by Air Force Research Laboratory (AFRL) under agreement number FA8750-19-1-1000. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation therein. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policiesor endorsements, either expressedor implied, of Air ForceLaboratory, DARPA or the U.S.Government.

## License
For academic use only. For commercial use, please contact the authors. 