_base_ = './cls_finetuner_clip_base48_all_train_resnet50.py'

# change the classification loss
# add the text encoder module
# fixed the all parametere except the ln of the text encoder
model = dict(
    backbone=dict(
        fixed_param=True,
        open_ln=True,
    ))