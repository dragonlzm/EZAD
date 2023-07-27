_base_ = './cls_finetuner_clip_base48_all_train.py'

data_root = 'data/coco/'

data = dict(train=dict(ann_file=data_root + 'annotations/train_100shots.json'))
#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

#optimizer = dict(lr=0.00005)
lr_config = dict(_delete_=True, policy='step', step=[14])
runner = dict(max_epochs=24)