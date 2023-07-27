_base_ = './cls_finetuner_clip_base48_all_train.py'

data_root = 'data/coco/'

data = dict(train=dict(ann_file=data_root + 'annotations/train_10shots.json'))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]
