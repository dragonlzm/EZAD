_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

## this script using the clip proposal + freq finetuned

# default bs = 2x2
# dataset settings
classes = ('air_conditioner', 'airplane', 'alarm_clock', 'antenna', 'apple', 'apron', 'armchair', 'trash_can', 'avocado', 'awning', 'baby_buggy', 'backpack', 'handbag', 'suitcase', 'ball', 'balloon', 'banana', 'bandanna', 'banner', 'barrel', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'bath_mat', 'bath_towel', 'bathtub', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'bicycle', 'visor', 'billboard', 'bird', 'birthday_cake', 'blackboard', 'blanket', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bolt', 'book', 'boot', 'bottle', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'box', 'bracelet', 'bread', 'bridal_gown', 'broccoli', 'bucket', 'bun', 'buoy', 'bus_(vehicle)', 'butter', 'button', 'cab_(taxi)', 'cabinet', 'cake', 'calendar', 'camera', 'can', 'candle', 'candle_holder', 'cap_(headwear)', 'bottle_cap', 'car_(automobile)', 'railcar_(part_of_a_train)', 'carrot', 'tote_bag', 'cat', 'cauliflower', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'cigarette', 'cistern', 'clock', 'clock_tower', 'coaster', 'coat', 'coffee_maker', 'coffee_table', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'edible_corn', 'cowboy_hat', 'crate', 'crossbar', 'crumb', 'cucumber', 'cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'deck_chair', 'desk', 'dining_table', 'dish', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'doorknob', 'doughnut', 'drawer', 'dress', 'dress_suit', 'dresser', 'duck', 'duffel_bag', 'earphone', 'earring', 'egg', 'refrigerator', 'elephant', 'fan', 'faucet', 'figurine', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fireplace', 'fireplug', 'fish', 'flag', 'flagpole', 'flip-flop_(sandal)', 'flower_arrangement', 'fork', 'frisbee', 'frying_pan', 'giraffe', 'glass_(drink_container)', 'glove', 'goggles', 'grape', 'green_bean', 'green_onion', 'grill', 'guitar', 'hairbrush', 'ham', 'hair_dryer', 'hand_towel', 'handle', 'hat', 'headband', 'headboard', 'headlight', 'helmet', 'hinge', 'home_plate_(baseball)', 'fume_hood', 'hook', 'horse', 'hose', 'polar_bear', 'iPod', 'jacket', 'jar', 'jean', 'jersey', 'key', 'kitchen_sink', 'kite', 'knee_pad', 'knife', 'knob', 'ladder', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lanyard', 'laptop_computer', 'latch', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'log', 'speaker_(stero_equipment)', 'magazine', 'magnet', 'mailbox_(at_home)', 'manhole', 'map', 'marker', 'mask', 'mast', 'mattress', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'monitor_(computer_equipment) computer_monitor', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'mug', 'mushroom', 'napkin', 'necklace', 'necktie', 'newspaper', 'notebook', 'nut', 'oar', 'onion', 'orange_(fruit)', 'ottoman', 'oven', 'paddle', 'painting', 'pajamas', 'pan_(for_cooking)', 'paper_plate', 'paper_towel', 'parking_meter', 'pastry', 'pear', 'pen', 'pencil', 'pepper', 'person', 'piano', 'pickle', 'pickup_truck', 'pillow', 'pineapple', 'pipe', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'pole', 'polo_shirt', 'pop_(soda)', 'poster', 'pot', 'flowerpot', 'potato', 'printer', 'propeller', 'quilt', 'radiator', 'rearview_mirror', 'reflector', 'remote_control', 'ring', 'rubber_band', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'sail', 'salad', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'scissors', 'scoreboard', 'scrubbing_brush', 'sheep', 'shirt', 'shoe', 'shopping_bag', 'short_pants', 'shoulder_bag', 'shower_head', 'shower_curtain', 'signboard', 'sink', 'skateboard', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'snowboard', 'soap', 'soccer_ball', 'sock', 'sofa', 'soup', 'spatula', 'spectacles', 'spoon', 'statue_(sculpture)', 'steering_wheel', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunglasses', 'surfboard', 'sweater', 'sweatshirt', 'swimsuit', 'table', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tarp', 'teapot', 'teddy_bear', 'telephone', 'telephone_pole', 'television_set', 'tennis_ball', 'tennis_racket', 'thermostat', 'tinfoil', 'tissue_paper', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'towel', 'towel_rack', 'toy', 'traffic_light', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tripod', 'trousers', 'truck', 'umbrella', 'underwear', 'urinal', 'vase', 'vent', 'vest', 'wall_socket', 'wallet', 'watch', 'water_bottle', 'watermelon', 'weathervane', 'wet_suit', 'wheel', 'windshield_wiper', 'wine_bottle', 'wineglass', 'blinder_(for_horses)', 'wristband', 'wristlet', 'zebra')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/lvis_v1/clip_proposal_feat/lvis_freq_finetuned_vision_and_text',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights']),
]

# oversample dataset
# dataset settings
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
data = dict(
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=classes)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        classes=classes))

# model config
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=405,
                       fg_vec_cfg=dict(load_path='data/embeddings/freq405_vnt_finetuned_freq405.pt')), 
        mask_head=dict(num_classes=405)))

# optimizer
optimizer = dict(lr=0.005)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=36, metric=['bbox', 'segm'])