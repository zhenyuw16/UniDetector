_base_ = [
    '../_base_/default_runtime.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        style='pytorch'),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        shared_head=dict(type='CLIPResLayer', layers=[3, 4, 6, 3]),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='BBoxHeadCLIPSplit3',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            with_cls=False,
            reg_class_agnostic=True,

            use_fed=True,
            cat_freq_path='data/OID/annotations/oid_challenge_2019_train_bbox_train0.2@50_cat_info.json',
            #use_hierarchy=True,
            #hierarchy_path='data/OID/annotations/challenge-2019-label500-hierarchy-list.json',

            zeroshot_path_coco='../Detic/datasets/metadata/coco_clip_a+cname_rn50_manyprompt.npy',
            zeroshot_path_obj365='../Detic/datasets/metadata/object365_clip_a+cname_rn50_manyprompt.npy',
            zeroshot_path_oid='../Detic/datasets/metadata/oid_clip_a+cname_rn50_manyprompt.npy',
            #zeroshot_path='../Detic/datasets/metadata/oid_clip_a+cname_rn50_manyprompt_background.npy',
            num_classes=500,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #loss_cls=dict(type="EQLv2", num_classes=500),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25, ##########
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            #score_thr=0.05,
            #nms=dict(type='nms', iou_threshold=0.5),
            score_thr=0.0001, #5,
            nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
            max_per_img=100)))


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/Object365/'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('person', 'sneakers', 'chair', 'hat', 'lamp', 'bottle', 'cabinet/shelf', 'cup', 'car', 'glasses', 'picture/frame', 'desk', 'handbag', 'street lights', 'book', 'plate', 'helmet', 'leather shoes', 'pillow', 'glove', 'potted plant', 'bracelet', 'flower', 'tv', 'storage box', 'vase', 'bench', 'wine glass', 'boots', 'bowl', 'dining table', 'umbrella', 'boat', 'flag', 'speaker', 'trash bin/can', 'stool', 'backpack', 'couch', 'belt', 'carpet', 'basket', 'towel/napkin', 'slippers', 'barrel/bucket', 'coffee table', 'suv', 'toy', 'tie', 'bed', 'traffic light', 'pen/pencil', 'microphone', 'sandals', 'canned', 'necklace', 'mirror', 'faucet', 'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch', 'sink', 'horse', 'fish', 'apple', 'camera', 'candle', 'teddy bear', 'cake', 'motorcycle', 'wild bird', 'laptop', 'knife', 'traffic sign', 'cell phone', 'paddle', 'truck', 'cow', 'power outlet', 'clock', 'drum', 'fork', 'bus', 'hanger', 'nightstand', 'pot/pan', 'sheep', 'guitar', 'traffic cone', 'tea pot', 'keyboard', 'tripod', 'hockey', 'fan', 'dog', 'spoon', 'blackboard/whiteboard', 'balloon', 'air conditioner', 'cymbal', 'mouse', 'telephone', 'pickup truck', 'orange', 'banana', 'airplane', 'luggage', 'skis', 'soccer', 'trolley', 'oven', 'remote', 'baseball glove', 'paper towel', 'refrigerator', 'train', 'tomato', 'machinery vehicle', 'tent', 'shampoo/shower gel', 'head phone', 'lantern', 'donut', 'cleaning products', 'sailboat', 'tangerine', 'pizza', 'kite', 'computer box', 'elephant', 'toiletries', 'gas stove', 'broccoli', 'toilet', 'stroller', 'shovel', 'baseball bat', 'microwave', 'skateboard', 'surfboard', 'surveillance camera', 'gun', 'life saver', 'cat', 'lemon', 'liquid soap', 'zebra', 'duck', 'sports car', 'giraffe', 'pumpkin', 'piano', 'stop sign', 'radiator', 'converter', 'tissue ', 'carrot', 'washing machine', 'vent', 'cookies', 'cutting/chopping board', 'tennis racket', 'candy', 'skating and skiing shoes', 'scissors', 'folder', 'baseball', 'strawberry', 'bow tie', 'pigeon', 'pepper', 'coffee machine', 'bathtub', 'snowboard', 'suitcase', 'grapes', 'ladder', 'pear', 'american football', 'basketball', 'potato', 'paint brush', 'printer', 'billiards', 'fire hydrant', 'goose', 'projector', 'sausage', 'fire extinguisher', 'extension cord', 'facial mask', 'tennis ball', 'chopsticks', 'electronic stove and gas stove', 'pie', 'frisbee', 'kettle', 'hamburger', 'golf club', 'cucumber', 'clutch', 'blender', 'tong', 'slide', 'hot dog', 'toothbrush', 'facial cleanser', 'mango', 'deer', 'egg', 'violin', 'marker', 'ship', 'chicken', 'onion', 'ice cream', 'tape', 'wheelchair', 'plum', 'bar soap', 'scale', 'watermelon', 'cabbage', 'router/modem', 'golf ball', 'pine apple', 'crane', 'fire truck', 'peach', 'cello', 'notepaper', 'tricycle', 'toaster', 'helicopter', 'green beans', 'brush', 'carriage', 'cigar', 'earphone', 'penguin', 'hurdle', 'swing', 'radio', 'CD', 'parking meter', 'swan', 'garlic', 'french fries', 'horn', 'avocado', 'saxophone', 'trumpet',\
'sandwich', 'cue', 'kiwi fruit', 'bear', 'fishing rod', 'cherry', 'tablet', 'green vegetables', 'nuts', 'corn', 'key', 'screwdriver', 'globe', 'broom', 'pliers', 'volleyball', 'hammer', 'eggplant', 'trophy', 'dates', 'board eraser', 'rice', 'tape measure/ruler', 'dumbbell', 'hamimelon', 'stapler', 'camel', 'lettuce', 'goldfish', 'meat balls', 'medal', 'toothpaste', 'antelope', 'shrimp', 'rickshaw', 'trombone', 'pomegranate', 'coconut', 'jellyfish', 'mushroom', 'calculator', 'treadmill', 'butterfly', 'egg tart', 'cheese', 'pig', 'pomelo', 'race car', 'rice cooker', 'tuba', 'crosswalk sign', 'papaya', 'hair drier', 'green onion', 'chips', 'dolphin', 'sushi', 'urinal', 'donkey', 'electric drill', 'spring rolls', 'tortoise/turtle', 'parrot', 'flute', 'measuring cup', 'shark', 'steak', 'poker card', 'binoculars', 'llama', 'radish', 'noodles', 'yak', 'mop', 'crab', 'microscope', 'barbell', 'bread/bun', 'baozi', 'lion', 'red cabbage', 'polar bear', 'lighter', 'seal', 'mangosteen', 'comb', 'eraser', 'pitaya', 'scallop', 'pencil case', 'saw', 'table tennis paddle', 'okra', 'starfish', 'eagle', 'monkey', 'durian', 'game board', 'rabbit', 'french horn', 'ambulance', 'asparagus', 'hoverboard', 'pasta', 'target', 'hotair balloon', 'chainsaw', 'lobster', 'iron', 'flashlight')



classes_oid = ('Infant bed', 'Rose', 'Flag', 'Flashlight', 'Sea turtle', 'Camera', 'Animal', 'Glove', 'Crocodile', 'Cattle', 'House', 'Guacamole', 'Penguin', 'Vehicle registration plate', 'Bench', 'Ladybug', 'Human nose', 'Watermelon', 'Flute', 'Butterfly', 'Washing machine', 'Raccoon', 'Segway', 'Taco', 'Jellyfish', 'Cake', 'Pen', 'Cannon', 'Bread', 'Tree', 'Shellfish', 'Bed', 'Hamster', 'Hat', 'Toaster', 'Sombrero', 'Tiara', 'Bowl', 'Dragonfly', 'Moths and butterflies', 'Antelope', 'Vegetable', 'Torch', 'Building', 'Power plugs and sockets', 'Blender', 'Billiard table', 'Cutting board', 'Bronze sculpture', 'Turtle', 'Broccoli', 'Tiger', 'Mirror', 'Bear', 'Zucchini', 'Dress', 'Volleyball', 'Guitar', 'Reptile', 'Golf cart', 'Tart', 'Fedora', 'Carnivore', 'Car', 'Lighthouse', 'Coffeemaker', 'Food processor', 'Truck', 'Bookcase', 'Surfboard', 'Footwear', 'Bench', 'Necklace', 'Flower', 'Radish', 'Marine mammal', 'Frying pan', 'Tap', 'Peach', 'Knife', 'Handbag', 'Laptop', 'Tent', 'Ambulance', 'Christmas tree', 'Eagle', 'Limousine', 'Kitchen & dining room table', 'Polar bear', 'Tower', 'Football', 'Willow', 'Human head', 'Stop sign', 'Banana', 'Mixer', 'Binoculars', 'Dessert', 'Bee', 'Chair', 'Wood-burning stove', 'Flowerpot', 'Beaker', 'Oyster', 'Woodpecker', 'Harp', 'Bathtub', 'Wall clock', 'Sports uniform', 'Rhinoceros', 'Beehive', 'Cupboard', 'Chicken', 'Man', 'Blue jay', 'Cucumber', 'Balloon', 'Kite', 'Fireplace', 'Lantern', 'Missile', 'Book', 'Spoon', 'Grapefruit', 'Squirrel', 'Orange', 'Coat', 'Punching bag', 'Zebra', 'Billboard', 'Bicycle', 'Door handle', 'Mechanical fan', 'Ring binder', 'Table', 'Parrot', 'Sock', 'Vase', 'Weapon', 'Shotgun', 'Glasses', 'Seahorse', 'Belt', 'Watercraft', 'Window', 'Giraffe', 'Lion', 'Tire', 'Vehicle', 'Canoe', 'Tie', 'Shelf', 'Picture frame', 'Printer', 'Human leg', 'Boat', 'Slow cooker', 'Croissant', 'Candle', 'Pancake', 'Pillow', 'Coin', 'Stretcher', 'Sandal', 'Woman', 'Stairs', 'Harpsichord', 'Stool', 'Bus', 'Suitcase', 'Human mouth', 'Juice', 'Skull', 'Door', 'Violin', 'Chopsticks', 'Digital clock', 'Sunflower', 'Leopard', 'Bell pepper', 'Harbor seal', 'Snake', 'Sewing machine', 'Goose', 'Helicopter', 'Seat belt', 'Coffee cup', 'Microwave oven', 'Hot dog', 'Countertop', 'Serving tray', 'Dog bed', 'Beer', 'Sunglasses', 'Golf ball', 'Waffle', 'Palm tree', 'Trumpet', 'Ruler', 'Helmet', 'Ladder', 'Office building', 'Tablet computer', 'Toilet paper', 'Pomegranate', 'Skirt', 'Gas stove', 'Cookie', 'Cart', 'Raven', 'Egg', 'Burrito', 'Goat', 'Kitchen knife', 'Skateboard', 'Salt and pepper shakers', 'Lynx', 'Boot', 'Platter', 'Ski', 'Swimwear', 'Swimming pool', 'Drinking straw', 'Wrench', 'Drum', 'Ant', 'Human ear', 'Headphones', 'Fountain', 'Bird', 'Jeans', 'Television', 'Crab', 'Microphone', 'Home appliance', \
'Snowplow', 'Beetle', 'Artichoke', 'Jet ski', 'Stationary bicycle', 'Human hair', 'Brown bear', 'Starfish', 'Fork', 'Lobster', 'Corded phone', 'Drink', 'Saucer', 'Carrot', 'Insect', 'Clock', 'Castle', 'Tennis racket', 'Ceiling fan', 'Asparagus', 'Jaguar', 'Musical instrument', 'Train', 'Cat', 'Rifle', 'Dumbbell', 'Mobile phone', 'Taxi', 'Shower', 'Pitcher', 'Lemon', 'Invertebrate', 'Turkey', 'High heels', 'Bust', 'Elephant', 'Scarf', 'Barrel', 'Trombone', 'Pumpkin', 'Box', 'Tomato', 'Frog', 'Bidet', 'Human face', 'Houseplant', 'Van', 'Shark', 'Ice cream', 'Swim cap', 'Falcon', 'Ostrich', 'Handgun', 'Whiteboard', 'Lizard', 'Pasta', 'Snowmobile', 'Light bulb', 'Window blind', 'Muffin', 'Pretzel', 'Computer monitor', 'Horn', 'Furniture', 'Sandwich', 'Fox', 'Convenience store', 'Fish', 'Fruit', 'Earrings', 'Curtain', 'Grape', 'Sofa bed', 'Horse', 'Luggage and bags', 'Desk', 'Crutch', 'Bicycle helmet', 'Tick', 'Airplane', 'Canary', 'Spatula', 'Watch', 'Lily', 'Kitchen appliance', 'Filing cabinet', 'Aircraft', 'Cake stand', 'Candy', 'Sink', 'Mouse', 'Wine', 'Wheelchair', 'Goldfish', 'Refrigerator', 'French fries', 'Drawer', 'Treadmill', 'Picnic basket', 'Dice', 'Cabbage', 'Football helmet', 'Pig', 'Person', 'Shorts', 'Gondola', 'Honeycomb', 'Doughnut', 'Chest of drawers', 'Land vehicle', 'Bat', 'Monkey', 'Dagger', 'Tableware', 'Human foot', 'Mug', 'Alarm clock', 'Pressure cooker', 'Human hand', 'Tortoise', 'Baseball glove', 'Sword', 'Pear', 'Miniskirt', 'Traffic sign', 'Girl', 'Roller skates', 'Dinosaur', 'Porch', 'Human beard', 'Submarine sandwich', 'Screwdriver', 'Strawberry', 'Wine glass', 'Seafood', 'Racket', 'Wheel', 'Sea lion', 'Toy', 'Tea', 'Tennis ball', 'Waste container', 'Mule', 'Cricket ball', 'Pineapple', 'Coconut', 'Doll', 'Coffee table', 'Snowman', 'Lavender', 'Shrimp', 'Maple', 'Cowboy hat', 'Goggles', 'Rugby ball', 'Caterpillar', 'Poster', 'Rocket', 'Organ', 'Saxophone', 'Traffic light', 'Cocktail', 'Plastic bag', 'Squash', 'Mushroom', 'Hamburger', 'Light switch', 'Parachute', 'Teddy bear', 'Winter melon', 'Deer', 'Musical keyboard', 'Plumbing fixture', 'Scoreboard', 'Baseball bat', 'Envelope', 'Adhesive tape', 'Briefcase', 'Paddle', 'Bow and arrow', 'Telephone', 'Sheep', 'Jacket', 'Boy', 'Pizza', 'Otter', 'Office supplies', 'Couch', 'Cello', 'Bull', 'Camel', 'Ball', 'Duck', 'Whale', 'Shirt', 'Tank', 'Motorcycle', 'Accordion', 'Owl', 'Porcupine', 'Sun hat', 'Nail', 'Scissors', 'Swan', 'Lamp', 'Crown', 'Piano', 'Sculpture', 'Cheetah', 'Oboe', 'Tin can', 'Mango', 'Tripod', 'Oven', 'Mouse', 'Barge', 'Coffee', 'Snowboard', 'Common fig', 'Salad', 'Marine invertebrates', 'Umbrella', 'Kangaroo', 'Human arm', 'Measuring cup', 'Snail', 'Loveseat', 'Suit', 'Teapot', 'Bottle', 'Alpaca', 'Kettle', 'Trousers', 'Popcorn', 'Centipede', 'Spider', \
'Sparrow', 'Plate', 'Bagel', 'Personal care', 'Apple', 'Brassiere', 'Bathroom cabinet', 'studio couch', 'Computer keyboard', 'Table tennis racket', 'Sushi', 'Cabinetry', 'Street light', 'Towel', 'Nightstand', 'Rabbit', 'Dolphin', 'Dog', 'Jug', 'Wok', 'Fire hydrant', 'Human eye', 'Skyscraper', 'Backpack', 'Potato', 'Paper towel', 'Lifejacket', 'Bicycle wheel', 'Toilet')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
             dict(
                 type=dataset_type,
                 ann_file='data/OID/annotations/oid_challenge_2019_train_bbox_train0.2@50.json',
                 img_prefix='data/OID/',
                 classes=classes_oid,
                 pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/objects365_train.2@10.json',
                img_prefix=data_root + 'train/',
                classes=classes,
                pipeline=train_pipeline),
            dict(
                type='RepeatDataset',
                times=1,
                dataset=dict(
                    type=dataset_type,
                    #ann_file='data/coco/annotations/instances_train2017_lvis.json',
                    #img_prefix='data/coco/train2017/',
                    ann_file='data/coco/annotations/instances_valminusminival2014.json',
                    img_prefix='data/coco/images/val2014/',
                    pipeline=train_pipeline))
            ]),
    #train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')

# optimizer
#optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0), 'roi_head':dict(lr_mult=0.1, decay_mult=1.0)  }) )

#optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


#optimizer = dict(type='AdamW', lr=0.00015, weight_decay=0.0001,  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}) )
#optimizer = dict(type='AdamW', lr=0.000015, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step', 
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12
