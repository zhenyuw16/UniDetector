# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OlnRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # Use a single anchor per location.
            scales=[8],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1.0,),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', linear=True, loss_weight=10.0),
        objectness_type='Centerness',
        loss_objectness=dict(type='L1Loss', loss_weight=1.0),
        ),
    roi_head=dict(
        type='OlnRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',  # 'BoxIoU' or 'Centerness'
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
            alpha=0.3,
            )),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            # Objectness assigner and sampler 
            objectness_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            objectness_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
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
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.0,  # No nms
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            #nms=dict(type='nms', iou_threshold=0.7),
            nms=dict(type='nms', iou_threshold=0.9),
            max_per_img=1500,
            )
    ))

# Dataset
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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

classes_obj365 = ['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 
'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 
'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 
'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 
'Van', 'Couch', 'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 
'Camera', 'Canned', 'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 
'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 
'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 
'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 
'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 
'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 
'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 
'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 
'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 
'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter', 'Bathtub', 'Wheelchair', 
'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket', 'Folder',
'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 
'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy',
'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 
'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign',
'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 
'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 
'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 
'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 
'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 
'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis ']


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
                ann_file='data/coco/annotations/instances_valminusminival2014.json',
                img_prefix='data/coco/images/val2014',
                pipeline=train_pipeline,
                class_agnostic=True),
            dict(
                type=dataset_type,
                ann_file='data/objects365/annotations/zhiyuan_objv2_train.1@3.5.json',
                img_prefix='data/objects365/train/',
                classes=classes_obj365,
                pipeline=train_pipeline,
                class_agnostic=True),
            dict(
                type=dataset_type,
                ann_file='data/openimages/oid_challenge_2019_train_bbox.1@4.5.json',
                img_prefix='data/openimages/',
                classes=classes_oid,
                pipeline=train_pipeline,
                class_agnostic=True)
            ]),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file='data/coco/annotations/instances_val2017.json',
        # img_prefix='data/coco/val2017/',
        ann_file=data_root + 'annotations/instances_valminusminival2014.json',
        img_prefix=data_root + 'images/val2014/',
        pipeline=test_pipeline))

evaluation = dict(interval=20, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# fp16 = dict(loss_scale=32.)