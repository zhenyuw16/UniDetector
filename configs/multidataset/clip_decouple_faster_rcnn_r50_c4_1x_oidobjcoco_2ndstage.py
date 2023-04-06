_base_ = [
    '../_base_/default_runtime.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        style='pytorch'),
    roi_head=dict(
        type='StandardRoIHead',
        shared_head=dict(type='CLIPResLayer', layers=[3, 4, 6, 3]),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='BBoxHeadCLIPPartitioned',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            with_cls=False,
            reg_class_agnostic=True,
            zeroshot_path=['./clip_embeddings/coco_clip_a+cname_rn50_manyprompt.npy', './clip_embeddings/objects365_clip_a+cname_rn50_manyprompt.npy', './clip_embeddings/oid_clip_a+cname_rn50_manyprompt.npy'],
            cat_freq_path=[None, 'data/objects365/annotations/zhiyuan_objv2_train.1@3.5_cat_info.json', 'data/openimages/oid_challenge_2019_train_bbox.1@4.5_cat_info.json'],
            num_classes=500,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  ###### the loss_cls here is not appicable in training
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
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
            score_thr=0.0001,
            nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
            max_per_img=100)))


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




# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 400), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposals']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
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
            dict(type='ToTensor', keys=['proposals']),
            dict(
                type='ToDataContainer',
                fields=[dict(key='proposals', stack=False)]),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]

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
                proposal_file='rp_train_0.pkl',
                dataset_id=0),
            dict(
                type=dataset_type,
                ann_file='data/objects365/annotations/zhiyuan_objv2_train.1@3.5.json',
                img_prefix='data/objects365/train/',
                classes=classes_obj365,
                pipeline=train_pipeline,
                proposal_file='rp_train_1.pkl',
                dataset_id=1),
            dict(
                type=dataset_type,
                ann_file='data/openimages/oid_challenge_2019_train_bbox.1@4.5.json',
                img_prefix='data/openimages/',
                classes=classes_oid,
                pipeline=train_pipeline,
                proposal_file='rp_train_2.pkl',
                dataset_id=2)
            ]),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline,
        proposal_file='rp_val.pkl'
        ),
    test=dict(
        type=dataset_type,
        ann_file = 'data/coco/annotations/instances_val2017.json',
        img_prefix = 'data/coco/val2017/',
        pipeline=test_pipeline,
        proposal_file='rp_val.pkl'
        ))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0), 'roi_head':dict(lr_mult=0.1, decay_mult=1.0)  }) )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


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
    type='EpochBasedRunner', max_epochs=12) 

# fp16 = dict(loss_scale=32.)
