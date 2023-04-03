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
            type='BBoxHeadCLIP',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            with_cls=False,
            reg_class_agnostic=True,
            zeroshot_path='./clip_embeddings/lvis_v0.5_clip_a+cname_rn50_manyprompt.npy',
            num_classes=1230,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
            score_thr=0.0001, #5,
            nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
            max_per_img=300)))


# dataset settings
dataset_type = 'LVISV05Dataset'
data_root = 'data/lvis_v0.5/'
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0,
        dataset=dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/lvis_v0.5_train.json',
                img_prefix=data_root + 'train2017/',
                pipeline=train_pipeline,
                with_score=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

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
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12
