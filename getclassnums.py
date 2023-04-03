'''
import pickle
from mmdet.datasets import build_dataset
dataset_type = 'LVISV05Dataset'  
data_root = 'data/lvis_v0.5/' 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
train=dict(
    type=dataset_type,
    ann_file='data/lvis_v0.5/annotations/lvis_v0.5_train.json',
    img_prefix=data_root + 'train2017/', pipeline=train_pipeline)

dataset = build_dataset(train)

cnum = []
for i in range(1230):
    cnum.append({'image_info_id':[], 'img_id':[], 'category_id':i+1, 'isntance_count':0 } )

num_images = len(dataset)
for idx in range(num_images):
    cat_ids = set(dataset.get_cat_ids(idx))
    for cat_id in cat_ids:
        cnum[cat_id-1]['image_info_id'].append(idx)
        cnum[cat_id-1]['img_id'].append(dataset.data_infos[idx]['id'])
        cnum[cat_id-1]['isntance_count'] += 1

pickle.dump(cnum, open('class_to_imageid_and_inscount.pkl','wb'))
'''

import pickle
import json
import numpy as np
b = json.load(open('b.bbox.json'))
#gt = json.load(open('data/coco/zero-shot/instances_val2017_all_2.json'))
#cid = [i['id'] for i in gt['categories']]
cnum = np.zeros((1230))
#cnum = np.zeros((1203))
#cnum = np.zeros((65))
for i in range(len(b)):
    cnum[b[i]['category_id']-1] += 1
    #cnum[cid.index(b[i]['category_id'])] += 1


pickle.dump(cnum, open('cnum5.pkl','wb'))
