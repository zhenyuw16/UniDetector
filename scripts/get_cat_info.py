# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default='datasets/lvis/lvis_v1_train.json')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cats = data['categories']
    image_count = {x['id']: set() for x in cats}
    ann_count = {x['id']: 0 for x in cats}
    for x in data['annotations']:
        image_count[x['category_id']].add(x['image_id'])
        ann_count[x['category_id']] += 1
    num_freqs = {x: 0 for x in ['r', 'f', 'c']}
    for x in cats:
        x['image_count'] = len(image_count[x['id']])
        x['instance_count'] = ann_count[x['id']]
    image_counts = sorted([x['image_count'] for x in cats])
    out = cats # {'categories': cats}
    out_path = args.ann[:-5] + '_cat_info.json'
    print('Saving to', out_path)
    json.dump(out, open(out_path, 'w'))
    
