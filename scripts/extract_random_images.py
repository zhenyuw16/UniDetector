# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled data for coco train.

Example:
python3 object_detection/prepare_coco_data.py
"""

import argparse
import numpy as np
import json
import os


def prepare_data(seed, percent, ann):
    def _save_anno(name, images, annotations):
        """Save annotation
        """
        print('>> Processing data {} saved ({} images {} annotations)'.format(
            name, len(images), len(annotations)))
        new_anno = {}
        new_anno['images'] = images
        new_anno['annotations'] = annotations
        new_anno['categories'] = anno['categories']
        #print(len(set([i['category_id'] for i in annotations])) )
        #new_anno['info'] = anno['info']

        with open(name, 'w') as f:
            json.dump(new_anno, f)
        print('>> Data {}.json saved ({} images {} annotations)'.format(
            name, len(images), len(annotations)))

    np.random.seed(seed)
    anno = json.load(open(ann))

    image_list = anno['images']
    labeled_tot = int(percent / 100. * len(image_list))
    labeled_ind = np.arange(len(image_list))
    np.random.shuffle(labeled_ind)
    labeled_ind = labeled_ind[0:labeled_tot]

    print(len(labeled_ind))
    labeled_id = []
    labeled_images = []
    unlabeled_images = []
    labeled_ind = set(labeled_ind)
    print(len(labeled_ind))
    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i])
            labeled_id.append(image_list[i]['id'])
        else:
            unlabeled_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_id = set(labeled_id)
    labeled_annotations = []
    unlabeled_annotations = []
    for an in anno['annotations']:
        if an['image_id'] in labeled_id:
            labeled_annotations.append(an)
        else:
            unlabeled_annotations.append(an)

    # save labeled and unlabeled
    save_name = ann[:-5] + '.' + str(seed) + '@' + str(percent) + '.json'
    _save_anno(save_name, labeled_images, labeled_annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default=20)
    parser.add_argument('--ann', type=str)
    parser.add_argument('--seed', type=int, help='seed', default=1)

    args = parser.parse_args()
    prepare_data(args.seed, args.percent, args.ann)

