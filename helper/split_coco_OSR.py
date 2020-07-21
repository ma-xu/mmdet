# This script converts image annotations in a val/test into a RLE format json dataset
# This script converts COCO annotations for Open set Recognition (OSR) task.
# For Training, we keep the first 20 classes.
# For validation, we convert the last 60 classes to unknown, with index of 21.

import os
import glob
import argparse
import json
import numpy as np
from scipy.misc import imread
from pycocotools import mask as COCOmask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation demo')
    parser.add_argument('--known_num', default=20)
    parser.add_argument('--ann_dir', default='C:\\Users\\xuma\Downloads')


    args = parser.parse_args()
    return args


def convert(args):
    rnd = np.random.RandomState(42)
    # Make sure generating identy known classeses.
    known_classes = rnd.choice(np.arange(1,80), args.known_num, replace=False)

    Training_parser = True
    Validation_parser = True

    if Training_parser:
        training_ann_path = os.path.join(args.ann_dir,'instances_train2017.json')
        training_ann = json.load(open(training_ann_path, 'r'))
        info = training_ann["info"]
        licenses = training_ann["licenses"]
        images = training_ann["images"]
        annotations = training_ann["annotations"]
        annotations_OSR = []
        categories = training_ann["categories"]


        #dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        print("Start processing {} annotations".format(len(annotations)))
        ann_id = 0
        for i in range(0, len(annotations)):
            if i % 10000 == 0:
                print('# Training annotations processed: {}'.format(i))
            ann = annotations[i]
            if ann["category_id"] not in known_classes:
                continue
            ann['id'] = ann_id
            ann_id = ann_id + 1
            ann['original_category_id'] = ann['category_id']
            ann['category_id'] = np.where(known_classes==ann['original_category_id'])[0][0] + 1
            annotations_OSR.append(ann)

        data_out = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': images,
            'annotations': annotations_OSR
        }
        with open(os.path.join(args.ann_dir,'instances_train2017_OSR.json'), 'w') as f:
            json.dump(data_out, f)

    if Validation_parser:
        validation_ann_path = os.path.join(args.ann_dir, 'instances_val2017.json')
        validation_ann = json.load(open(validation_ann_path, 'r'))
        info = validation_ann["info"]
        licenses = validation_ann["licenses"]
        images = validation_ann["images"]
        annotations = validation_ann["annotations"]
        annotations_OSR = []
        categories = validation_ann["categories"]

        print("Start processing {} annotations".format(len(annotations)))
        ann_id = 0

        for i in range(0, len(annotations)):
            if i % 5000 == 0:
                print('# Validation annotations processed: {}'.format(i))
            ann = annotations[i]
            ann['original_category_id'] = ann['category_id']
            if ann["category_id"] not in known_classes:
                ann['category_id'] =args.known_num+1
            else:
                ann['category_id'] = np.where(known_classes == ann['original_category_id'])[0][0] + 1
            ann['id'] = ann_id
            ann_id = ann_id + 1

            annotations_OSR.append(ann)

        data_out = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': images,
            'annotations': annotations_OSR
        }
        with open(os.path.join(args.ann_dir,'instances_val2017_OSR.json'), 'w') as f:
            json.dump(data_out, f)


if __name__ == '__main__':
    args = parse_args()
    convert(args)