# This script converts image annotations in a val/test into a RLE format json dataset
# This script converts COCO annotations for Open set Recognition (OSR) task.
# For Training, we keep the first 20 classes.
# For validation, we convert the last 60 classes to unknown, with index of 21.

import os
import glob
import argparse
import random
import json
import numpy as np
from scipy.misc import imread
from pycocotools import mask as COCOmask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation demo')
    parser.add_argument('--known_num', default=50)
    parser.add_argument('--ann_dir', default='/Users/melody/Downloads/annotations')


    args = parser.parse_args()
    return args


def convert(args):
    rnd = np.random.RandomState(42)
    # Make sure generating identy known classeses.
    original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # known_classes = rnd.choice(np.arange(1,80), args.known_num, replace=False)
    # known_classes = random.sample(original,args.known_num)
    known_classes = rnd.choice(original, args.known_num, replace=False).tolist()
    print(known_classes)

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
            if i % 50000 == 0:
                print('# Training annotations processed: {}'.format(i))
            ann = annotations[i]
            if ann["category_id"] not in known_classes:
                continue
            else:
                ann['id'] = ann_id
                ann_id = ann_id + 1
                # ann['original_category_id'] = ann['category_id']
                # ann['category_id'] = known_classes.index(ann['original_category_id']) + 1
                annotations_OSR.append(ann)

        data_out = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': images,
            'annotations': annotations_OSR
        }
        with open(os.path.join(args.ann_dir,'instances_train2017_OSR'+str(args.known_num)+'.json'), 'w') as f:
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
                # ann['category_id'] =args.known_num+1
                ann['category_id'] = 91
            else:
                ann['category_id'] = known_classes.index(ann['original_category_id']) + 1
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
        with open(os.path.join(args.ann_dir,'instances_val2017_OSR'+str(args.known_num)+'.json'), 'w') as f:
            json.dump(data_out, f)


if __name__ == '__main__':
    args = parse_args()
    convert(args)
