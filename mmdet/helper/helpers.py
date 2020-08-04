import copy
import itertools
import json
import os
import time
from collections import defaultdict
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO
from pycocotools.coco import maskUtils

def loadRes(coco, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in coco.dataset['images']]

    print('Loading and preparing results...')
    tic = time.time()
    if type(resFile) == str:
        anns= [ ]
        try:
            filepath = resFile + '.segm'  + '.json'
            anns = json.load(open(filepath))
        except:
            for i in range(0,1000000):
                filepath = resFile+'.segm'+str(i)+'.json'
                if os.path.exists(filepath):
                    print("loading {}".format(filepath))
                    anns_sub = json.load(open(filepath))
                    anns.extend(anns_sub)
                else:
                    print("Finish load all segm json files")
                    break
    elif type(resFile) == np.ndarray:
        anns = coco.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(coco.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set(
            [ann['image_id'] for ann in anns])
        res.dataset['images'] = [
            img for img in res.dataset['images'] if img['id'] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(
            coco.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(
            coco.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation
            # results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(
            coco.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x1 - x0) * (y1 - y0)
            ann['id'] = id + 1
            ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
    print('DONE (t={:0.2f}s)'.format(time.time() - tic))

    res.dataset['annotations'] = anns
    res.createIndex()
    print("Finish load cocoDt")
    return res