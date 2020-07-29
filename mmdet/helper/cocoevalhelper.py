import copy
import datetime
import time
from collections import defaultdict

import numpy as np

# from . import mask as maskUtils
from pycocotools.coco import maskUtils


def _prepare(cocoeval):
    '''
    Prepare ._gts and ._dts for evaluation based on params
    :return: None
    '''

    def _toMask(anns, coco):
        # modify ann['segmentation'] by reference
        for ann in anns:
            rle = coco.annToRLE(ann)
            ann['segmentation'] = rle

    p = cocoeval.params
    if p.useCats:
        gts = cocoeval.cocoGt.loadAnns(
            cocoeval.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = cocoeval.cocoDt.loadAnns(
            cocoeval.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
    else:
        gts = cocoeval.cocoGt.loadAnns(cocoeval.cocoGt.getAnnIds(imgIds=p.imgIds))
        dts = cocoeval.cocoDt.loadAnns(cocoeval.cocoDt.getAnnIds(imgIds=p.imgIds))

    # convert ground truth to mask if iouType == 'segm'
    if p.iouType == 'segm':
        _toMask(gts, cocoeval.cocoGt)
        _toMask(dts, cocoeval.cocoDt)
    # set ignore flag
    for gt in gts:
        gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
        gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        if p.iouType == 'keypoints':
            gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
    cocoeval._gts = defaultdict(list)  # gt for evaluation
    cocoeval._dts = defaultdict(list)  # dt for evaluation
    for gt in gts:
        cocoeval._gts[gt['image_id'], gt['category_id']].append(gt)
    for dt in dts:
        cocoeval._dts[dt['image_id'], dt['category_id']].append(dt)
    cocoeval.evalImgs = defaultdict(
        list)  # per-image per-category evaluation results
    cocoeval.eval = {}  # accumulated evaluation results


def evaluate(cocoeval):
    '''
    Run per image evaluation on given images and store results
     (a list of dict) in self.evalImgs
    :return: None
    '''
    tic = time.time()
    print('Running per image evaluation...')
    p = cocoeval.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.
              format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    cocoeval.params = p

    _prepare(cocoeval)
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = cocoeval.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = cocoeval.computeOks
    cocoeval.ious = {(imgId, catId): computeIoU(imgId, catId)
                 for imgId in p.imgIds for catId in catIds}

    evaluateImg = cocoeval.evaluateImg
    maxDet = p.maxDets[-1]
    cocoeval.evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds
        for areaRng in p.areaRng for imgId in p.imgIds
    ]
    cocoeval._paramsEval = copy.deepcopy(cocoeval.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))