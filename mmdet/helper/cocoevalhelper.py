import copy
import datetime
import time
from collections import defaultdict
import numpy as np
import pickle

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
        cocoeval.ious = {(imgId, catId): computeIoUhelper(cocoeval,imgId, catId)
                         for imgId in p.imgIds for catId in catIds}
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


def computeIoUhelper(cocoeval, imgId, catId):
    p = cocoeval.params
    if p.useCats:
        gt = cocoeval._gts[imgId, catId]
        dt = cocoeval._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in cocoeval._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in cocoeval._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
        dt = dt[0:p.maxDets[-1]]

    if p.iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif p.iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')

    # compute iou between each dt and gt region
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d, g, iscrowd)
    return ious

def computeCentroids(cocoeval, iou_thr=0.75,save_dir='', save_path='centroids.pkl'):

    save_path = save_dir+save_path
    p = cocoeval.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    catIds = p.catIds if p.useCats else [-1]

    centroids = {catId: [] for catId in catIds}
    # cocoeval.ious = {(imgId, catId): computeIoUhelper(cocoeval, imgId, catId)
    #                  for imgId in p.imgIds for catId in catIds}
    for imgId in p.imgIds:
        for catId in catIds:
            if p.useCats:
                gt = cocoeval._gts[imgId, catId]
                dt = cocoeval._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in cocoeval._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in cocoeval._dts[imgId, cId]]
            if len(gt) == 0 or len(dt) == 0:
                continue
            inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in inds]
            if len(dt) > p.maxDets[-1]:
                dt = dt[0:p.maxDets[-1]]

            if p.iouType == 'segm':
                g = [g['segmentation'] for g in gt]
                d = [d['segmentation'] for d in dt]
                fea = [d['feature'] for d in dt]
            elif p.iouType == 'bbox':
                g = [g['bbox'] for g in gt]
                d = [d['bbox'] for d in dt]
            else:
                raise Exception('unknown iouType for iou computation')

            # compute iou between each dt and gt region
            iscrowd = [int(o['iscrowd']) for o in gt]
            ious = maskUtils.iou(d, g, iscrowd)
            max_ious = ious.max(axis=1)
            indexes = np.where(max_ious>iou_thr)[0]
            select_fea = [fea[i] for i in indexes]
            for v in select_fea:
                centroids[catId].append(v)

    # filehandler = open('scores.pkl', 'wb')
    # pickle.dump(centroids, filehandler)
    # print("Scores have been saved to: {}".format(save_path))

    for catId in catIds:
        if len(centroids[catId])>0:
            filehandler = open(save_dir+'scores_'+str(catId)+'.pkl', 'wb')
            pickle.dump(centroids[catId], filehandler)
            print("Scores of class {} have been saved to: {}".format(catId,save_path))
            centroids[catId] = np.sum(centroids[catId],axis=0)/len(centroids[catId])
        # print("class {} length is : {}".format(catId,len(centroids[catId])))

    # mm = pickle.load(open("centroids.pkl", "rb"))
    filehandler = open(save_path, 'wb')
    pickle.dump(centroids, filehandler)
    print("centroids have been saved to: {}".format(save_path))



def opensummarize(cocoEval):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter
    setting
    '''
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, F1=False, open_range=None):
        p = cocoEval.params
        iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        if F1:
            titleStr = 'Average F1Score'
            typeStr = '(F1)'

        if open_range is not None:
            titleStr.replace("Average",open_range)


        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [
            i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
        ]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if F1:
            precision = cocoEval.eval['precision']
            recall = cocoEval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                precision = precision[t]
                recall = recall[t]
            if open_range == "Known":
                precision = precision[:, :, :50, aind, mind]
                recall = recall[:, :50, aind, mind]
            elif open_range =="Unknown":
                precision = precision[:, :, 50:, aind, mind]
                recall = recall[:, 50:, aind, mind]
            else:
                precision = precision[:, :, :, aind, mind]
                recall = recall[:, :, aind, mind]

            if len(precision[precision > -1]) == 0:
                mean_precision = -1
            else:
                mean_precision = np.mean(precision[precision > -1])
            if len(recall[recall > -1]) == 0:
                mean_recall = -1
            else:
                mean_recall = np.mean(recall[recall > -1])
            mean_F1 = 2*mean_recall*mean_precision/(mean_recall+mean_precision+np.spacing(1))
            print(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_F1))
            return mean_F1

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = cocoEval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if open_range == "Known":
                s = s[:, :, :50, aind, mind]
            elif open_range =="Unknown":
                s = s[:, :, 50:, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = cocoEval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if open_range == "Known":
                s = s[:, :50, aind, mind]
            elif open_range =="Unknown":
                s = s[:, 50:, aind, mind]
            else:
                s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(
            iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                        mean_s))
        return mean_s

    def _summarizeDets():
        stats = np.zeros((15, ))
        # All
        stats[0] = _summarize(1)
        stats[1] = _summarize(0, maxDets=cocoEval.params.maxDets[2])
        stats[2] = _summarize(1,F1=True)
        # Known
        stats[3] = _summarize(1,open_range="Known")
        stats[4] = _summarize(0, maxDets=cocoEval.params.maxDets[2],open_range="Known")
        stats[5] = _summarize(1, F1=True,open_range="Known")
        #Unkown
        stats[6] = _summarize(1,open_range="Unknown")
        stats[7] = _summarize(0, maxDets=cocoEval.params.maxDets[2],open_range="Unknown")
        stats[8] = _summarize(1, F1=True,open_range="Unknown")
        return stats


    if not cocoEval.eval:
        raise Exception('Please run accumulate() first')
    iouType = cocoEval.params.iouType
    summarize = _summarizeDets
    cocoEval.stats = summarize()

