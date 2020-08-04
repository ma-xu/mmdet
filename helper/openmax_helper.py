import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.datasets.coco import CocoDataset
import warnings
from mmdet.helper.openmax import *
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Openmax Helper')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--centroids', default='/home/xuma/mmdet/work_dirs/mask_rcnn_osr50/centroids.pkl', help='centroids path')
    parser.add_argument('--score_path', default='/home/xuma/mmdet/work_dirs/mask_rcnn_osr50/scores_', help='score path')
    parser.add_argument('--save_path', default='/home/xuma/mmdet/work_dirs/mask_rcnn_osr50/', help='score path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--knownclass', type=int, default=50)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    warnings.filterwarnings('ignore')


    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)




    centroids = mmcv.load(args.centroids)
    print("Centroids loaded!!")

    dists=[]
    mavs = []
    for i in range(0,args.knownclass):
        catid = i+1
        if os.path.exists(args.score_path+str(catid)+'.pkl'):
            score = mmcv.load(args.score_path+str(catid)+'.pkl')
            score = [[s] for s in score]
        else:
            score=[[[0] * len(dataset.CLASSES)]]
        if len(score)>50000:
            score = score[0:50000]
        try:
            centroid = [centroids[catid].tolist()]
        except:
            centroid= [[0] * len(dataset.CLASSES)]
        dist = compute_channel_distances(centroid,score)
        # print("Finish calculating dist for class {}.".format(catid))
        dists.append(dist)
        mavs.append(centroid)
    mavs = np.array(mavs)
    weibull_model = fit_weibull(mavs,dists, dataset.CLASSES,200,'euclidean')

    filehandler = open(args.save_path+'weibull_model.pkl', 'wb')
    pickle.dump(weibull_model, filehandler)
    print("weibull model has been saved.")


    # rank, _ = get_dist_info()
    # if rank == 0:
    #     kwargs = {} if args.options is None else args.options
    #



if __name__ == '__main__':
    main()
