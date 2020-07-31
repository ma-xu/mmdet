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


def parse_args():
    parser = argparse.ArgumentParser(description='Openmax Helper')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('out', default='', help='test config file path')
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

    outputs = mmcv.load(args.out)
    print("Output loaded!!")

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # result_files, tmp_dir = dataset.format_results(results=outputs, jsonfile_prefix='/home/xuma/mmdet/work_dirs/json/result_train')
            # print("format files dones!!!")
            #
            # print(result_files)
            # print(tmp_dir)
            # dataset.evaluate(outputs, args.eval, **kwargs)
            result_files = {'bbox': '/home/xuma/mmdet/result_train.bbox.json',
                            'proposal': '/home/xuma/mmdet/result_train.bbox.json',
                            'segm': '/home/xuma/mmdet/work_dirs/json/result_train'}
            dataset.evaluate2(result_files, args.eval, **kwargs)


if __name__ == '__main__':
    main()
