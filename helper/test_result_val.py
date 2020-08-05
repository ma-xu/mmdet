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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Lightweight MMDet test (and eval) a model from the results.pkl')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='/home/xuma/mmdet/work_dirs/mask_rcnn_osr50/output_val.pkl',
                        help='output result file in pickle format')
    parser.add_argument('--weibull', default='/home/xuma/mmdet/work_dirs/mask_rcnn_osr50/weibull_model.pkl',
                        help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    warnings.filterwarnings('ignore')

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

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

    weibull_model = mmcv.load(args.weibull)

    new_out= []
    for image in outputs:
        bboxes, segs, feas = image
        for fea in feas:
            for cat_fea in fea:
                if len(cat_fea)>0:
                    for roi_cat_fea in cat_fea:
                        so, _ = openmax(weibull_model, dataset.CLASSES, feas, 0.5, 3, "euclidean")
                        print(so)





        so, _ = openmax(weibull_model, dataset.CLASSES, feas, 0.5, 3, "euclidean")
        seg = np.argsort(segs, so)
        bbox = np.argsort(bboxes,so)
        fea = np.argsort(feas,so)
        sub_out = bbox,seg,fea
        new_out.append(sub_out)



    print("Output loaded!!")

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(new_out, **kwargs)
        if args.eval:

            dataset.evaluate3(new_out, args.eval, **kwargs)


if __name__ == '__main__':
    main()
