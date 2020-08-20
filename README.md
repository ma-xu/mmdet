# OpenMax with MMdetection(Mask RCNN)

## Requirement
- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- mmcv
(my runing environment: pytorch 1.6 CUDA10.1)

## Installation
install mmcv
```
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
```
install required libaries and mmdetection
```
cd mmdet
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Data Preparation
Download MSCOCO dataset to your data path (e.g., DATA/coco):
```
cd ~
cd DATA/coco # modify it if not
# download training data
wget http://images.cocodataset.org/zips/train2017.zip 
# download validation data
wget http://images.cocodataset.org/zips/val2017.zip
# download annotations 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
Unzip these files to train2017, val2017, and annotations folders.

Split annotations into known classes and unkown classes
```
# using python2; python3 would meet some problems due to json library
cd mmdet/helper
python2 split_coco_OSR.py --known_num 50 --ann_dir /path_to_coco_annotations
```
This command will generate two files in annotations folder: instances_train2017_OSR{known_num}.json and instances_val2017_OSR{known_num}.json



