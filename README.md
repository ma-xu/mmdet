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
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Data Preparation

