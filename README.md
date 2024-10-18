# Mamba-YOLO
Official pytorch implementation of “Mamba-YOLO：SSMs-based for Object Detection”

![Python 3.11](https://img.shields.io/badge/python-3.11-g)
![pytorch 2.3.0](https://img.shields.io/badge/pytorch-2.3.0-blue.svg)
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)

![](asserts/SOTACompare.png)

## Installation
``` shell
# pip install required packages
conda create -n mambayolo -y python=3.11
conda activate mambayolo
pip3 install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops wandb
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

## Training

```shell
python mbyolo_train.py --task train --data /home/dmovchan/repos/zorya/datasets/10_12_2024_polygon_finetune_nolast/data.yaml --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml --batch_size 48 --img 640 --device 0,1,2 --epochs 40 --optimizer AdamW --amp --project mamba-yolo --name mambayolo_b

```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba).