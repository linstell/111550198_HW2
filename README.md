# Digit Detection using Deformable DETR

## Introduction
This project implements a digit detection model using Deformable DETR with a ResNet-50 backbone. The task is to detect digit classes and bounding boxes in RGB images.

## Environment Setup
- Python 3.10
- PyTorch
- transformers
- pycocotools
- Pillow
- tqdm

Install dependencies:
```bash
pip install torch torchvision transformers pycocotools pillow tqdm
```
# Usage
## Training
```bash
python train_hf.py
```
## Inference
```bash
python infer_hf.py --checkpoint "checkpoints/deformable_detr_r50_sz512_q100_ep20_subset10000_lr1e4/best_model.pth" --test_dir test --out pred.json --image_size 512 --conf 0.05 --max_boxes 100
```
## Performance Snapshot
- Best validation mAP: 0.3567
- AP50: 0.672
- Model: Deformable DETR
- Backbone: ResNet-50 (pretrained backbone only)
