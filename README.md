# Spatial Transform Decoupling 

## Project Introduction

Spatial Transform Decoupling (STD) is a rotation-aware object detection method based on Vision Transformer (ViT). This approach employs spatial transform decoupling mechanism with separate network branches for predicting bounding box coordinates, dimensions, and rotation angles, significantly enhancing detection performance for rotated objects.

This project implements the STD algorithm based on the MMRotate framework, achieving state-of-the-art performance on benchmark datasets including DOTA-v1.0 and HRSC2016.

![Framework Diagram](./figures/framework1.PNG)

## Key Features

- **Advanced Architecture**: Vision Transformers for high-precision rotated object detection
- **Innovative Technology**: Spatial transform decoupling mechanism improves detection of rotated objects

- **Complete Toolchain**: Integrated tools for training, evaluation, and visualization

## Environment Setup

### Dependencies

- Python 3.7+
- PyTorch 1.7.0+
- MMCV 1.6.0+
- MMDetection 2.25.1+
- MMRotate 0.3.4+

### Installation Steps

```bash
# Create conda environment
conda create -n rotation python=3.7 -y
conda activate rotation

# Install PyTorch
conda install pytorch=1.8.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Install base dependencies
pip install openmim
mim install mmcv-full==1.6.1
mim install mmdet==2.25.1

# Install MMRotate
git clone https://github.com/open-mmlab/mmrotate.git  
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
cd ..

# Install additional dependencies
pip install timm apex

# Copy project files
git clone https://github.com/yuhongtian17/Spatial-Transform-Decoupling.git  
cp -r Spatial-Transform-Decoupling/mmrotate-main/* mmrotate/
```

## Data Preparation

### Dataset Structure

Organize your dataset as follows:

```
mmrotatemain/data/dxc/
  ├── train/
  │   ├── images/   # Training images
  │   └── annfiles/ # Training annotation files
  ├── val/
  │   ├── images/   # Validation images
  │   └── annfiles/ # Validation annotation files
  └── test/
      ├── images/   # Test images
      └── annfiles/ # Test annotation files
```

### Data Placement Steps

1. Place training images in `mmrotatemain/data/dxc/train/images/`
2. Place corresponding annotation files in `mmrotatemain/data/dxc/train/annfiles/`
3. Similarly, place validation and test data in corresponding directories

## Model Training

### Training Workflow Overview

The training workflow includes:
1. Parsing training parameters and config files
2. Initializing distributed training environment (if needed)
3. Creating working directories and logs
4. Building model, dataset and optimizer
5. Executing training loop with checkpoint saving
6. Validating model performance (optional)

### Configuration Details

Default config location: `mmrotatemain/configs/train_config.py`

#### 1. Dataset Configuration

```python
dataset_type = 'DOTADataset'
data_root = './mmrotatemain/data/dxc/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='DOTADataset',
        ann_file='./mmrotatemain/data/dxc/train/annfiles/',
        img_prefix='./mmrotatemain/data/dxc/train/images/',
        pipeline=train_pipeline,
        version='le90'),  # Angle representation version
    # Validation/test configurations...
)
```

#### 2. Model Configuration

```python
model = dict(
    type='RotatedimTED',
    proposals_dim=6,
    backbone=dict(
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depths=[2, 2, 20],
        # Additional backbone parameters...
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        # RPN parameters...
    ),
    roi_head=dict(
        type='OrientedStandardRoIHeadimTED',
        bbox_roi_extractor=[...],
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            # BBox head parameters...
            dc_mode_str_list=['', '', '', 'XY', '', 'A', '', 'WH'],
            am_mode_str_list=['', '', 'V', 'V', 'V', 'V', 'V', 'V'],
            rois_mode='rbbox'
        )
    ),
    # Training/testing configurations...
)
```

#### 3. Optimizer Configuration

```python
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='HiViTLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=20, layer_decay_rate=0.9)
)
```

### Training Methods

#### Method 1: Using launch script (Recommended)

```bash
cd Spatial-Transform-Decoupling-main
python mmrotatemain/start_train.py --work-dir=./work_dirs/my_training
```

#### Method 2: Using MMRotate native tools

```bash
cd Spatial-Transform-Decoupling-main
python mmrotatemain/tools/train.py mmrotatemain/configs/train_config.py --work-dir=./work_dirs/my_training
```

#### Distributed Training (Multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 mmrotatemain/tools/train.py mmrotatemain/configs/train_config.py --launcher pytorch
```

### Training Monitoring

Training logs include:
- Environment info (Python/CUDA versions)
- Config file contents
- Iteration losses (classification, regression, etc.)
- Learning rate changes
- Validation results (if enabled)

## Model Evaluation

### Performance Evaluation

```bash
cd Spatial-Transform-Decoupling-main
python evaluation/evaluate_model.py
```

Evaluation workflow:
1. Load config and model weights
2. Build test dataset and dataloader
3. Run inference
4. Calculate metrics (mAP, precision, recall)

### Result Visualization

#### Batch Visualization

```bash
python evaluation/visualize_results.py
```

Results saved in `./visualization_results` with ground truth (green boxes) and predictions (red boxes).

#### Single Image Evaluation

```bash
python evaluation/visualize_single_image.py --img /path/to/your/image.jpg --out output.jpg
```

Parameters:
- `--img`: Input image path
- `--out`: Output path (default: `output_result.jpg`)
- `--score-thr`: Detection threshold (default: 0.3)

## Core Concepts of STD

Spatial Transform Decoupling introduces key innovations:

1. **Spatial Transform Decoupling**:
   - Separate branches for position, size, and angle prediction
   - Solves angle sensitivity in rotated object detection

2. **Cascaded Activation Masks (CAMs)**:
   - Compute activation masks from regression parameters
   - Gradually enhance ROI features
   - Complements self-attention for better feature representation

3. **Vision Transformer Architecture**:
   - Leverages ViT's powerful feature extraction
   - Optimized for rotated object detection tasks

## Pretrained Models

Available models:

| Model Name | Download Link | Extraction Code |
|-----------|----------------|----------------|
| Imagenet MAE pretrained ViT-S | [Baidu Drive](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | STDC |
| Imagenet MAE pretrained ViT-B | [Official Link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth) | - |
| Imagenet MAE pretrained HiViT-B | [Baidu Drive](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | STDC |

## Frequently Asked Questions

### Path Issues
```markdown
**Problem**: Data loading failed with path error  
**Solution**:
1. Verify data paths in config file
2. Replace path separators with `\\` on Windows
3. Ensure dataset structure compliance
```

### Out of Memory
```markdown
**Problem**: CUDA out of memory error  
**Solution**:
1. Reduce `samples_per_gpu` in config
2. Use gradient accumulation
3. Lower image resolution
```

### Weight Loading Issues
```markdown
**Problem**: Model weight loading failed  
**Solution**:
1. Check model architecture compatibility
2. Verify complete model download
3. Use `strict=False` parameter for partial loading
```

## Acknowledgments

This project builds upon:
- [imTED](https://github.com/LiewFeng/imTED): Rotated object detection framework  
- [HiViT](https://github.com/zhangxiaosong18/hivit): Efficient Vision Transformers  
- [MMRotate](https://github.com/open-mmlab/mmrotate): Rotated detection toolkit  

Special thanks to [Xue Yang](https://yangxue0827.github.io/) for pioneering work in this field.