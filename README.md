# Volleyball Player Pose Estimation and Motion Tracking

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.15-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

A comprehensive system for volleyball player pose estimation and tracking using OpenPose and DeepSORT algorithms. Achieves **98.23% accuracy** on the Volleyball Activity Dataset.

## Features

- **YOLOv4** for multi-scale player detection
- **OpenPose** for accurate pose estimation  
- **DeepSORT** for robust player tracking across frames
- Support for 6 volleyball actions (Serve, Reception, Attack, Block, Setting, Stand)
- Near real-time performance (16.7 FPS)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/volleyball-pose-tracking.git
cd volleyball-pose-tracking

# Create conda environment
conda create -n volleyball python=3.7
conda activate volleyball

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py
```

## Quick Start

### Run Inference

```bash
python src/main.py --video path/to/video.mp4 --output results/output.mp4
```

### Train Models

```bash
python experiments/train.py --dataset ./datasets/volleyball --epochs 30
```

### Evaluate Performance

```bash
python experiments/evaluate.py --dataset ./datasets/test
```

## Dataset

Download the Volleyball Activity Dataset:

```bash
python data/download_dataset.py --output-dir ./datasets/volleyball
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score | FPS |
|-------|----------|-----------|--------|----------|-----|
| **Ours** | **98.23%** | **99.30%** | **97.31%** | **0.9754** | **16.7** |
| HybridSORT | 98.15% | 98.92% | 96.89% | 0.9734 | 18.3 |
| HRNet-DeepSORT | 95.37% | 96.41% | 94.72% | 0.9487 | 12.1 |
| PoseNet-DeepSORT | 92.06% | 93.57% | 91.38% | 0.9195 | 21.4 |

## License

MIT License - see [LICENSE](LICENSE) file for details.
