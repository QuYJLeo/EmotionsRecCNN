# EmotionsRecCNN

**Facial Emotion Recognition Based on Deep Convolutional Neural Networks with Attention Mechanisms**

![GitHub License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Environment Setup](#environment-setup)
6. [Data Preparation](#data-preparation)
7. [Training](#training)
8. [Testing & Inference](#testing--inference)
9. [ONNX Deployment](#onnx-deployment)
10. [Configuration](#configuration)
11. [Results & Performance](#results--performance)
12. [References](#references)

---

## Introduction

EmotionsRecCNN is a state-of-the-art facial emotion recognition system built on top of deep convolutional neural networks (CNNs). This project implements a novel **Residual Masking Network (ResMasking)** architecture, which combines the power of residual networks with attention mechanisms to achieve improved accuracy in emotion classification tasks.

The system supports 8 emotion categories:
- Anger
- Contempt
- Disgust
- Fear
- Happy
- Neural (Neutral)
- Sad
- Surprise

In addition to emotion recognition, the project also includes a lightweight face detection module (SlimNet) for real-time processing pipelines.

### Demo Result

![Emotion Recognition Demo](res.png)

*Figure: Real-time facial emotion recognition demonstration showing face detection bounding box and emotion probability visualization*

### Pre-trained Models

Download pre-trained models from Google Drive:  
[https://drive.google.com/drive/folders/1T4eZRJw3ln60dYt_wFYML2jimQSnwUeI](https://drive.google.com/drive/folders/1T4eZRJw3ln60dYt_wFYML2jimQSnwUeI)

---

## Features

### Core Features

- **ResMasking Architecture**: Innovative attention mechanism applied across different layers of ResNet
- **Multiple Backbones**: Support for various ResNet variants (ResNet-18, ResNet-34, ResNet-50, etc.)
- **Data Augmentation**: Built-in augmentation pipeline using imgaug
- **Optimized Training**: Supports RAdam, AdamW, and other modern optimizers
- **Early Stopping & LR Scheduling**: ReduceLROnPlateau with plateau detection
- **Real-time Inference**: Includes both PyTorch and ONNX inference support
- **Face Detection Pipeline**: Integrated SlimNet lightweight face detector

### Technical Highlights

- Attention-based feature refinement
- Dropout regularization for improved generalization
- Batch normalization and residual connections
- Comprehensive training metrics logging
- Checkpoint management

---

## Project Structure

```
EmotionsRecCNN/
├── README.md                 # Project documentation (English)
├── README-CN.md             # Project documentation (Chinese)
├── LICENSE                  # License file
├── checkpoints/             # Model checkpoints and pre-trained weights
├── configs/
│   └── ck_config.json       # Training configuration file
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── resnet.py            # ResNet base models
│   ├── resmasking.py        # ResMasking architecture
│   ├── resmasking_naive.py  # Naive version of ResMasking
│   ├── masking.py           # Masking network components
│   ├── basic_layers.py      # Basic building blocks
│   └── utils.py             # Model utilities
├── trainers/                # Training pipelines
│   ├── __init__.py
│   └── mydataset_trainer.py # Trainer class
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── radam.py             # RAdam optimizer implementation
│   ├── generals.py          # General utilities
│   ├── utils.py             # Additional utilities
│   ├── augmenters/
│   │   └── augment.py       # Data augmentation pipelines
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── mydataset.py     # Dataset class
│   └── metrics/
│       ├── __init__.py
│       ├── metrics.py       # Classification metrics
│       └── segment_metrics.py # Segmentation metrics
└── tools/                   # Tools and scripts
    ├── _train.py            # Training script
    ├── _test.py             # Testing/inference script (with camera)
    ├── Fps.py               # FPS calculation utility
    ├── net_slim.py          # SlimNet face detector
    ├── to_onnx.py           # Convert model to ONNX format
    ├── use_onnx.py          # ONNX inference script
    └── util.py              # Utility functions for detection
```

---

## Model Architecture

### ResMasking Network

The ResMasking network is the core of this project, built upon ResNet-34 with attention masking modules applied at each layer.

#### Architecture Overview

```
Input Image (3x224x224)
    ↓
Conv7x7 + BatchNorm + ReLU + MaxPool
    ↓
Layer 1 (64 channels) + Masking4 → x*(1+m)
    ↓
Layer 2 (128 channels) + Masking3 → x*(1+m)
    ↓
Layer 3 (256 channels) + Masking2 → x*(1+m)
    ↓
Layer 4 (512 channels) + Masking1 → x*(1+m)
    ↓
Adaptive AvgPool + Flatten + Dropout(0.4)
    ↓
Linear Layer (512 → 8 classes)
    ↓
Softmax → Emotion Probabilities
```

#### Residual Masking Modules

The project implements four different masking networks with varying depths:

| Masking Module | Depth | Encoder-Decoder Structure |
|----------------|-------|---------------------------|
| Masking1       | 1     | Single block, no downsampling |
| Masking2       | 2     | 2-level downsampling + upsampling |
| Masking3       | 3     | 3-level downsampling + upsampling |
| Masking4       | 4     | 4-level downsampling + upsampling |

Each masking module uses U-Net-like encoder-decoder architecture with residual blocks to generate attention maps.

### SlimNet (Face Detector)

A lightweight face detection network based on depthwise separable convolutions for real-time applications.

---

## Environment Setup

### Requirements

- Python 3.7+
- PyTorch 1.0+
- torchvision
- CUDA (recommended for training/inference)
- OpenCV (cv2)
- NumPy
- imgaug
- onnx (optional, for ONNX export)
- onnxruntime (optional, for ONNX inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EmotionsRecCNN
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   conda create -n emotions python=3.8
   conda activate emotions
   ```

3. **Install PyTorch**
   - For CUDA:
     ```bash
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
   - For CPU only:
     ```bash
     conda install pytorch torchvision torchaudio cpuonly -c pytorch
     ```

4. **Install other dependencies**
   ```bash
   pip install opencv-python numpy imgaug
   pip install onnx onnxruntime-gpu  # optional, for ONNX
   ```

---

## Data Preparation

### Dataset Structure

Organize your dataset in the following directory structure:

```
your_dataset_path/
├── Anger/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Contempt/
├── Disgust/
├── Fear/
├── Happy/
├── Neural/
├── Sad/
└── Surprise/
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Other common image formats supported by OpenCV

### Data Augmentation

The training pipeline includes built-in data augmentation using `imgaug`:

```python
seg = iaa.Sequential([
    iaa.Fliplr(p=0.5),              # Horizontal flip with 50% probability
    iaa.Affine(rotate=(-30, 30)),  # Random rotation (-30° to +30°)
])
```

**Note**: Currently, data augmentation is applied during training but commented out in the provided code. Uncomment additional augmentations as needed.

---

## Training

### Configuration

Before training, edit the configuration file in `configs/ck_config.json`:

```json
{
    "image_size": 224,
    "in_channels": 3,
    "num_classes": 8,
    "arch": "resmasking_dropout",
    "lr": 0.0001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 10,
    "num_workers": 0,
    "device": "cuda:0",
    "max_epoch_num": 80,
    "max_plateau_count": 20,
    "plateau_patience": 4,
    "data_path": "path/to/your/dataset",
    "weight_path": ""
}
```

### Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `arch` | Model architecture to use (`resmasking_dropout` recommended) |
| `lr` | Initial learning rate |
| `batch_size` | Batch size for training/validation |
| `device` | Device to use (`cuda:0` or `cpu`) |
| `max_epoch_num` | Maximum number of training epochs |
| `max_plateau_count` | Stop training after this many epochs without improvement |
| `plateau_patience` | Epochs to wait before reducing learning rate |
| `data_path` | Path to your dataset directory |
| `weight_path` | Path to pre-trained weights (optional, for fine-tuning) |

### Starting Training

```bash
cd tools
python _train.py
```

### Training Process

The training pipeline includes:

1. **Data Loading**: Automatically splits dataset into train (70%) and validation (30%)
2. **Training Loop**:
   - Forward pass
   - Loss calculation (CrossEntropyLoss)
   - Backward pass
   - Optimizer step (Adam)
3. **Validation**: Evaluate on validation set after each epoch
4. **Checkpointing**: Saves best model based on validation accuracy
5. **Learning Rate Scheduling**: Uses ReduceLROnPlateau
6. **Early Stopping**: Stops if no improvement after `max_plateau_count` epochs

### Training Output

- Best model weights saved to `saved/checkpoints/`
- Console output showing training/validation loss and accuracy
- Best performance metrics recorded

---

## Testing & Inference

### Real-time Webcam Inference

The `_test.py` script provides real-time emotion recognition from webcam:

```bash
cd tools
python _test.py
```

#### Features

- Real-time face detection using SlimNet
- Emotion classification using ResMasking
- Visualizes bounding boxes, emotion labels, and probability bars
- Display 8 emotion probabilities
- FPS monitoring (optional)

#### How It Works

1. Captures video from default webcam (device 0)
2. Detects faces using SlimNet
3. For each detected face:
   - Crop and resize to 224x224
   - Normalize and pass through ResMasking model
   - Apply softmax to get probabilities
4. Draws visualization on frame
5. Displays result

#### Customization

- To use a video file instead of webcam, modify line 45 in `_test.py`:
  ```python
  cap = 'path/to/your/video.avi'
  ```
- Adjust face detection confidence threshold (line 62):
  ```python
  inds = np.where(scores > 0.999)[0]
  ```

### Batch Inference

For batch inference on images, you can extend the provided code or create a custom script.

---

## ONNX Deployment

### Export to ONNX

Use `to_onnx.py` to convert the trained PyTorch model to ONNX format:

```bash
cd tools
python to_onnx.py
```

### ONNX Inference

Use `use_onnx.py` for inference with ONNX Runtime:

```bash
cd tools
python use_onnx.py
```

### Benefits of ONNX Deployment

- Framework-agnostic deployment
- Optimized inference performance
- Deployment to various platforms (mobile, edge devices, web)
- Quantization support for further optimization

---

## Configuration

### Emotion Categories

The 8 emotion categories with their labels:

```python
Film_DICT = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neural",
    6: "Sad",
    7: "Surprise",

    "Anger": 0,
    "Contempt": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happy": 4,
    "Neural": 5,
    "Sad": 6,
    "Surprise": 7,
}
```

### Model Variants

The project supports multiple model variants:

1. **ResMasking** - Base ResMasking architecture
2. **ResMasking Dropout** - ResMasking with dropout (0.4) before final layer
3. **ResMasking Naive** - Naive version without mask application (for comparison)

---

## Results & Performance

### Expected Performance

With proper training on a good quality dataset, you can expect:

- **Training Accuracy**: >95%
- **Validation Accuracy**: >85% (depends on dataset quality and size)

### Performance Tips

1. **Data Quality**: High-quality, balanced datasets yield best results
2. **Data Augmentation**: Experiment with additional augmentations (blur, noise, color jitter)
3. **Learning Rate**: Adjust based on your dataset size
4. **Batch Size**: Larger batch sizes (if memory permits) can improve stability
5. **Pre-trained Weights**: Consider fine-tuning from a pre-trained model on similar tasks

### Inference Speed

- **GPU (NVIDIA)**: ~30-60 FPS (including face detection)
- **CPU**: ~5-10 FPS

---

## References

### Papers

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. CVPR.
   - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

2. Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2019). **On the Variance of the Adaptive Learning Rate and Beyond**. ICLR.
   - [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) (RAdam)

3. Loshchilov, I., & Hutter, F. (2019). **Decoupled Weight Decay Regularization**. ICLR.
   - [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) (AdamW)

### Related Projects

- ResNet implementation from torchvision
- imgaug library for data augmentation
- ONNX for model export

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Contributors to imgaug and ONNX projects
- The computer vision research community for continued innovation

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---

## Contact

For questions or issues, please open an Issue on GitHub.

---

**Happy training!** 🎭😊
