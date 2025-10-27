# E-ConvNeXt: A Lightweight and Efficient ConvNeXt Variant with Cross-Stage Partial Connections

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PaddlePaddle 2.4+](https://img.shields.io/badge/PaddlePaddle-2.4+-orange.svg)](https://www.paddlepaddle.org.cn/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**🚀 [Quick Start](QUICKSTART.md)** | **📓 [Jupyter Notebook](E_ConvNeXt_Flexible_Training.ipynb)** | **📖 [Usage Guide](NOTEBOOK_USAGE.md)** | **🧪 [Testing](TESTING.md)**

## 📑 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## ✨ Features

🎯 **Flexible Training**: Support for both image classification and object detection tasks  
⚡ **Efficient Architecture**: Up to 80% reduction in network complexity compared to ConvNeXt  
🚀 **Optimized for Laptops**: Configured for RTX 3050 GPU and Intel i7 CPU  
📊 **Pre-trained Models**: Transfer learning with ImageNet pre-trained weights  
💾 **Multiple Export Formats**: Export to H5, ONNX, and PaddlePaddle inference format  
🔌 **Offline Ready**: Full support for offline production environments  
📓 **Interactive Notebook**: Complete Jupyter notebook for easy experimentation  

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended for Beginners)

```bash
# Install dependencies
pip install paddlepaddle-gpu jupyter

# Launch notebook
jupyter lab E_ConvNeXt_Flexible_Training.ipynb
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Option 2: Command Line Training

For image classification:
```bash
cd classification
python tools/train.py -c ppcls/configs/ImageNet/ConvNext/E-ConvNext_mini.yaml
```

For object detection:
```bash
cd detection
python tools/train.py -c configs/ppyoloe_E_ConvNeXt/ppyoloe_m_cspconvnext_mini_36e_coco.yml
```

## 🏗️ Model Architecture

E-ConvNeXt introduces three key innovations:

1. **CSPNet Integration**: Cross Stage Partial Network connections for 80% complexity reduction
2. **Optimized Structures**: Enhanced Stem and Block designs for better efficiency
3. **Channel Attention**: Replaces Layer Scale with more effective attention mechanism

### Available Models

| Model | Parameters | FLOPs | Top-1 Acc | Speed (RTX 3050)* |
|-------|-----------|-------|-----------|-------------------|
| E-ConvNeXt-mini | 7.6M | 0.9G | 78.3% | ~180 FPS |
| E-ConvNeXt-tiny | 13.2M | 2.0G | 80.6% | ~130 FPS |
| E-ConvNeXt-small | 19.4M | 3.1G | 81.9% | ~100 FPS |

*Batch size = 32, Image size = 224×224

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)** - Complete notebook guide (10K+ words)
- **[TESTING.md](TESTING.md)** - Testing and validation guide
- **[config_example.json](config_example.json)** - Example configuration file
- **[prepare_dataset.py](prepare_dataset.py)** - Dataset preparation helper

## 📊 Performance

<div align="center">
  <img src="Images/FLOPs_ACC.png" width="600px" />
</div>

## Classification Results
### Image Classification for [ImageNet-1K](https://www.image-net.org)

| Model                    | FLOPs | #Params | Resolution | Top-1 |
|--------------------------|:-----:|:-------:|:----------:|:-----:|
| E-ConvNeXt-mini           | 0.9G  |  7.6M   | 224 x 224  | 78.3  |
| E-ConvNeXt-tiny           | 2.0G  |  13.2M   | 224 x 224  | 80.6  |
| E-ConvNeXt-small           | 3.1G  |  19.4M   | 224 x 224  | 81.9  |

## Downstream Results
## Object Detection for underwater sonar images
[underwater sonor dataset link](https://github.com/violetweir/Sonor_dataset)

### E-ConvNeXt as the backbone for PP-YOLOE and YOLOv10

| PP-YOLO-E with different Backbones | FLOPs | mAP |
|--------|:-----:|:-------:|
|PP-YOLOE-S| 17.4|42.6|
|PP-YOLO-S E-ConvNeXt-mini | 20.56 | 50.6 |
|PP-YOLOE-L | 110.7 | 49.0 |
|PP-YOLOE-L E-ConvNeXt-Tiny | 82.6 |  51.3 |



| YOLOv10 with different Backbones | FLOPs | mAP |
|--------|:-----:|:-------:|
| YOLOv10-L | 120 |46.1|
| YOLOv10-M ConvNeXt-tiny | 94.3 | 50.1 |
|PP-YOLOE-L | 110.7 | 49.0 |
| YOLOv10-M E-ConvNeXt-Tiny | 69.9 |  51.3 |

## Object Detection for underwater optical images(DUO dataset)
[Detecting Underwater Objects (DUO) link](https://github.com/chongweiliu/DUO)

| YOLOv10 with different Backbones | FLOPs | mAP |
|--------|:-----:|:-------:|
| YOLOv10-L | 120 |56.4|
| YOLOv10-L E-ConvNeXt-tiny | 78.0 | 61.2 |

## 💻 Usage Options

### Option 1: Jupyter Notebook (Recommended)

For interactive training with flexible dataset support:
- **[E_ConvNeXt_Flexible_Training.ipynb](E_ConvNeXt_Flexible_Training.ipynb)** - Complete training pipeline
- **[NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)** - Detailed usage guide
- **[prepare_dataset.py](prepare_dataset.py)** - Dataset preparation helper

### Option 2: Command Line

For advanced users and production training:
- **Classification**: See [classification/README](classification/README.md)
- **Object Detection**: See [detection/README](detection/README.md)

## 📚 Additional Resources

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start guide
- **[TESTING.md](TESTING.md)** - Testing and validation guide
- **[config_example.json](config_example.json)** - Configuration template

## Acknowledgements

We are grateful, but not limited to, to the following knowledge bases and communities for their assistance in our research:

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

- [openi](https://openi.pcl.ac.cn/)

- [ai studio](https://aistudio.baidu.com/)


