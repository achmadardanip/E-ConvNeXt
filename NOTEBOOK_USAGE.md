# E-ConvNeXt Flexible Training Notebook - User Guide

## Overview

This Jupyter notebook (`E_ConvNeXt_Flexible_Training.ipynb`) provides a comprehensive, flexible framework for training E-ConvNeXt models for both **Image Classification** and **Object Detection** tasks. It's optimized for offline production environments and laptop training with Intel i7 CPU and NVIDIA RTX 3050 GPU.

## Key Features

✅ **Dual Task Support**: Switch between image classification and object detection  
✅ **Flexible Dataset Configuration**: Easy integration with custom datasets  
✅ **Model Boosting**: Uses pre-trained E-ConvNeXt baseline for transfer learning  
✅ **H5 Export**: Save trained models in portable H5 format  
✅ **Laptop-Optimized**: Configured for efficient training on i7 + RTX 3050  
✅ **Offline Ready**: No internet required after initial setup  
✅ **Performance Monitoring**: Built-in benchmarking and visualization  

## Quick Start

### 1. Prerequisites

```bash
# Create conda environment
conda create -n econvnext python=3.8 -y
conda activate econvnext

# Install PaddlePaddle GPU version
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# Install dependencies
cd classification
pip install -r requirements.txt
cd ..
```

### 2. Prepare Your Dataset

#### For Image Classification:

Create your dataset structure:
```
dataset/custom_dataset/
├── train_list.txt      # Format: image_path label
├── val_list.txt        # Format: image_path label
└── images/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class2/
    └── ...
```

Example `train_list.txt`:
```
images/class1/img001.jpg 0
images/class1/img002.jpg 0
images/class2/img001.jpg 1
images/class2/img002.jpg 1
```

#### For Object Detection:

Use COCO format:
```
dataset/coco_format/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    │   ├── img001.jpg
    │   └── img002.jpg
    └── val/
        ├── img001.jpg
        └── img002.jpg
```

### 3. Configure the Notebook

Open `E_ConvNeXt_Flexible_Training.ipynb` in Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab E_ConvNeXt_Flexible_Training.ipynb
```

In **Section 2 (Configuration)**, modify the following:

```python
# Select your task
TASK = 'classification'  # or 'detection'

# Choose model size
MODEL_ARCH = 'mini'  # Options: 'mini' (7.6M), 'tiny' (13.2M), 'small' (19.4M)

# Dataset paths (for classification)
DATASET_ROOT = 'dataset/custom_dataset'
NUM_CLASSES = 10
TRAIN_LIST = 'dataset/custom_dataset/train_list.txt'
VAL_LIST = 'dataset/custom_dataset/val_list.txt'

# Training parameters (optimized for RTX 3050)
BATCH_SIZE = 32  # Adjust based on GPU memory
EPOCHS = 50
LEARNING_RATE = 0.0001
```

### 4. Run the Notebook

Execute cells sequentially:
1. **Cell 1-2**: Setup and imports
2. **Cell 3**: Configure your settings
3. **Cell 4-5**: Dataset preparation
4. **Cell 6**: Model creation with boosting
5. **Cell 7**: Training
6. **Cell 8**: Export to H5
7. **Cell 9-10**: Inference and benchmarking
8. **Cell 11**: Visualization

## Model Architecture Options

| Model | Parameters | FLOPs | Top-1 Acc (ImageNet) | Recommended Use |
|-------|-----------|-------|---------------------|-----------------|
| E-ConvNeXt-mini | 7.6M | 0.9G | 78.3% | Fast inference, resource-constrained |
| E-ConvNeXt-tiny | 13.2M | 2.0G | 80.6% | Balanced accuracy/speed |
| E-ConvNeXt-small | 19.4M | 3.1G | 81.9% | Maximum accuracy |

## Performance Optimization for RTX 3050

The notebook is pre-configured for optimal performance on RTX 3050 (4GB VRAM):

### Recommended Settings:

**For Image Classification:**
- Batch Size: 32-64
- Image Size: 224x224
- Mixed Precision: Enabled (auto)
- Expected Training Time: ~2-3 hours for 50 epochs on 10K images

**For Object Detection:**
- Batch Size: 8-16
- Image Size: 640x640
- Mixed Precision: Enabled (auto)
- Expected Training Time: ~4-6 hours for 50 epochs on 5K images

### Memory Management Tips:

```python
# If you encounter OOM (Out of Memory) errors:

# 1. Reduce batch size
BATCH_SIZE = 16  # for classification
BATCH_SIZE = 4   # for detection

# 2. Reduce image size
IMAGE_SIZE = 192  # instead of 224

# 3. Reduce number of workers
NUM_WORKERS = 2  # instead of 4
```

## Model Boosting

The notebook supports model boosting (transfer learning) from pre-trained E-ConvNeXt models:

```python
# Enable boosting
USE_PRETRAINED = True

# The model will automatically load pre-trained ImageNet weights
# and fine-tune on your dataset
```

### Where to Get Pre-trained Weights:

1. **From repository**: Check `classification/output/` for existing weights
2. **Download**: Use the official E-ConvNeXt pretrained models
3. **Train yourself**: Train on a large dataset first, then use for boosting

## Output Files

After training, you'll find:

```
output/
├── best_model_mini.pdparams          # Best model weights (PaddlePaddle format)
├── checkpoint_epoch_*.pdparams       # Training checkpoints
├── econvnext_mini_weights.pdparams   # Portable weights
├── training_history.json             # Training metrics
├── model_info.json                   # Model metadata
├── training_curves.png               # Loss/accuracy plots
└── inference_model/                  # Inference-optimized model
    ├── model.pdiparams
    ├── model.pdmodel
    └── model.pdiparams.info
```

## Inference Usage

### In the Notebook:

```python
# Load model
model = create_model(task='classification', arch='mini', num_classes=10)
model.set_state_dict(paddle.load('output/best_model_mini.pdparams'))

# Run inference
results, inf_time = inference(model, 'test_image.jpg')
print(f"Prediction: Class {results[0]['class_id']} ({results[0]['confidence']:.2%})")
print(f"Inference time: {inf_time:.2f}ms")
```

### Standalone Python Script:

```python
import paddle
from PIL import Image
import numpy as np

# Load model
model = create_model(task='classification', arch='mini', num_classes=10)
model.set_state_dict(paddle.load('output/best_model_mini.pdparams'))
model.eval()

# Preprocess image
image = Image.open('test.jpg').convert('RGB')
image = image.resize((224, 224))
image = np.array(image).astype('float32') / 255.0
mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
image = (image - mean) / std
image = image.transpose((2, 0, 1))[np.newaxis, :]

# Predict
with paddle.no_grad():
    output = model(paddle.to_tensor(image))
    pred = output.argmax(axis=1)[0].item()
    
print(f"Predicted class: {pred}")
```

## Benchmarking

The notebook includes performance benchmarking:

```python
# Run benchmark (automatically done in the notebook)
avg_time, fps = benchmark_model(model, val_loader, num_batches=50)

# Expected performance on RTX 3050:
# - E-ConvNeXt-mini: ~150-200 FPS (batch_size=32)
# - E-ConvNeXt-tiny: ~100-150 FPS (batch_size=32)
# - E-ConvNeXt-small: ~80-120 FPS (batch_size=32)
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8

# Reduce image size
IMAGE_SIZE = 192

# Clear cache between runs
import gc
import paddle
paddle.device.cuda.empty_cache()
gc.collect()
```

### Issue: Dataset Not Found

**Solution:**
- Check that paths in Section 2 are correct
- Ensure `train_list.txt` and `val_list.txt` exist
- Verify image paths in the list files are relative to `DATASET_ROOT`

### Issue: Slow Training

**Solution:**
```python
# Enable more workers (if CPU has enough cores)
NUM_WORKERS = 6

# Use smaller model
MODEL_ARCH = 'mini'

# Reduce epochs for testing
EPOCHS = 10
```

### Issue: Model Export Fails

**Solution:**
- Ensure model is trained first
- Check output directory has write permissions
- Verify PaddlePaddle version is 2.4.2 or later

## Advanced Usage

### Custom Data Augmentation:

Modify the dataset class in Section 4:

```python
def __getitem__(self, idx):
    img_path, label = self.samples[idx]
    image = Image.open(full_path).convert('RGB')
    
    # Add custom augmentation
    if self.is_train:
        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation
        angle = random.randint(-15, 15)
        image = image.rotate(angle)
    
    # Continue with preprocessing...
```

### Knowledge Distillation:

For model compression:

```python
# Load teacher model (larger, pre-trained)
teacher = create_model(arch='small', num_classes=NUM_CLASSES)
teacher.set_state_dict(paddle.load('teacher_weights.pdparams'))
teacher.eval()

# Train student model (smaller)
student = create_model(arch='mini', num_classes=NUM_CLASSES)

# Implement distillation loss
# (Add to training loop in Section 7)
```

### Multi-GPU Training:

```python
# Modify GPU configuration in Section 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use 2 GPUs

# Use distributed training (requires modification of training loop)
paddle.distributed.init_parallel_env()
model = paddle.DataParallel(model)
```

## Export and Deployment

### Convert to ONNX:

```python
import paddle2onnx

# Export to ONNX
onnx_model = paddle2onnx.export(
    model_file='output/inference_model/model.pdmodel',
    params_file='output/inference_model/model.pdiparams',
    save_file='output/model.onnx',
    opset_version=11
)
```

### TensorRT Optimization:

```python
# For maximum inference speed on NVIDIA GPUs
from paddle.inference import Config, create_predictor

config = Config('output/inference_model/model.pdmodel',
                'output/inference_model/model.pdiparams')
config.enable_use_gpu(1000, 0)
config.enable_tensorrt_engine()

predictor = create_predictor(config)
```

## Citation

If you use this notebook in your research, please cite:

```bibtex
@article{econvnext2024,
  title={E-ConvNeXt: A Lightweight and Efficient ConvNeXt Variant with Cross-Stage Partial Connections},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues and questions:
- Check this guide first
- Review the notebook comments
- Open an issue on GitHub
- Check PaddlePaddle documentation

## License

This notebook follows the same license as the E-ConvNeXt repository.

---

**Happy Training! 🚀**
