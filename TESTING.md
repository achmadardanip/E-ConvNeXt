# Testing Guide for E-ConvNeXt Flexible Training

This guide explains how to test the flexible training notebook and ensure everything works correctly.

## Prerequisites Check

Before running tests, verify your environment:

```bash
# Check Python version (should be 3.8+)
python --version

# Check if PaddlePaddle is installed
python -c "import paddle; print(f'PaddlePaddle {paddle.__version__}')"

# Check if GPU is available
python -c "import paddle; print(f'GPU Available: {paddle.is_compiled_with_cuda()}')"

# Check CUDA version (if GPU available)
nvidia-smi
```

## Test 1: Verify File Integrity

```bash
# Navigate to repository
cd E-ConvNeXt

# Check all required files exist
ls -l E_ConvNeXt_Flexible_Training.ipynb
ls -l NOTEBOOK_USAGE.md
ls -l QUICKSTART.md
ls -l prepare_dataset.py
ls -l config_example.json

# Validate notebook JSON structure
python -c "import json; json.load(open('E_ConvNeXt_Flexible_Training.ipynb'))"
echo "✓ Notebook is valid JSON"

# Validate Python scripts
python -m py_compile prepare_dataset.py
echo "✓ prepare_dataset.py is valid"
```

## Test 2: Test Dataset Preparation Script

### Create Sample Dataset Structure

```bash
# Create a small test dataset
mkdir -p test_dataset/images/{cat,dog}
mkdir -p test_dataset/output

# Download or copy some sample images
# For testing, you can use any images - just put them in cat/ and dog/ folders
```

### Run Dataset Preparation

```bash
# Test classification dataset preparation
python prepare_dataset.py \
    --task classification \
    --root-dir test_dataset/images \
    --output-dir test_dataset/output \
    --train-split 0.8 \
    --create-config

# Verify output files
ls -l test_dataset/output/train_list.txt
ls -l test_dataset/output/val_list.txt
ls -l test_dataset/output/class_names.txt
ls -l test_dataset/output/config_classification.json

echo "✓ Dataset preparation successful"
```

## Test 3: Test Model Architecture Import

```bash
# Test if model can be imported (requires PaddlePaddle)
cd classification
python << EOF
import sys
sys.path.insert(0, '.')

# Test import
from ppcls.arch.backbone.model_zoo.cspconvnext import CSPConvNeXt

# Create a small model instance for testing
print("Testing model creation...")
model = CSPConvNeXt(
    arch='mini',
    class_num=10,
    drop_path_rate=0.1,
    layer_scale_init_value=1e-6
)

print(f"✓ Model created successfully")
print(f"✓ Model type: {type(model)}")

# Test forward pass with dummy data
import paddle
dummy_input = paddle.randn([1, 3, 224, 224])
output = model(dummy_input)
print(f"✓ Forward pass successful")
print(f"✓ Output shape: {output.shape}")
EOF

cd ..
```

## Test 4: Test Notebook Cells (Manual)

Open the notebook and test each section:

```bash
# Start Jupyter
jupyter lab E_ConvNeXt_Flexible_Training.ipynb
```

### Test Checklist:

1. **Cell 1-2**: Installation and Setup
   - [ ] All imports work without errors
   - [ ] GPU detection works (if GPU available)

2. **Cell 3**: Configuration
   - [ ] Can modify TASK variable (classification/detection)
   - [ ] Can modify MODEL_ARCH (mini/tiny/small)
   - [ ] Can set custom dataset paths

3. **Cell 4**: Dataset Preparation
   - [ ] ClassificationDataset class loads data
   - [ ] DetectionDataset class loads COCO format
   - [ ] Dataset length is correct

4. **Cell 5**: Model Creation
   - [ ] Model loads successfully
   - [ ] Parameter count is displayed
   - [ ] Can load pretrained weights (if available)

5. **Cell 6**: Training Setup
   - [ ] Optimizer initializes correctly
   - [ ] Learning rate scheduler works
   - [ ] Loss function is correct

6. **Cell 7**: Training Loop (Optional - time consuming)
   - [ ] Training starts without errors
   - [ ] Progress bars display correctly
   - [ ] Loss decreases over epochs
   - [ ] Model saves checkpoints

7. **Cell 8**: Model Export
   - [ ] Export to inference format works
   - [ ] Model info JSON is created
   - [ ] No errors during export

8. **Cell 9**: Inference
   - [ ] Single image inference works
   - [ ] Inference time is reasonable
   - [ ] Results are formatted correctly

9. **Cell 10**: Benchmarking
   - [ ] Benchmark runs without errors
   - [ ] FPS is calculated correctly
   - [ ] Results match expected performance

10. **Cell 11**: Visualization
    - [ ] Training curves plot correctly
    - [ ] Plot is saved to file

## Test 5: End-to-End Integration Test

This tests the complete workflow with a tiny dataset:

```bash
# Create mini test dataset (10 images per class)
mkdir -p mini_test/images/{class0,class1}

# Copy or create 10 images in each folder
# (For quick testing, you can duplicate images)

# Prepare dataset
python prepare_dataset.py \
    --task classification \
    --root-dir mini_test/images \
    --output-dir mini_test \
    --train-split 0.8

# Create a simple test script
cat > test_training.py << 'EOF'
import sys
import os
sys.path.insert(0, 'classification')

import paddle
from ppcls.arch.backbone.model_zoo.cspconvnext import CSPConvNeXt

# Set to CPU for testing (faster on small data)
paddle.set_device('cpu')

print("Creating model...")
model = CSPConvNeXt(
    arch='mini',
    class_num=2,
    drop_path_rate=0.1,
    layer_scale_init_value=1e-6
)

print("Testing forward pass...")
dummy_input = paddle.randn([1, 3, 224, 224])
output = model(dummy_input)

print(f"✓ Output shape: {output.shape}")
print(f"✓ Expected shape: [1, 2]")
assert output.shape == [1, 2], "Output shape mismatch!"

print("\n✓ All tests passed!")
EOF

python test_training.py
rm test_training.py
```

## Test 6: Performance Benchmarks

Test inference speed on your hardware:

```bash
cat > benchmark_test.py << 'EOF'
import sys
sys.path.insert(0, 'classification')

import paddle
import time
import numpy as np
from ppcls.arch.backbone.model_zoo.cspconvnext import CSPConvNeXt

# Create model
model = CSPConvNeXt(arch='mini', class_num=1000)
model.eval()

# Warm up
for _ in range(10):
    dummy_input = paddle.randn([1, 3, 224, 224])
    _ = model(dummy_input)

# Benchmark
times = []
for _ in range(100):
    dummy_input = paddle.randn([1, 3, 224, 224])
    start = time.time()
    with paddle.no_grad():
        _ = model(dummy_input)
    times.append((time.time() - start) * 1000)

avg_time = np.mean(times)
fps = 1000 / avg_time

print(f"\nBenchmark Results:")
print(f"  Average inference time: {avg_time:.2f} ms")
print(f"  FPS: {fps:.2f}")
print(f"  Device: {'GPU' if paddle.is_compiled_with_cuda() else 'CPU'}")

# Expected performance on RTX 3050 (approximate)
if paddle.is_compiled_with_cuda():
    print(f"\n  Expected on RTX 3050: ~5-10 ms (100-200 FPS)")
else:
    print(f"\n  Expected on CPU: ~50-100 ms (10-20 FPS)")
EOF

python benchmark_test.py
rm benchmark_test.py
```

## Test 7: Export Functionality

Test model export to different formats:

```bash
cat > test_export.py << 'EOF'
import sys
sys.path.insert(0, 'classification')

import paddle
import os
from ppcls.arch.backbone.model_zoo.cspconvnext import CSPConvNeXt

# Create model
model = CSPConvNeXt(arch='mini', class_num=10)
model.eval()

# Test inference model export
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)

print("Exporting to inference format...")
paddle.jit.save(
    layer=model,
    path=os.path.join(output_dir, 'model'),
    input_spec=[paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype='float32')]
)

print(f"✓ Model exported to {output_dir}/")

# Verify exported files
assert os.path.exists(f'{output_dir}/model.pdmodel')
assert os.path.exists(f'{output_dir}/model.pdiparams')

print("✓ All export files exist")

# Clean up
import shutil
shutil.rmtree(output_dir)
print("✓ Export test passed!")
EOF

python test_export.py
rm test_export.py
```

## Common Test Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'paddle'"

**Solution:**
```bash
pip install paddlepaddle-gpu  # For GPU
# or
pip install paddlepaddle  # For CPU only
```

### Issue: "CUDA out of memory" during testing

**Solution:**
```python
# Use smaller batch size in tests
BATCH_SIZE = 1  # or 2

# Or use CPU for testing
paddle.set_device('cpu')
```

### Issue: Dataset files not found

**Solution:**
```bash
# Check paths are correct
ls -l dataset/custom_dataset/train_list.txt

# Ensure paths in list files are relative
head dataset/custom_dataset/train_list.txt
```

### Issue: Import errors in notebook

**Solution:**
```python
# Ensure paths are added correctly
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'classification'))
sys.path.insert(0, os.path.join(os.getcwd(), 'detection'))
```

## Automated Test Script

Create a comprehensive test script:

```bash
cat > run_all_tests.sh << 'EOF'
#!/bin/bash

echo "=================================="
echo "E-ConvNeXt Automated Test Suite"
echo "=================================="

# Test 1: File integrity
echo -e "\n[Test 1] Checking file integrity..."
python -c "import json; json.load(open('E_ConvNeXt_Flexible_Training.ipynb'))" && \
    echo "✓ Notebook valid" || echo "✗ Notebook invalid"

python -m py_compile prepare_dataset.py && \
    echo "✓ prepare_dataset.py valid" || echo "✗ prepare_dataset.py invalid"

# Test 2: Check PaddlePaddle
echo -e "\n[Test 2] Checking PaddlePaddle installation..."
python -c "import paddle; print(f'✓ PaddlePaddle {paddle.__version__} installed')" || \
    echo "✗ PaddlePaddle not installed"

# Test 3: Model import
echo -e "\n[Test 3] Testing model import..."
cd classification
python -c "from ppcls.arch.backbone.model_zoo.cspconvnext import CSPConvNeXt; print('✓ Model import successful')" || \
    echo "✗ Model import failed"
cd ..

# Test 4: GPU availability
echo -e "\n[Test 4] Checking GPU..."
python -c "import paddle; print(f'✓ GPU Available: {paddle.is_compiled_with_cuda()}')"

echo -e "\n=================================="
echo "Test suite completed!"
echo "=================================="
EOF

chmod +x run_all_tests.sh
./run_all_tests.sh
```

## Expected Test Results

### On RTX 3050 GPU:

- Model creation: < 1 second
- Forward pass (batch=1): ~5-10 ms
- Training step (batch=32): ~100-200 ms
- Inference FPS: ~150-200

### On CPU (i7):

- Model creation: < 1 second
- Forward pass (batch=1): ~50-100 ms
- Training step (batch=32): ~1-2 seconds
- Inference FPS: ~10-20

## Next Steps After Testing

Once all tests pass:

1. ✅ Start training on your real dataset
2. ✅ Monitor training progress
3. ✅ Export trained model
4. ✅ Benchmark on target hardware
5. ✅ Deploy to production

## Reporting Issues

If tests fail:

1. Check error messages carefully
2. Verify all prerequisites are installed
3. Check dataset format
4. Review [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)
5. Open an issue on GitHub with:
   - Error message
   - Test that failed
   - System configuration
   - Steps to reproduce

---

**Happy Testing! 🧪**
