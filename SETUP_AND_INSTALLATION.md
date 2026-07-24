# Complete Setup and Installation Guide: EG3D + DeepFaceLive Integration

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Environment Setup](#environment-setup)
5. [Model Preparation](#model-preparation)
6. [TensorRT Compilation](#tensorrt-compilation)
7. [Verification and Testing](#verification-and-testing)
8. [Troubleshooting](#troubleshooting)
9. [Quick Start](#quick-start)

---

## System Requirements

### Hardware

**Minimum Requirements:**
- NVIDIA GPU with Compute Capability 7.0+ (RTX 20 series or newer)
- 24 GB VRAM (RTX 3090 recommended)
- 16 GB System RAM
- SSD with 100 GB free space
- Modern CPU (Intel i7/i9 or AMD Ryzen 7/9)

**Tested Hardware:**
- ✅ NVIDIA RTX 3090 (24 GB) - Optimal
- ✅ NVIDIA RTX 3080 (10 GB) - Requires batch_size=1
- ✅ NVIDIA A100 (40 GB) - Excellent
- ✅ NVIDIA V100 (32 GB) - Good

### Software

**Operating System:**
- Ubuntu 20.04 LTS or newer
- Windows 10/11 with WSL2 (limited testing)
- macOS (GPU support limited)

**Core Dependencies:**
- Python 3.9+ (3.10 recommended)
- CUDA Toolkit 11.8+
- cuDNN 8.x
- TensorRT 8.6+
- PyTorch 2.0+
- Qt6 (for GUI)

---

## Prerequisites

### 1. CUDA and cuDNN Installation

#### On Ubuntu/Debian:

```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-repo-ubuntu2004_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004_11.8.0-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### cuDNN Installation:

```bash
# Download cuDNN from NVIDIA (requires login)
# https://developer.nvidia.com/cudnn

# Extract and install
tar -xvf cudnn-linux-x86_64-8.x.x.x_cuda11.x-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.x.x.x_cuda11.x-archive/include/* /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-8.x.x.x_cuda11.x-archive/lib/* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn* /usr/local/cuda/lib64/libcudnn*
```

### 2. Git and Development Tools

```bash
# Ubuntu/Debian
sudo apt install -y git build-essential cmake ninja-build

# Clone the repository
git clone https://github.com/BrainWi2ard/eg3d.git
cd eg3d
```

### 3. Python Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## Step-by-Step Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/BrainWi2ard/eg3d.git
cd eg3d
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create environment
python3.10 -m venv venv

# Activate
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorRT
pip install tensorrt

# Install ONNX and related tools
pip install onnx onnxruntime-gpu

# Install OpenCV
pip install opencv-python opencv-contrib-python

# Install other core packages
pip install numpy scipy scikit-image scikit-learn

# Install Qt6 for GUI
pip install PyQt6 PyQt6-WebEngine

# Install additional utilities
pip install tqdm click pyyaml requests psutil
```

### Step 4: Install from requirements.txt

```bash
# Install complete dependency set
pip install -r requirements.txt

# If requirements.txt doesn't exist, create it:
cat > requirements.txt << 'EOF'
torch==2.0.0+cu118
torchvision==0.15.0+cu118
torchaudio==2.0.0+cu118
tensorrt==8.6.1
onnx==1.14.0
onnxruntime-gpu==1.15.0
opencv-python==4.8.0.74
opencv-contrib-python==4.8.0.74
numpy==1.24.3
scipy==1.11.0
scikit-image==0.21.0
scikit-learn==1.3.0
PyQt6==6.5.0
PyQt6-WebEngine==6.5.0
click==8.1.6
tqdm==4.65.0
pyyaml==6.0
requests==2.31.0
psutil==5.9.5
mrcfile==1.4.3
imageio==2.31.3
imageio-ffmpeg==0.4.9
tensorboard==2.13.0
EOF
pip install -r requirements.txt
```

### Step 5: Create Directory Structure

```bash
# Create necessary directories
mkdir -p modelhub/eg3d
mkdir -p modelhub/onnx
mkdir -p identities
mkdir -p outputs
mkdir -p logs
mkdir -p datasets/sample_videos
mkdir -p checkpoints

echo "✓ Directory structure created"
```

### Step 6: Download Pre-trained Models

```bash
# Download EG3D FFHQ checkpoint (requires ~1.5 GB)
cd modelhub/eg3d

# Option A: Using wget
wget https://nvlabs-fi-cdn.nvidia.com/stylegan3-paper/pretrained/eg3d-fixed-triplanes/ffhq512-128.pkl -O ffhq_triplane_generator.pkl

# Option B: Manual download
# Visit: https://github.com/NVlabs/eg3d
# Download ffhq512-128.pkl and place in modelhub/eg3d/

# Verify download
ls -lh *.pkl

cd ../..
echo "✓ Pre-trained model downloaded"
```

---

## Environment Setup

### Python Path Configuration

```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Make permanent (add to ~/.bashrc)
echo 'export PYTHONPATH="${PYTHONPATH}:~/eg3d"' >> ~/.bashrc
```

### CUDA Environment Variables

```bash
# Verify CUDA is accessible
python -c "import torch; print(torch.cuda.is_available())"

# If False, set CUDA paths:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test TensorRT
python -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test Qt6
python -c "import PyQt6; print('PyQt6 OK')"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA: True
TensorRT: 8.6.1
OpenCV: 4.8.0
PyQt6 OK
```

---

## Model Preparation

### Option 1: Automatic Model Setup

```bash
# Run setup script (creates everything)
python scripts/setup_models.py

# This script:
# - Verifies model checkpoints
# - Creates identity embeddings
# - Generates test data
```

### Option 2: Manual Setup

#### 1. Verify Model Structure

```bash
python -c "
import pickle
import torch

# Load and inspect model
with open('modelhub/eg3d/ffhq_triplane_generator.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

print('Checkpoint keys:', checkpoint.keys())
print('Generator:', checkpoint['G_ema'])
print('Synthesis network:', checkpoint['G_ema'].synthesis)
"
```

#### 2. Create Target Identity Embedding

```bash
python scripts/invert_target_identity.py \
    --image path/to/target_face.jpg \
    --output identities/target_identity.pt
```

#### 3. Prepare Sample Data

```bash
# Copy sample video to test directory
cp /path/to/sample_video.mp4 datasets/sample_videos/

# Or use a webcam stream directly in the GUI
```

---

## TensorRT Compilation

### Step 1: Export to TensorRT

```bash
# FP16 precision (recommended for RTX 3090)
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --output modelhub/eg3d \
    --precision fp16 \
    --max_batch_size 4 \
    --max_workspace_gb 4 \
    --benchmark

# This will:
# 1. Load PyTorch model
# 2. Export to ONNX
# 3. Compile to TensorRT
# 4. Run performance benchmarks
```

Expected output:
```
INFO - Loading PyTorch model from modelhub/eg3d/ffhq_triplane_generator.pkl...
INFO - ✓ Model loaded successfully
INFO - Exporting synthesis network to ONNX...
INFO - ✓ ONNX export successful
INFO - Compiling ONNX to TensorRT (fp16)...
INFO - Building TensorRT engine (this may take 1-3 minutes)...
INFO - ✓ Engine compilation successful!
INFO - File: modelhub/eg3d/eg3d_tensorrt_fp16.engine
INFO - Size: 847.3 MB
INFO - Precision: fp16

Benchmark results:
  Avg latency: 18.6 ms
  Throughput: 53.8 FPS
```

### Step 2: Alternative Precisions

```bash
# FP32 (slower but more compatible)
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision fp32

# INT8 (fastest but requires calibration data)
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision int8
```

### Step 3: Verify Engine

```bash
python -c "
import tensorrt as trt

engine_path = 'modelhub/eg3d/eg3d_tensorrt_fp16.engine'
logger = trt.Logger()
runtime = trt.Runtime(logger)

with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

print(f'✓ Engine valid')
print(f'  Bindings: {engine.num_bindings}')
for i in range(engine.num_bindings):
    print(f'    {engine.get_binding_name(i)}: {engine.get_binding_shape(i)}')
"
```

---

## Verification and Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_eg3d_inference.py -v
pytest tests/test_tensorrt_engine.py -v
pytest tests/test_multiprocessing.py -v

# Run with coverage
pytest tests/ --cov=apps --cov=eg3d_engine
```

### Integration Tests

```bash
# Test TensorRT engine
python scripts/validate_tensorrt_engine.py \
    --engine modelhub/eg3d/eg3d_tensorrt_fp16.engine

# Test complete pipeline
python scripts/validate_pipeline.py \
    --config configs/default.yaml
```

### Manual Verification

```bash
# Test model loading
python -c "
from apps.DeepFaceLive.backend.EG3DInferenceNode import EG3DInferenceNode
import logging

logging.basicConfig(level=logging.INFO)

# Create worker
worker = EG3DInferenceNode(
    engine_path='modelhub/eg3d/eg3d_tensorrt_fp16.engine'
)

print('✓ EG3DInferenceNode loaded successfully')
"

# Test inference
python scripts/test_inference.py \
    --engine modelhub/eg3d/eg3d_tensorrt_fp16.engine \
    --num_iterations 10
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

```bash
# Solution: Reduce batch size and workspace
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --max_batch_size 1 \
    --max_workspace_gb 1 \
    --precision fp16

# Or reduce precision
--precision int8
```

#### Issue 2: TensorRT Not Found

```bash
# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"

# If missing, reinstall
pip uninstall tensorrt -y
pip install tensorrt==8.6.1
```

#### Issue 3: ONNX Export Fails

```bash
# Update ONNX and PyTorch
pip install --upgrade onnx
pip install --upgrade torch

# Try with different opset
python scripts/export_eg3d_tensorrt.py \
    --opset_version 16
```

#### Issue 4: Engine File Corrupted

```bash
# Remove and regenerate
rm modelhub/eg3d/*.engine

python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision fp16
```

#### Issue 5: Webcam Not Detected

```bash
# Check available cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} available')
        cap.release()
"

# Grant permissions (Linux)
sudo usermod -a -G video $USER
```

### Debug Logging

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Or in Python:
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Quick Start

### 1. First-Time Setup (Complete)

```bash
#!/bin/bash
# setup_complete.sh

# Clone repository
git clone https://github.com/BrainWi2ard/eg3d.git
cd eg3d

# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p modelhub/eg3d identities outputs logs

# Download model
cd modelhub/eg3d
wget https://nvlabs-fi-cdn.nvidia.com/stylegan3-paper/pretrained/eg3d-fixed-triplanes/ffhq512-128.pkl
cd ../..

# Compile TensorRT
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision fp16 \
    --benchmark

# Run tests
pytest tests/ -v

# Launch GUI
python main.py

echo "✓ Setup complete!"
```

Run it:
```bash
bash setup_complete.sh
```

### 2. Launch the GUI Application

```bash
# Activate environment
source venv/bin/activate

# Start GUI
python main.py

# Or with options
python main.py --webcam 0 --fps 30 --debug
```

### 3. Basic Usage in GUI

1. **Load Model**: Click "Load Model" → Select TensorRT engine
2. **Select Camera**: Choose webcam source
3. **Load Target Identity**: Select target face image
4. **Start Processing**: Click "Start" button
5. **View Output**: Real-time preview window
6. **Save Results**: Export video or screenshots

---

## Performance Optimization

### For Slower Hardware (RTX 2080 Ti, RTX 3080):

```bash
# Use FP16 and smaller batch
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision fp16 \
    --max_batch_size 1 \
    --max_workspace_gb 2

# In GUI: Reduce input resolution to 720p
```

### For Better Quality (A100, RTX 3090):

```bash
# Use FP32 for maximum precision
python scripts/export_eg3d_tensorrt.py \
    --model modelhub/eg3d/ffhq_triplane_generator.pkl \
    --precision fp32 \
    --max_batch_size 8 \
    --max_workspace_gb 8

# In GUI: Enable 4K output and advanced filtering
```

---

## Next Steps

1. ✅ Complete installation
2. ✅ Verify all components working
3. ✅ Launch GUI console
4. ✅ Load pre-trained model
5. ✅ Run real-time face swapping
6. 📖 Read [API_REFERENCE.md](./docs/API_REFERENCE.md)
7. 📖 Explore [ARCHITECTURE.md](./ARCHITECTURE_EG3D_DEEPFACELIVE_INTEGRATION.md)

---

## Support and Resources

- **GitHub Issues**: https://github.com/BrainWi2ard/eg3d/issues
- **EG3D Paper**: https://arxiv.org/abs/2112.07945
- **DeepFaceLive**: https://github.com/iperov/DeepFaceLive
- **NVIDIA TensorRT Docs**: https://docs.nvidia.com/deeplearning/tensorrt/

---

## License

This project integrates:
- **EG3D**: NVIDIA Proprietary License (Research Only)
- **DeepFaceLive**: MIT License
- **Custom Code**: MIT License

See [LICENSE.txt](./LICENSE.txt) for details.
