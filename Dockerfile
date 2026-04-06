# NVIDIA CUDA 11.3 + Ubuntu 20.04 (Python 3.8)
# Use the devel image so CUDA extensions such as MMCV deformable attention
# can be built or validated correctly for training workloads.
# PyTorch cu113 wheels are linux/amd64-only; pin platform so `docker build` works on ARM hosts.
FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.6"

# Install system dependencies and Python tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    pkg-config \
    wget \
    vim \
    nano \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Verify Python version
RUN python --version && python -c "import sys; print(sys.version_info); assert sys.version_info >= (3, 8), 'Python 3.8+ required'"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 1.12.1 + torchvision for CUDA 11.3 (cu113 wheels are on the CUDA-specific index)
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenMMLab tooling and mm-series packages.
# We keep the matching MMCV version, but use a devel image so the CUDA ops are
# available for actual training rather than just import-time resolution.
RUN pip install openmim && \
    mim install "mmcv-full==1.5.2" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html && \
    pip install mmdet==2.26.0 && \
    pip install mmsegmentation==0.29.1 && \
    pip install mmdet3d==1.0.0rc6

# Create app directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install project dependencies
# openlanev2==2.1.0 declares ortools==9.2.9972 which was never published;
# install it without deps and supply a compatible ortools via requirements.txt
# chardet/ninja: openlanev2 metadata expects them; omit jupyter (not needed for train/test).
# Shapely stays at MMDet3D/nuscenes-devkit pin (<2); OpenLane-V2 lanesegment IO works with 1.8.x.
RUN pip install --no-deps openlanev2==2.1.0 && \
    pip install -r requirements.txt && \
    pip install chardet ninja

# Quick environment check so Docker builds fail early if CUDA extension support
# is obviously broken.
RUN python -c "import torch, mmcv; print('torch_cuda', torch.cuda.is_available()); print('cuda_home', torch.utils.cpp_extension.CUDA_HOME); print('mmcv', mmcv.__version__)"

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/work_dirs /workspace/output

# Expose common ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Set default command
CMD ["/bin/bash"]
