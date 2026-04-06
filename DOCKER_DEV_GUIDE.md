# Docker Development Guide for LaneSegNet

This guide explains how to use Docker for development of LaneSegNet.

## Prerequisites

- **Docker**: [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Apple Silicon (M1/M2/M3)**: The image targets **linux/amd64** (CUDA wheels and NVIDIA stack). `docker-compose.yml` sets `platform: linux/amd64`; for a plain `docker build`, use `docker build --platform linux/amd64 -t lanesegnet:latest .` if the host is ARM.
- **NVIDIA GPU in Docker**: Only practical on **Linux** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). **Docker Desktop on macOS does not ship a `nvidia` runtime** (and Macs are not NVIDIA CUDA training targets); use the default `docker-compose.yml` there. On Linux with an NVIDIA GPU, start services with both compose files: `docker compose -f docker-compose.yml -f docker-compose.linux-gpu.yml up -d`.
- **NVIDIA GPU** (recommended for model training)

## Quick Start

### 1. Build the Docker Image

```bash
# Using Docker Compose (recommended)
docker compose build

# Or using docker directly
docker build -t lanesegnet:latest .
```

### 2. Run the Container

```bash
# Option A: Interactive container (macOS / CPU, or Linux without GPU override)
docker compose up lanesegnet -d
docker compose exec lanesegnet bash

# Linux + NVIDIA GPU (merge the GPU compose fragment)
docker compose -f docker-compose.yml -f docker-compose.linux-gpu.yml up lanesegnet -d
docker compose -f docker-compose.yml -f docker-compose.linux-gpu.yml exec lanesegnet bash

# Option B: Single command
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/work_dirs:/workspace/work_dirs \
  lanesegnet:latest bash

# Option C: Run a specific training command
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/work_dirs:/workspace/work_dirs \
  lanesegnet:latest \
  python tools/train.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py --work-dir work_dirs/lanesegnet
```

### 3. With Jupyter Notebook

```bash
# Start the Jupyter service
docker-compose up jupyter -d

# Access at http://localhost:8889
# Token will be displayed in logs:
docker-compose logs jupyter
```

## Common Use Cases

### Training the Model

```bash
# Inside container
cd /workspace
python tools/train.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \
  --work-dir work_dirs/lanesegnet \
  --gpu-id 0

# Or with distributed training (8 GPUs)
# Note: config and work-dir are hardcoded in dist_train.sh
bash tools/dist_train.sh 8
```

### Direct Training (without entering shell)

Launch training directly from host without an interactive shell:

```bash
# Single GPU training using docker-compose
docker-compose run --rm lanesegnet \
  python tools/train.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \
  --work-dir work_dirs/lanesegnet --gpu-id 0

# Multi-GPU training (8 GPUs) using docker-compose
docker-compose run --rm lanesegnet bash tools/dist_train.sh 8

# Or using docker run directly (multi-GPU example)
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/work_dirs:/workspace/work_dirs \
  lanesegnet:latest \
  bash tools/dist_train.sh 8
```

The `--rm` flag automatically removes the container after training completes. Volume mounts persist training outputs and checkpoints to your host machine.

### Testing the Model

```bash
# Inside container
python tools/test.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \
  work_dirs/lanesegnet/latest.pth \
  --format-only
```

### Data Processing

```bash
# Inside container
python tools/data_process.py
```

## Volume Mounts

The docker-compose.yml file includes the following volume mounts:

- **`.:/workspace`** - Entire project directory (source code)
- **`./data:/workspace/data`** - Dataset directory
- **`./work_dirs:/workspace/work_dirs`** - Training outputs and checkpoints
- **`./output:/workspace/output`** - Generated outputs

This allows you to:
- Edit code on your host machine and run in container
- Access training outputs on your host machine
- Persist data across container restarts

## Port Mappings

- **Port 8888**: Jupyter Notebook (main service)
- **Port 8889**: Alternative Jupyter instance (jupyter service)
- **Port 6006**: TensorBoard
- **Port 5000**: Additional services

## GPU Support

### Verify GPU Access

```bash
# Inside container
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Using Specific GPUs

```bash
# Use GPU 0 and 1
docker run --rm -it --gpus '"device=0,1"' \
  -v $(pwd):/workspace \
  lanesegnet:latest bash

# Use all GPUs
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  lanesegnet:latest bash
```

## Development Workflow

### 1. Start Container

```bash
docker-compose up -d lanesegnet
```

### 2. Access Container

```bash
docker-compose exec lanesegnet bash
# or
docker exec -it lanesegnet-dev bash
```

### 3. Install Additional Packages

```bash
# Inside container
pip install [package_name]
```

### 4. Edit Code

Edit files on your host machine using your favorite editor. Changes are immediately reflected in the container.

### 5. Run Experiments

```bash
# Inside container
python tools/train.py ...
```

### 6. Stop Container

```bash
docker-compose down
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.1.1-runtime-ubuntu20.04 nvidia-smi

# If error, ensure nvidia-docker is properly installed
# See: https://github.com/NVIDIA/nvidia-docker
```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run container as current user (Linux)
docker run --rm -it --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  lanesegnet:latest bash
```

### Out of Memory

Reduce batch size or model size in config files. Check GPU memory:

```bash
# Inside container
nvidia-smi
```

### Container Too Large

The `.dockerignore` file is configured to exclude unnecessary files. To further reduce size:

```bash
# Remove data and work_dirs from .dockerignore entries
# Rebuild: docker-compose build --no-cache
```

## Building Production Image

For production deployment (without development tools):

```dockerfile
# Modify Dockerfile to use slim base image and remove unnecessary packages
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# ... install only necessary packages ...

# Don't use CUDA sample utils, development headers, etc.
```

## Advanced Configuration

### Custom Environment Variables

Create a `.env` file:

```
CUDA_VISIBLE_DEVICES=0,1
PYTHONUNBUFFERED=1
```

Then use in docker-compose:

```yaml
env_file: .env
```

### Multi-Stage Build

For optimized images:

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 as builder
# ... build dependencies ...

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
COPY --from=builder /usr/local ...
```

## Additional Resources

- [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [LaneSegNet Repository](https://github.com/OpenDriveLab/LaneSegNet)
- [OpenLane-V2 Repository](https://github.com/OpenDriveLab/OpenLane-V2)
