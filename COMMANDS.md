```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env

# Install CUDA 
pip install --no-build-isolation torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

python hello.py

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

sudo apt update
sudo apt install clang build-essential


```

- Regular pytorch impl: Forward: 235.543 us | Backward 431.620 us
- GPU impl: 226.908 us | Backward 695.002 us
- CUDA impl: 