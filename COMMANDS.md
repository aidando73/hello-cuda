```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
conda activate ./env
source ~/miniconda3/bin/activate ./env

# Install CUDA 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python hello.py



# For CUDA easy intro into
# https://developer.nvidia.com/blog/even-easier-introduction-cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

sudo apt update
sudo apt install clang build-essential


```

- Regular pytorch impl: Forward: 235.543 us | Backward 431.620 us
- GPU impl: 226.908 us | Backward 695.002 us
- CUDA impl: 