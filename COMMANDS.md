```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
conda activate ./env

# Not sure if this is the correct version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python hello.py



# For CUDA easy intro into
# https://developer.nvidia.com/blog/even-easier-introduction-cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```