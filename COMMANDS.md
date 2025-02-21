```bash
source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10
conda activate ./env

# Not sure if this is the correct version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python hello.py
```