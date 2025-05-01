# :yum: Installation guide

This file provides a step-by-step installation guide to install [conda / mamba](https://github.com/conda-forge/miniforge), create a virtual environment, and install the required libraries for GPU compatibility !

## General software installation (`Debian`)

This command installs basic requirements for python installation :
```bash
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev wget
```

This command installs some additional requirements for some libraries
```bash
sudo apt install -y ffmpeg graphviz poppler-utils lzma
```

This command installs specific requirements for the `TensorRT-LLM` library
```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev git-lfs
```

## Mamba installation

```bash
wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
sudo bash Miniforge3.sh -b -p /usr/local/conda

source /usr/local/conda/etc/profile.d/conda.sh
source /usr/local/conda/etc/profile.d/mamba.sh

mamba init
```

### Building a new environment

```bash
# Create a new environment with Python 3.12
mamba create -n yui_project python=3.12

# Activate the environment
mamba activate yui_project

# De activate the environment
mamba deactivate
```

### Basic packages installation

```bash
pip install jupyter jupyterlab notebook

# Optional : if the project requires `TensorRT-LLM`
pip install --upgrade --extra-index-url https://pypi.nvidia.com tensorrt-llm

# It is important to execute this line after the installation of TRT-LLM to have the correct cuda/cudnn versions for tensorflow
pip install --upgrade tensorflow[and-cuda]
```

## Specific project installation

```bash
# Replace `base_dl_project` by the repository you want to run
git clone https://github.com/yui-mhcp/base_dl_project
cd base_dl_project

pip install -r requirements.txt

# Start the jupyter notebook server !
jupyter lab --ip 0.0.0.0 --allow-root
```

Now let's ensure that the GPU is correctly detected !
```python
#import jax
#import torch
#import tensorrt_llm
import tensorflow as tf

#print('GPU is available in JAX        : {}'.format(jax.lib.xla_bridge.get_backend().platform == 'gpu'))
#print('GPU is available in pytorch    : {}'.format(torch.cuda.is_available()))
print('GPU is available in tensorflow : {}'.format(tf.config.list_physical_devices('GPU') != []))
```

You can also run the unitest suites :
```bash
KERAS_BACKEND=tensorflow python -m unittest discover -v -t . -s tests -p test_*.py
```

Note : there is currently some errors when executing the tests with other backends. However, these are non-critical for any of the provided codes.

