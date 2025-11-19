# System Monitors Setup

## Intel PCM

```
sudo apt update
sudo apt install linux-tools-common linux-tools-$(uname -r)

# Clone and build PCM
git clone https://github.com/intel/pcm.git
cd pcm
mkdir build
cd build
cmake ..
cmake --build .
sudo cp pcm-memory /usr/local/bin/

```

### Initialize PCM Service

```
sudo modprobe msr
# make sure this runs without error
sudo pcm-memory 0.05
```

## DCGM

```
sudo apt update
sudo apt install datacenter-gpu-manager
sudo apt install moreutils

```

### Enable DCGM Service

```
# check status
sudo systemctl status nvidia-dcgm
sudo systemctl start nvidia-dcgm
sudo systemctl enable nvidia-dcgm   # optional

```
### WSL
Follow the guide here https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html#ubuntu-lts-and-debian

Remember to first register CUDA key ring https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#wsl-installation


## For Power Usage

```
pip install psutil nvidia-ml-py3 py3nvml
```