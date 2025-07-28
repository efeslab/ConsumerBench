# System Monitors Setup

## Intel PCM

```
sudo apt update
sudo apt install linux-tools-common linux-tools-$(uname -r)

# Clone and build PCM
git clone https://github.com/intel/pcm.git
cd pcm
make
sudo cp pcm-memory /usr/local/bin/

```

## DCGM

```
sudo apt update
sudo apt install datacenter-gpu-manager
sudo apt install moreutils

```

## For Power Usage

```
pip install psutil nvidia-ml-py3 py3nvml
```