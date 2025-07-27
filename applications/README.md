## Configuring new applications with ConsumerBench
Users can configure any custom applications that use local GenAI models, to run with ConsumerBench. The process of adding a new application to ConsumerBench is:

1. Create a new sub-directory in this folder for the application
2. Install the application in the sub-directory
3. Inherit a child class from the `Application` class and implement its corresponding functions. (Please see the existing applications, such as DeepResearch: https://github.com/efeslab/ConsumerBench/blob/master/applications/DeepResearch/DeepResearch.py)
4. You can then add your own applications to the workflows (specified in `configs/`), and the application will be monitored automatically with ConsumerBench

## Setting up existing applications

Currently, the ConsumerBench repository contains with 4 applications: Chatbot, DeepResearch, LiveCaptions and Imagegen. We have already added their classes in the corresponding directories. 

Following are the steps to install the applications, setup the inference backend with the model and the datasets specified in the paper. While we specify the model and dataset here, users are free to download their own models and datasets to use with the applications. 

### Chatbot 


### DeepResearch
Create a new conda environment with python 3.10. Activate the environment.

#### Install Application
```
$ cd DeepResearch/smolagents/examples/open_deep_research
$ pip install -r requirements.txt 
$ pip install -e ../../.[dev]
```

#### Download GenAI model
Download the Llama-3.2-3B model from huggingface. Note that you may need a huggingface account, and permission to download the gated llama model. 
```
$ wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf
$ mv Llama-3.2-3B-Instruct-f16.gguf ../models/
```

#### Download Dataset



### Imagegen
Create a conda environment with python 3.10. Activate the environment.

#### Install Application
```
pip install -r requirements.txt
pip install diffusers
pip install transformers==4.50.3
```

#### Download GenAI model
Download the stable-diffusion-3.5-large model from huggingface. Note that you may need a huggingface account
```
$ git lfs install
$ git clone https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo
$ mv stable-diffusion-3.5-medium-turbo ../models/
```

#### Download Dataset



### LiveCaptions
Create a conda environment with python 3.10. Activate the environment.

#### Install Application
```
pip install librosa soundfile
pip install faster-whisper
pip install torch torchaudio
pip install transformers
pip install datasets
pip install torchcodec
```

#### Download GenAI model
Download the Whisper-Large-V3-Turbo model from huggingface. Note that you may need a huggingface account
```
git lfs install
git clone https://huggingface.co/openai/whisper-large-v3-turbo
mv whisper-large-v3-turbo ../models/
```

#### Download Dataset