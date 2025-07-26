## Configuring new applications with ConsumerBench
Users can configure any custom applications that use local GenAI models, to run with ConsumerBench. The process of adding a new application to ConsumerBench is:

1. Create a new sub-directory in this folder for the application
2. Install the application in the sub-directory
3. Inherit a child class from the `Application` class and implement its corresponding functions. (Please see the existing applications, such as DeepResearch: <>)
4. You can then add your own applications to the workflows (specified in `configs/`), and the application will be monitored automatically with ConsumerBench

## Setting up existing applications

Currently, the ConsumerBench repository contains with 4 applications: Chatbot, DeepResearch, LiveCaptions and Imagegen. We have already added their classes in the corresponding directories. Only installation of applications is necessary for working with ConsumerBench.

### Installing existing applications
We recommend creating a separate conda environment for each application, due to potential package version mismatches across applications. We create a conda environment with python version 3.10 for each application.

### Chatbot 

### DeepResearch
```
cd smolagents/examples/open_deep_research
pip install -r requirements.txt 
pip install -e ../../.[dev]
```

### Imagegen

### LiveCaptions

