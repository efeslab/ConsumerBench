#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate benchmark

models=(
    # "/home/cc/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf"
    # "/home/cc/models/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-fp16-00001-of-00002.gguf"
    # "/home/cc/models/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    # "/home/cc/models/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-f16.gguf"
    "/home/cc/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf"
)

result_base_path="/home/cc/os-llm/results/accuracy"
mkdir -p $result_base_path

# Loop through each model
for model in "${models[@]}"; do
    # Extract the model name from the path
    model_name=$(basename "$model")
    
    # Create a directory for the model's results
    model_result_path="$result_base_path/$model_name"
    mkdir -p $model_result_path

    # init llama.cpp server
    /home/cc/llama.cpp/build/bin/llama-server --port 5000 -m $model -ngl 99 --parallel 16 -c 32000 &
    SERVER_PID=$!
    sleep 30

    # Run the benchmark script with the model and save the output to a file
    echo "Running benchmark for $model_name..."
    python3 /home/cc/os-llm/scripts/benchmark_accuracy.py --api-base 'http://localhost:5000/v1' --model $model --output-dir $model_result_path
    
    # Check if the benchmark script ran successfully
    if [ $? -eq 0 ]; then
        echo "Benchmark for $model_name completed successfully."
    else
        echo "Benchmark for $model_name failed."
    fi

    kill -2 $SERVER_PID
    sleep 5
done