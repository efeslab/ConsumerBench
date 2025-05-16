import torch
import nvtx
import time
import sys

# Print NVTX module information
print("NVTX module information:")
print(f"Type: {type(nvtx)}")
print(f"Available methods: {dir(nvtx)}")
print("-" * 50)

def main():
    print("Starting program")

    # Try using NVTX in the simplest way
    print("Pushing NVTX range 'Main'")
    try:
        nvtx.push_range("Main")
        print("NVTX push successful")
    except Exception as e:
        print(f"NVTX push error: {e}")
        sys.exit(1)

    # Simple CUDA operation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    a = torch.rand(1000, 1000, device=device)
    b = torch.rand(1000, 1000, device=device)
    torch.cuda.synchronize()

    # Matrix multiplication
    print("Pushing NVTX range 'MatMul'")
    try:
        nvtx.push_range("MatMul")
        print("MatMul NVTX push successful")
    except Exception as e:
        print(f"MatMul NVTX push error: {e}")

    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    print("Popping MatMul range")
    try:
        nvtx.pop_range()
        print("MatMul NVTX pop successful")
    except Exception as e:
        print(f"MatMul NVTX pop error: {e}")

    # End of main range
    print("Popping Main range")
    try:
        nvtx.pop_range()
        print("Main NVTX pop successful")
    except Exception as e:
        print(f"Main NVTX pop error: {e}")

    print("Program complete")

if __name__ == "__main__":
    main()