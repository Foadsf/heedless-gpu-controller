import torch
import sys

print("--- CLOUD GPU REPORT ---")
print(f"Python Version: {sys.version}")

if torch.cuda.is_available():
    print(f"SUCCESS: CUDA is available!")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Do some math on the GPU to prove it works
    print("Running Matrix Multiplication Test...")
    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()
    z = torch.matmul(x, y)
    print("Matrix multiplication complete. GPU is active and healthy.")
else:
    print("FAILURE: No GPU detected.")
