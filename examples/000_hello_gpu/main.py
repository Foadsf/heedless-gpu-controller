import torch
import sys

print("--- CLOUD GPU REPORT ---")
print(f"Python Version: {sys.version}")

if torch.cuda.is_available():
    print(f"SUCCESS: CUDA is available!")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Confirm WHICH accelerator you actually got. Requesting NvidiaTeslaT4 in
    # kernel-metadata.json is a request, not a guarantee -- historically it could
    # be silently reset to P100. Trust this line over what you asked for.
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {major}.{minor}  (T4/Turing = 7.5, P100/Pascal = 6.0)")
    if major < 7:
        print("WARNING: pre-Turing GPU -- you did NOT get the T4 you asked for.")

    # Do some math on the GPU to prove it works
    print("Running Matrix Multiplication Test...")
    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()
    z = torch.matmul(x, y)
    print("Matrix multiplication complete. GPU is active and healthy.")
else:
    print("FAILURE: No GPU detected.")
