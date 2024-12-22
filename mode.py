import torch

# 查询 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 查询 CUDA 版本
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")