import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available (torch.cuda.is_available()): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot find the CUDA runtime.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: **{device}**")