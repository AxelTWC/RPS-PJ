import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version PyTorch built with:", torch.version.cuda)
