import torch
num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)