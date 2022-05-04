import torch

DEVICE_STATUS = torch.cuda.is_available()
DEVICE_COUNT = torch.cuda.device_count()
