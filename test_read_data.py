import torch
import numpy as np

data = torch.load('/Users/wangchen47/Downloads/test_2d.pth.tar')
for key in data:
    print(key)
