import model_torch
import torch
import numpy as np
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()

model = model_torch.LinearModel()
model.eval()

print(model)

# torch.save(model.state_dict(), 'torch_weight.pth')

weight_dict = torch.load('torch_weight.pth')

model.load_state_dict(weight_dict['state_dict'])

fake_data = np.load("fake_data.npy")
fake_data = torch.Tensor(fake_data)

out = model(fake_data)

reprod_logger.add("logits", out.cpu().detach().numpy())
reprod_logger.save("forward_torch.npy")