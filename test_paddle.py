import model_paddle
import paddle
import numpy as np
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()

model = model_paddle.LinearModel()
model.eval()

print(model)

# paddle.save(model.state_dict(), 'torch_weight.pth')

weight_dict = paddle.load('paddle_weight.pdparams')

model.set_state_dict(weight_dict)

fake_data = np.load("fake_data.npy")
fake_data = paddle.to_tensor(fake_data)

out = model(fake_data)

reprod_logger.add("logits", out.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")