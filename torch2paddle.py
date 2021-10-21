import torch
import paddle
import numpy as np


def transfer():
    input_fp = "torch_weight.pth"
    output_fp = "paddle_weight.pdparams"
    torch_dict = torch.load(input_fp)['state_dict']
    paddle_dict = {}
    fc_names = [
        "w1.weight", "linear_stages.0.batch_norm1.weight", "linear_stages.0.batch_norm2.weight",
        "linear_stages.1.batch_norm1.weight", "linear_stages.1.batch_norm2.weight", "w2.weight"
    ]
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        # print(key)
        key_ends = key.split('.')[-1]
        if key_ends == 'running_mean':
            key_starts = key.split('running_mean')[0]
            key = key_starts + '_mean'
        if key_ends == 'running_var':
            key_starts = key.split('running_var')[0]
            key = key_starts + '_variance'
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        print(key)
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)


transfer()
