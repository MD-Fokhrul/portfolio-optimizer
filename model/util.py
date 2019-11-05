import torch
import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


# convert numpy arrays to torch tensors with device information (cpu/cuda)
def to_tensor(np_arr, device=torch.device('cpu')):
    return torch.from_numpy(np_arr).float().to(device)


def extract_tensors_from_buffer_map(buffer_map, device=torch.device('cpu')):
    tensors = []
    for key in buffer_map.keys():
        tensors.append(to_tensor(buffer_map[key], device))  # add device
    return tensors


# convert modules to cuda if required
def to_cuda_if_needed(modules, device_type):
    if device_type == 'cuda':
        if type(modules) is not list:
            modules = [modules]

        for module in modules:
            module.cuda()

