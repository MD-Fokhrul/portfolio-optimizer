import torch
import numpy as np
import os
import pickle


# for initializing nn weights
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


# convert numpy arrays to torch tensors with device information (cpu/cuda)
def to_tensor(np_arr, device=torch.device('cpu')):
    return torch.from_numpy(np_arr).float().to(device)


# extract list of tensors from (s0,r0,a0,s1) map
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


def determine_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def save_model(dir_path, actor, critic, replay_buffer, identifier):
    # TODO: save replay buffer?
    model_dir_path = '{}/{}'.format(dir_path, identifier)
    os.makedirs(model_dir_path, exist_ok=True)

    torch.save(actor.state_dict(), '{}/actor.pth'.format(model_dir_path))
    torch.save(critic.state_dict(), '{}/critic.pth'.format(model_dir_path))

    with open('{}/replay_buffer.pickle'.format(model_dir_path), 'wb') as file:
        pickle.dump(replay_buffer.buffer_map, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_dir_path, actor, critic, replay_buffer):
    actor.load_state_dict(torch.load('{}/actor.pth'.format(model_dir_path)))
    critic.load_state_dict(torch.load('{}/critic.pth'.format(model_dir_path)))

    with open('{}/replay_buffer.pickle'.format(model_dir_path), 'rb') as file:
        replay_buffer.buffer_map = pickle.load(file)









