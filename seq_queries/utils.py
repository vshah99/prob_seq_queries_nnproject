import torch
import torch.nn as nn
import math

from datetime import datetime


class Log(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'log': Log, 
    'identity': Identity,
    'gelu': GELU,
    'softplus': nn.Softplus,
}


def print_log(*args):
    """Custom print function that records time of function call."""
    print("[{}]".format(datetime.now()), *args)

def truncated_normal(size, scale=1, limit=2):
    """Samples a tensor from an approximately truncated normal tensor.
    
    Arguments:
        size {tuple of ints} -- Size of desired tensor
    
    Keyword Arguments:
        scale {int} -- Standard deviation of normal distribution (default: {1})
        limit {int} -- Number of standard deviations to truncate at (default: {2})
    
    Returns:
        torch.FloatTensor -- A truncated normal sample of requested size
    """
    return torch.fmod(torch.randn(size),limit) * scale

def xavier_truncated_normal(size, limit=2, no_average=False):
    """Samples from a truncated normal where the standard deviation is automatically chosen based on size."""
    if isinstance(size, int):
        size = (size,)
    
    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2
    
    return truncated_normal(size, scale=(1/n_avg)**0.5, limit=2)

def flatten(list_of_lists):
    """Turn a list of lists (or any iterable) into a flattened list."""
    return [item for sublist in list_of_lists for item in sublist]