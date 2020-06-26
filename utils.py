import numpy as np
import os
import torch
from scipy import sparse
import constants
from torch import nn


def cooc_path(args, key):
    if args[key] is None: return
    args[key] = os.path.join(constants.RC['cooccurrence_dir'], args[key])


def emb_path(args, key):
    if args[key] is None: return
    args[key] = os.path.join(constants.RC['embeddings_dir'], args[key])


def corpus_path(args, key):
    if args[key] is None: return
    args[key] = os.path.join(constants.RC['embeddings_dir'], args[key])


def get_device(device=None):
    return constants.RC['device'] if device is None else device


def get_dtype(dtype=None):
    return constants.RC['dtype'] if dtype is None else dtype


def pmi(Nxx, Nx, Nxt, N):
    return torch.log((Nxx / N) * (N / Nx) * (N / Nxt))


def load_shard(
        source,
        shard=None,
        dtype=constants.DEFAULT_DTYPE,
        device=None):
    # Handle Scipy sparse matrix types
    if isinstance(source, (sparse.csr_matrix, sparse.lil_matrix)):
        shard = shard or slice(None)
        return torch.tensor(source[shard].toarray(), dtype=dtype, device=device)

    # Handle Numpy matrix types
    elif isinstance(source, np.matrix):
        return torch.tensor(
            np.asarray(source[shard]), dtype=dtype, device=device)

    # Handle primitive values (don't subscript with shard).
    elif isinstance(source, (int, float)):
        return torch.tensor(source, dtype=dtype, device=device)

    # Handle Numpy arrays and lists.
    return torch.tensor(source[shard], dtype=dtype, device=device)


def norm(array_or_tensor, ord=2, axis=None, keepdims=False):
    if isinstance(array_or_tensor, np.ndarray):
        return np.linalg.norm(array_or_tensor, ord, axis, keepdims)
    elif isinstance(array_or_tensor, torch.Tensor):
        if axis is None:
            return torch.norm(array_or_tensor, ord)
        else:
            return torch.norm(array_or_tensor, ord, axis, keepdims)
    raise ValueError(
        'Expected either numpy ndarray or torch Tensor.  Got %s'
        % type(array_or_tensor).__name__
    )


def normalize(array_or_tensor, ord=2, axis=None):
    if axis is None:
        raise ValueError('axis must be specifiec (int)')

    return array_or_tensor / norm(array_or_tensor, ord, axis, keepdims=True)

def init_weight(tensor, method='xavier'):
    if method == 'orthogonal':
        torch.nn.init.orthogonal_(tensor)
    elif method == 'xavier':
        nn.init.xavier_uniform_(tensor)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(tensor)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


def count_param_num(nn_module):
    return np.sum([np.prod(param.size()) for param in nn_module.parameters() if param.requires_grad])




