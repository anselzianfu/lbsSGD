"""
Various utility functions used across several files.
"""

import os
import collections
import hashlib
import random

from absl import flags
import numpy as np
import torch

from .log import debug


def compute_num_mini_mini_batches(batch_size):
    """ Compute number of chunks for breaking up a large batch """
    max_samples_per_gpu = (flags.FLAGS.max_samples_per_gpu or batch_size)
    max_samples_per_gpu = min(max_samples_per_gpu, batch_size)
    assert batch_size % max_samples_per_gpu == 0, (
        flags.FLAGS.max_samples_per_gpu, batch_size)
    # assumes at most one gpu
    return batch_size // max_samples_per_gpu


def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2**32, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    debug('seeding with seed {}', seed)
    np.random.seed(seed)
    rand_seed, torch_cpu_seed, torch_gpu_seed = _next_seeds(3)
    random.seed(rand_seed)
    torch.manual_seed(torch_cpu_seed)
    torch.cuda.manual_seed_all(torch_gpu_seed)


def gpus():
    """ Retrieve gpus from env var CUDA_VISIBLE_DEVICES """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpulist = list(range(torch.cuda.device_count()))
    else:
        gpulist = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpulist = list(map(int, filter(None, gpulist)))
    gpulist.sort()
    return gpulist


class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""

    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        return self._total / len(self._items)


class WrappedIterator:
    """
    Wraps a dataloder where we sample w/replacement
    so we can access its length correctly
    """

    def __init__(self, iter_fn, callable=False, len=None, endless=True):
        self.len = len
        self.endless = endless
        if callable:
            self.iterator = iter_fn
        else:
            self.iterator = lambda: iter_fn

    def __iter__(self):
        if self.endless:
            while True:
                it = self.iterator()
                for x in it:
                    yield x
        else:
            for x in self.iterator():
                yield x

    def __len__(self):
        return self.len


def repackage_hidden(h):
    """Wraps hidden states in new Tensors to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt
