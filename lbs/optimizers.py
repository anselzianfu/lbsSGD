"""
Contains specification for the optimization performed by
training.
"""

from absl import flags
from torch import optim
import math

flags.DEFINE_float(
    'learning_rate', 0.01, 'learning rate, or baseline learning rate with '
    'linear scaling')

flags.DEFINE_boolean('linear_lr', False, 'using linear learning rate or not')
flags.DEFINE_boolean('sqrt_lr', False, 'use sqrt scaling rule')

flags.DEFINE_integer('baseline_bz', 32,
                     'baseline batch size for linear learning rate schedule')

flags.DEFINE_integer('warmup_batch_idx', 100,
                     'warm up batch idx for linear learning rate schedule')

# a bit unintuitively used in do_training()...
flags.DEFINE_integer(
    'warmup_epochs', None,
    'overrides warmup_batch_idx, defining number of warmup '
    'epochs to train')

flags.DEFINE_boolean('anneal', False, 'use geometric annealing')
flags.DEFINE_float('anneal_rate', 0.1, 'geometric factor to anneal by')
flags.DEFINE_multi_integer('anneal_epochs', [60, 120, 180],
                           'epochs to anneal on. overrides anneal_iters.')
flags.DEFINE_multi_integer('anneal_iters', [80*300, 120*300, 160*300],
                           'iterations to anneal on.')


def lr_schedule(batch_size, batch_idx, optimizer):
    """ schedule the learning rate by linear LR schedule  w/warmup """
    if flags.FLAGS.linear_lr or flags.FLAGS.sqrt_lr:
        bs_ratio = float(batch_size) / flags.FLAGS.baseline_bz
        baseline_lr = flags.FLAGS.learning_rate
        final_lr = baseline_lr * bs_ratio
        if flags.FLAGS.sqrt_lr:
            final_lr = math.sqrt(final_lr)
        if flags.FLAGS.sqrt_lr or bs_ratio < 1 or batch_idx >= flags.FLAGS.warmup_batch_idx:
            lr = final_lr
        else:
            lr = (final_lr - baseline_lr)  \
                / flags.FLAGS.warmup_batch_idx * batch_idx \
                + baseline_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if flags.FLAGS.anneal:
        if batch_idx in flags.FLAGS.anneal_iters:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * flags.FLAGS.anneal_rate
    return optimizer


def build_optimizer(params):
    """Return the optimizer per the command line argument specification."""
    return optim.SGD(
        filter(lambda p: p.requires_grad, params),
        lr=flags.FLAGS.learning_rate)
