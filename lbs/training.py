"""
The training module contains the main training loop for all neural networks.

This module orchestrates:

* recovering the model from a checkpoint
* intermittently saving intermediate versions of the model
* sampling mini batches every iteration and performing a GD-based optimization
  step
* intermittently logging loss metrics
* performing early stopping

Note we sample with replacement.
"""
import itertools
import os

from absl import flags
import torch

from . import dataset
from . import log
from . import optimizers
from .utils import gpus, RollingAverageWindow, repackage_hidden,\
    compute_num_mini_mini_batches

flags.DEFINE_integer('max_batches', 50, 'maximum number of mini-batches.')
flags.DEFINE_integer(
    'max_epochs', None, 'overrides max_batches by setting a'
    ' fixed number of epochs of training')
flags.DEFINE_integer('batch_size', 32, 'mini-batch size')
flags.DEFINE_string('restore_checkpoint', None, 'checkpoint to restore '
                    'training from')
flags.DEFINE_integer(
    'max_samples_per_gpu', None, 'max samples per GPU'
    ' this is a purely physical constraint; it has nothing to'
    ' do with the algorithm, though it should divide '
    'batch_size evenly')
flags.DEFINE_integer(
    'evaluate_every', 10,
    'period of mini-batches between evaluations (0 to disable)')
flags.DEFINE_integer(
    'persist_every', 25,
    'period of mini-batches between evaluations (0 to disable)')
flags.DEFINE_integer(
    'evaluate_n', None,
    'evaluate at a period such that there are about n total evaluations'
    ' overrides evaluate_every.')
flags.DEFINE_integer('persist_n', None,
                     'like evaluate_n but for persist_every')
flags.DEFINE_integer('eval_batches', 100,
                     'number of mini-batches to use to print diagnostic info')
flags.DEFINE_integer(
    'hidden_reset_period', 512,
    'number of iterations before we reset hidden state, since'
    ' normally we reset per-epoch')
flags.DEFINE_float('grad_clip', -1,
                   'Optional gradient clipping. Disable if <= 0')


def do_training(all_training_data, model, store_hidden=False):
    """Performs gradient-based supervised training to optimize
    loss(model(x), y) where (x, y) is a tuple contained in
    the torch.utils.data.Dataset data"""

    device = _check_gpu()
    torch.save(
        model.to(torch.device('cpu')),
        os.path.join(log.logging_directory(), 'untrained_model.pth'))
    model = model.to(device)

    train, val = dataset.split_dataset(all_training_data)
    train_loader = dataset.create_loader(train, flags.FLAGS.dataset,
                                         flags.FLAGS.batch_size)

    # TODO if val loss doesn't affect the generalization gap then
    # let's get rid of this dataset splitting
    optimizer = optimizers.build_optimizer(model.parameters())
    training_state = _TrainingState()
    if flags.FLAGS.restore_checkpoint:
        log.debug('restoring model from {}', flags.FLAGS.restore_checkpoint)
        _load_checkpoint(flags.FLAGS.restore_checkpoint, model, optimizer,
                         training_state)
    loss_window = RollingAverageWindow(flags.FLAGS.eval_batches)

    # Initialize hidden state if we're using an LSTM
    nb = compute_num_mini_mini_batches(flags.FLAGS.batch_size)
    mini_mini_batch_size = flags.FLAGS.batch_size // nb

    if store_hidden:
        hidden = model.init_hidden(mini_mini_batch_size)
    else:
        hidden = None

    _override_flags(len(train))

    model.train()
    for x, y in train_loader:
        training_state.batch_idx += 1  # intentional 1-indexing
        if training_state.batch_idx > flags.FLAGS.max_batches:
            break

        # Reset the hidden state every x sequences so that we don't
        # get screwed up by differently-sized datasets
        if store_hidden and training_state.batch_idx \
                % flags.FLAGS.hidden_reset_period == 0:
            hidden = model.init_hidden(mini_mini_batch_size)

        # TODO timing would be nice
        def _closure(compute_backward=False, hidden=None):
            xs = x if store_hidden else x.chunk(nb)  # pylint: disable=cell-var-from-loop
            ys = y if store_hidden else y.chunk(nb)  # pylint: disable=cell-var-from-loop
            mini_batch_loss = 0

            for small_x, small_y in zip(xs, ys):
                small_x = small_x.to(device)
                small_y = small_y.to(device)

                if store_hidden:
                    hidden = repackage_hidden(hidden)
                    mini_mini_batch_loss, hidden = model(
                        small_x, small_y, hidden)
                else:
                    mini_mini_batch_loss = model(small_x, small_y)

                if compute_backward:
                    mini_mini_batch_loss.backward(
                        torch.ones(()).to(device) / nb)
                with torch.no_grad():
                    mini_batch_loss += mini_mini_batch_loss.item() / nb
            return mini_batch_loss, hidden

        optimizer.zero_grad()
        mini_batch_loss, hidden = _closure(
            compute_backward=True, hidden=hidden)
        loss_window.update(mini_batch_loss)
        if flags.FLAGS.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          flags.FLAGS.grad_clip)
        optimizer.step()

        # Apply the linear learning rate schedule
        optimizer = optimizers.lr_schedule(flags.FLAGS.batch_size,
                                           training_state.batch_idx,
                                           optimizer)

        if _check_period(training_state.batch_idx, flags.FLAGS.evaluate_every):
            val_diagnostics = _sample_average_over_dataset(
                val, model, device, store_hidden)
            if store_hidden:
                hidden = model.init_hidden(mini_mini_batch_size)

            fmt1 = '{:' + str(len(str(flags.FLAGS.max_batches))) + 'd}'
            fmt2 = '{:' + str(
                len(
                    str(flags.FLAGS.max_batches * flags.FLAGS.batch_size //
                        len(train)))) + 'd}'
            log_str = 'batch ' + fmt1 + '/' + fmt1
            for k, v in val_diagnostics.items():
                log_str += ' (sampled) val {0} {1} '.format(k, v)
            log_str += 'train loss {:8.4g} epochs ' + fmt2
            log.debug(
                log_str, training_state.batch_idx, flags.FLAGS.max_batches,
                loss_window.value(), training_state.batch_idx *
                flags.FLAGS.batch_size // len(train))
            model.train()

        if _check_period(training_state.batch_idx, flags.FLAGS.persist_every):
            fmt = '{:0' + str(len(str(flags.FLAGS.max_batches))) + 'd}.pth'
            checkpoint_file = os.path.join(
                log.logging_directory(), 'checkpoints',
                fmt.format(training_state.batch_idx))
            log.debug('persisting model to {}', checkpoint_file)
            _save_checkpoint(checkpoint_file, model, optimizer, training_state)


def _check_gpu():
    # multiple GPUs would require DistributedDataParallel (DataParallel
    # doesn't use NCCL and puts all outputs on gpu0). That complexity
    # isn't worth it right now.
    gpulist = gpus()
    log.debug('using gpus: {}', gpulist)
    assert len(gpulist) <= 1, gpulist
    if len(gpulist) == 1:
        return torch.device('cuda')
    return torch.device('cpu')


def _sample_average_over_dataset(all_data, model, device, store_hidden=False):
    model.eval()  # Disable dropout, etc.
    loader = dataset.create_loader(
        all_data, flags.FLAGS.dataset,
        flags.FLAGS.batch_size,
        endless=False)

    nb = compute_num_mini_mini_batches(flags.FLAGS.batch_size)
    mini_mini_batch_size = flags.FLAGS.batch_size / nb

    if store_hidden:
        hidden = model.init_hidden(mini_mini_batch_size)

    with torch.no_grad():
        diagnostics = {}
        iteration = 1
        for x, y in itertools.islice(loader, flags.FLAGS.eval_batches):
            if store_hidden and iteration \
                    % flags.FLAGS.hidden_reset_period == 0:
                hidden = model.init_hidden(mini_mini_batch_size)
            iteration += 1

            if store_hidden:
                xs = x
                ys = y
            else:
                xs = x.chunk(nb)  # pylint: disable=cell-var-from-loop
                ys = y.chunk(nb)  # pylint: disable=cell-var-from-loop

            for small_x, small_y in zip(xs, ys):
                small_x = small_x.to(device)
                small_y = small_y.to(device)

                if store_hidden:
                    hidden = repackage_hidden(hidden)
                    mini_diagnostics, hidden = model.diagnose(
                        small_x, small_y, hidden)
                else:
                    mini_diagnostics = model.diagnose(small_x, small_y)

                for k, v in mini_diagnostics.items():
                    if k in diagnostics.keys():
                        diagnostics[k] += v / (flags.FLAGS.eval_batches * nb)
                    else:
                        diagnostics[k] = v / (flags.FLAGS.eval_batches * nb)
    return diagnostics


def _check_period(idx, period):
    if period == 0:
        return False
    return idx == 1 or idx == flags.FLAGS.max_batches or idx % period == 0


def _load_checkpoint(checkpoint_file, model, optimizer, training_state):
    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    training_state.load_state_dict(state_dict['training_state'])


def _save_checkpoint(checkpoint_file, model, optimizer, training_state):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_state': training_state.state_dict()
    }

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        torch.save(state_dict, f)


class _TrainingState:
    def __init__(self):
        self.batch_idx = 0

    def state_dict(self):
        """return all training state for algorithm"""
        return {'batch_idx': self.batch_idx}

    def load_state_dict(self, d):
        """re-load training state from dictionary"""
        self.batch_idx = d['batch_idx']


def _override_flags(train_len):
    if flags.FLAGS.max_epochs:
        flags.FLAGS.max_batches = (
            flags.FLAGS.max_epochs * train_len // flags.FLAGS.batch_size)
    if flags.FLAGS.warmup_epochs:  # defined in optimizer.py
        flags.FLAGS.warmup_batch_idx = (
            flags.FLAGS.warmup_epochs * train_len // flags.FLAGS.batch_size)
    if flags.FLAGS.anneal_epochs: # defined in optimizer.py
        flags.FLAGS.anneal_iters = [x * train_len // flags.FLAGS.batch_size
                                    for x in flags.FLAGS.anneal_epochs]
    if flags.FLAGS.evaluate_n:
        flags.FLAGS.evaluate_every = max(
            flags.FLAGS.max_batches // flags.FLAGS.evaluate_n, 1)
    if flags.FLAGS.persist_n:
        flags.FLAGS.persist_every = max(
            flags.FLAGS.max_batches // flags.FLAGS.persist_n, 1)
