"""
Loads several checkpointed DNNs from a single experiment directory across all
seeds withi that directory and evaluates their training, validation, and test
performance.
"""
import json
import os
import itertools
import pickle

from absl import flags
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

import lbs

flags.DEFINE_string(
    'results_fname', 'evaluation.pkl',
    'Filename where evaluation dictionary should be '
    'saved within experiment directory')

flags.DEFINE_integer(
    'eval_batch_size', 512, 'max samples per GPU for eval'
    ' this is a purely physical constraint; it has nothing to'
    ' do with the algorithm, though it should divide '
    'batch_size evenly')
flags.DEFINE_integer('num_eval_batches', 0,
                     'number of samples (0 implies use the whole dataset)')

flags.DEFINE_string('groupby', None,
                    'experiment flag to group by for trial plotting')


def _group_experiment_dirs(root, groupby=None):
    """
    Given a root directory containing several experiments with multiple seeds
    each, groups all experiment directories by the groupby key
    """
    grouped_directories = {}
    experiment_subdirectories = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    for exp in experiment_subdirectories:
        exp_dir = os.path.join(root, exp)
        # the next line assumes each seed has same flags config
        main_seed_dir = list(filter(_is_seed_dir, os.listdir(exp_dir)))[0]
        network_dir = os.path.join(exp_dir, main_seed_dir)
        flag_value = _parse_flag_key(network_dir, key=groupby)
        if flag_value in grouped_directories.keys():
            grouped_directories[flag_value].append(exp_dir)
        else:
            grouped_directories[flag_value] = [exp_dir]
    return grouped_directories


def single_experiment_result(exp_dir, training_module='lbs/main/train.py'):
    """
    Evaluate the runs of a single experiment configuration,
    or return the cached results if we have already run this analysis.
    Calculates train, val, test loss as well as other diagnostic
    information e.g. accuracy or perplexity.

    exp_dir: directory of a single experiment containing seeds
             as subdirectories.

    returns a dictionary with train/val/test/batch_idxs as keys.
    We store the diagnostic information for each mode over all seeds.
    """
    results_fname = os.path.join(exp_dir, flags.FLAGS.results_fname)
    if not os.path.isfile(results_fname):
        return _evaluate_experiment(exp_dir, training_module)
    with open(results_fname, 'rb') as results:
        return pickle.load(results)


def all_experiment_results(exp_dir,
                           groupby=None,
                           training_module='lbs/main/train.py'):
    """
    Evaluate all runs of all experiment configurations,
    grouped by some configuration parameter e.g. batch size.

    exp_dir: directory containing results for all relevant configurations.

    returns a dictionary with CL-argument values as keys.

    each value contains the diagnostic information for the
    batch_ids/train/val/test modes across all experiment seeds
    whose argument value matches the key.
    """
    grouped_directories = _group_experiment_dirs(exp_dir, groupby)
    vals = ['batch_idxs', 'train', 'val', 'test']
    raw_results = {}
    for k, dirs in grouped_directories.items():
        per_seed_results = []
        for d in dirs:
            single_exp = single_experiment_result(d, training_module)
            # Exp result stored like
            # 'batch_idxs: {seed: inds},
            # 'train': {seed: vals}, ...
            seeds = single_exp['batch_idxs'].keys()
            for seed in seeds:
                seed_result = {val: single_exp[val][seed] for val in vals}
                per_seed_results.append(seed_result)
        raw_results[k] = per_seed_results
    results = {}
    for k, key_results in raw_results.items():
        results[k] = {}
        for v in vals:
            results[k][v] = [r[v] for r in key_results]
    return results


def _evaluate_experiment(logdir, training_module='lbs/main/train.py'):
    """
    Evaluates all seeds for a single experiment in {logdir}.
    Calculates train, val, test loss and optionally exports results to a
    pickle file in the same directory.

    returns dictionary with keys:
        batch_idxs: batch indices for each seed
        train_diagnostics: train diagnostics dict for each seed
        val_diagnostics: validation diagnostics dict for each seed
        test_diagnostics: test diagnostics dict for each seed
    """
    # Iterations for which we logged the model
    idx = {}
    # Dictionaries that store diagnostics like loss, accuracy, etc.
    train_diagnostics = {}
    val_diagnostics = {}
    test_diagnostics = {}

    for seed_dir in os.listdir(logdir):
        if _is_seed_dir(seed_dir):
            seed_idx, seed_train, seed_val, seed_test = _evaluate(
                os.path.join(logdir, seed_dir), training_module)
            seed = int(seed_dir[len('seed-'):])
            idx[seed] = seed_idx
            train_diagnostics[seed] = seed_train
            val_diagnostics[seed] = seed_val
            test_diagnostics[seed] = seed_test

            seed_idx = np.array(seed_idx)
            perm = seed_idx.argsort()
            idx[seed] = np.asarray(seed_idx[perm])
            # Sort all arrays for all diagnostics
            for d in [train_diagnostics, val_diagnostics, test_diagnostics]:
                for k, vals in d[seed].items():
                    if vals:  # b/c val could be empty so array may be empty due to
                        # "if k"
                        # in val list comprehension
                        d[seed][k] = np.array(vals)[perm]
            final_results = {
                'batch_idxs': idx,
                'train': train_diagnostics,
                'val': val_diagnostics,
                'test': test_diagnostics
            }
            outf_name = os.path.join(logdir, 'evaluation-{}.pkl'.format(seed))
            lbs.log.debug('saving evaluation results to: {}'.format(outf_name))
            with open(outf_name, 'wb') as outf:
                pickle.dump(final_results, outf)
            return final_results


def _parse_flag_key(seed_dir, module='lbs.training', key='batch_size'):
    """
    Parses the value of a flag for the seed of a particular experiment
    """
    flags_fname = os.path.join(seed_dir, 'flags.json')
    with open(flags_fname) as flags_file:
        seed_flags = json.load(flags_file)
        # HACK: try replacing / with .
        if module not in seed_flags.keys():
            module = module.replace('/', '.')[:-len('.py')]
        module_flags = seed_flags[module]
        return module_flags[key]


def parse_flag_key(experiment_dir, key, module='lbs.training'):
    """extract the key from the training module's flags in the first available
    seed directory for an experiment directory"""
    for seed_dir in os.listdir(experiment_dir):
        if _is_seed_dir(seed_dir):
            seed_dir = os.path.join(experiment_dir, seed_dir)
            return _parse_flag_key(seed_dir, module=module, key=key)
    raise ValueError('no runs in dir {}'.format(experiment_dir))


def _evaluate(network_dir, training_module):
    print('evaluating network in', network_dir)
    dataset_name = _parse_flag_key(
        network_dir, module=training_module, key='dataset')
    model_name = _parse_flag_key(
        network_dir, module=training_module, key='model')
    is_hidden = lbs.models.needs_hidden_state(model_name)
    train, test, _ = lbs.dataset.from_name(dataset_name)
    train_idx = torch.load(os.path.join(network_dir, 'train_indices.pth'))
    val_idx = torch.load(os.path.join(network_dir, 'val_indices.pth'))
    val = data.dataset.Subset(train, val_idx)
    train = data.dataset.Subset(train, train_idx)

    batch_idxs = []
    train_diagnostics = []
    val_diagnostics = []
    test_diagnostics = []
    model = torch.load(os.path.join(network_dir, 'untrained_model.pth'))
    ckpt_dir = os.path.join(network_dir, 'checkpoints')
    for ckpt_file in os.listdir(ckpt_dir):
        # TODO the output of this (inner loop) should be cached in ./data
        if not ckpt_file.endswith('.pth'):
            continue
        print('    checkpoint', ckpt_file)
        state_dict = torch.load(os.path.join(ckpt_dir, ckpt_file))
        model.load_state_dict(state_dict['model'])
        train_iter = state_dict['training_state']['batch_idx']
        batch_idxs.append(train_iter)
        train_diagnostics.append(
            _average_over_dataset(train, model, dataset_name, is_hidden))
        val_diagnostics.append(
            _average_over_dataset(val, model, dataset_name, is_hidden))
        test_diagnostics.append(
            _average_over_dataset(test, model, dataset_name, is_hidden))

    # Diagnostic info currenty stored like [{'loss': x1}, {'loss': x2}] per
    # iter. flatten it out here to {'loss': [x1, x2]}
    final_train_diag = {}
    final_val_diag = {}
    final_test_diag = {}
    for k in train_diagnostics[0].keys():
        final_train_diag[k] = [iter[k] for iter in train_diagnostics]
        final_val_diag[k] = [iter[k] for iter in val_diagnostics if k in iter]
        final_test_diag[k] = [iter[k] for iter in test_diagnostics]
    return batch_idxs, final_train_diag, final_val_diag, final_test_diag


def _average_over_dataset(all_data, model, dataset_name, store_hidden=False):
    # assume 1 gpu at most
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        model.eval()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loader = lbs.dataset.create_loader(all_data, dataset_name,
                                       flags.FLAGS.eval_batch_size,
                                       endless=False)
    nb = lbs.utils.compute_num_mini_mini_batches(flags.FLAGS.eval_batch_size)
    mini_mini_batch_size = flags.FLAGS.eval_batch_size // nb
    diagnostics = {}
    n_seen = 0
    with torch.no_grad():
        if flags.FLAGS.num_eval_batches:
            loader = itertools.islice(loader, flags.FLAGS.num_eval_batches)
        if store_hidden:
            hidden = model.init_hidden(mini_mini_batch_size)
        for x, y in tqdm(loader):
            xs = x if store_hidden else x.chunk(nb)  # pylint: disable=cell-var-from-loop
            ys = y if store_hidden else y.chunk(nb)  # pylint: disable=cell-var-from-loop
            for small_x, small_y in zip(xs, ys):
                # Hacky handling of  empty batches for corpus batching
                if small_x.nelement() == 0:
                    continue
                n_seen += small_x.size(0)
                small_x = small_x.to(device)
                small_y = small_y.to(device)
                if store_hidden:
                    hidden = lbs.utils.repackage_hidden(hidden)
                    mini_diagnostics, hidden = model.diagnose(
                        small_x, small_y, hidden)
                else:
                    mini_diagnostics = model.diagnose(small_x, small_y)
                for k, v in mini_diagnostics.items():
                    if k not in diagnostics.keys():
                        diagnostics[k] = v * small_x.size(0)
                    else:
                        diagnostics[k] += v * small_x.size(0)
    diagnostics = {k: v / n_seen for k, v in diagnostics.items()}
    return diagnostics


def _is_seed_dir(seed_dir):
    return seed_dir.startswith('seed-') and seed_dir[len('seed-'):].isdigit()
