"""
Handles plotting results for a single experiment or across all experiments.
"""

import os
import itertools
import pickle

from absl import app
from absl import flags
import numpy as np
from matplotlib.lines import Line2D

import lbs
from lbs.evaluate import parse_flag_key
from lbs.utils import import_matplotlib

flags.DEFINE_bool(
    'validation', False,
    'If true, plots the average training and validation curves. If false, plots training and test curves, as well as horizontal lines for final performance using an common tolerance-based early stopping rule'
)

flags.DEFINE_string(
    'outfile', 'loss_curves.pdf', 'out file for image; can '
    'contain python format values of the plot_key type. '
    'E.g., if you supply --plot_key lbs.training/batch_size '
    'then the outfile loss-bs{batch_size}.pdf will be written '
    'out with the interpolated batch_size value')

flags.DEFINE_multi_string(
    'plot_key', 'lbs.training:batch_size',
    'a flag name in the form of module:flagname '
    'to use in the title')

flags.DEFINE_string(
    'diagnostic', 'loss',
    'Diagnostic to plot on the y-axis; e.g. accuracy for classification tasks, perplexity for language modeling tasks'
)

flags.DEFINE_boolean(
    'early_stop', 'True',
    'If True, draw horizontal line across error plot using an early stopping'
    'heuristic based on validation error')

flags.DEFINE_boolean(
    'final', False,
    'If True and groupby is set,  plot key vs. best diagnostic value')
flags.DEFINE_string('experiment_directory', './logs',
                    'location of a single experiment')


def _plot_experiment(results, batch_size, diagnostic='loss'):
    """
    Plot the results of a single experiment configuration.
    """
    first_seed = list(results['batch_idxs'].keys())[0]
    shared_index = results['batch_idxs'][first_seed]
    # convert shared_index of iterations to number of samples seen
    shared_index = shared_index * batch_size

    train_diagnostic = np.vstack(
        [vals[diagnostic] for vals in results['train'].values()])
    val_diagnostic = np.vstack(
        [vals[diagnostic] for vals in results['val'].values()])
    test_diagnostic = np.vstack(
        [vals[diagnostic] for vals in results['test'].values()])

    plt = import_matplotlib()
    plt.clf()

    plt.xlabel('Number of samples seen')
    plt.ylabel('{}'.format(diagnostic))
    title = []
    plot_keys = {}
    for key in flags.FLAGS.plot_key:
        module, flag = key.split(':')
        flagval = parse_flag_key(
            flags.FLAGS.experiment_directory, module=module, key=flag)
        title.append(flag + '=' + str(flagval))
        plot_keys[flag] = flagval
    title = ' '.join(title)
    plt.title(title)
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    _plot_quantiles(plt, next(colors), shared_index, train_diagnostic, 'train')

    if flags.FLAGS.validation:
        _plot_quantiles(plt, next(colors), shared_index, val_diagnostic, 'val')
        stop_indices = _get_early_stopping(val_diagnostic)
        true_val_errors = []
        for i, stopping_idx in enumerate(stop_indices):
            true_val_errors.append(val_diagnostic[i][stopping_idx])
        plt.axhline(
            np.mean(true_val_errors),
            label='early stop val',
            ls=':',
            c=next(colors))
    else:
        _plot_quantiles(plt, next(colors), shared_index, test_diagnostic,
                        'test')
        stop_indices = _get_early_stopping(val_diagnostic)
        true_val_errors = []
        for i, stopping_idx in enumerate(stop_indices):
            true_val_errors.append(test_diagnostic[i][stopping_idx])
        plt.axhline(
            np.mean(true_val_errors),
            label='early stop test',
            ls=':',
            c=next(colors))

    outfile = flags.FLAGS.outfile.format(**plot_keys)
    plt.legend()
    _mkdir(outfile)
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


def _plot_quantity_across_experiments(plt,
                                      results_dict,
                                      linestyle='-',
                                      mode='train',
                                      diagnostic='loss',
                                      prefix=None,
                                      show_early_stop=False):
    """
    Plots a single quantity, e.g. train_loss, across all experiment
    configurations when gropued by a given key.
    """
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    keys = sorted(list(results_dict.keys()))

    for k in keys:
        assert isinstance(
            k,
            int), "patch to use old evaluation.pkl-s: only groupby batch size"
        color = next(colors)
        values = [v[diagnostic] for v in results_dict[k][mode]]
        label = prefix + str(k) if prefix is not None else None
        # scale by batch_size k to get num samples seen
        index = k * results_dict[k]['batch_idxs'][0]
        _plot_quantiles(plt, color, index, values, label, linestyle=linestyle)
        if show_early_stop:
            val_loss = [v['loss'] for v in results_dict[k]['val']]
            stop_indices = _get_early_stopping(val_loss)
            true_val_errors = []
            for i, stopping_idx in enumerate(stop_indices):
                true_val_errors.append(values[i][stopping_idx])
            plt.axhline(np.mean(true_val_errors), ls=':', c=color)


def _plot_final_quantity_across_experiments(results_dict,
                                            diagnostic='loss',
                                            early_stop=False):
    """
    Plots final train and test loss versus configuration, e.g.
    early stopping loss vs. batch size to see a generalization gap
    """
    plt = import_matplotlib()
    plt.clf()
    plt.xlabel(flags.FLAGS.groupby)
    plt.ylabel('final {}'.format(diagnostic))
    plt.title('Final {0} vs. {1}'.format(diagnostic, flags.FLAGS.groupby))
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    keys = sorted(list(results_dict.keys()))
    train_y_values = []
    test_y_values = []
    for k in keys:
        train_values = [v[diagnostic] for v in results_dict[k]['train']]
        mode = 'val' if flags.FLAGS.validation else 'test'
        key_values = [v[diagnostic] for v in results_dict[k][mode]]
        train_values = [v[diagnostic] for v in results_dict[k]['train']]
        val_loss = [v['loss'] for v in results_dict[k]['val']]
        # Find stopping indices in either case
        if early_stop:
            best_inds = _get_early_stopping(val_loss)
        else:
            best_inds = [-1 for x in val_loss]
        best_test_values = [arr[i] for i, arr in zip(best_inds, key_values)]
        best_train_values = [arr[i] for i, arr in zip(best_inds, train_values)]
        # TODO get quantiles when we have a different number of seeds per key
        train_y_values.append(np.mean(best_train_values))
        test_y_values.append(np.mean(best_test_values))
    test_y_values = np.array(test_y_values)[:, None].T
    train_y_values = np.array(train_y_values)[:, None].T
    _plot_quantiles(
        plt, next(colors), keys, train_y_values, 'train', linestyle='-')
    _plot_quantiles(
        plt, next(colors), keys, test_y_values, mode, linestyle='-')
    plt.legend()

    outfile = flags.FLAGS.outfile
    _mkdir(outfile)
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


def _plot_across_experiments(results_dict,
                             diagnostic='loss',
                             show_early_stop=False):
    """
    Plots train and test loss across all experiment configurations
    when the evaluation was performed by grouping by a given key.

    If key_as_x_axis is True, the x-axis value is the keys,
    and the y-axis value is the best/early stopping error.
    """
    plt = import_matplotlib()
    plt.clf()

    plt.xlabel('Number of samples seen')
    plt.ylabel('{} (cross-entropy)'.format(diagnostic))
    plt.title('learning curves vs. {0}'.format(flags.FLAGS.groupby))
    _plot_quantity_across_experiments(
        plt,
        results_dict,
        linestyle='-',
        mode='train',
        diagnostic=diagnostic,
        prefix='BS ',
        show_early_stop=False)
    if flags.FLAGS.validation:
        _plot_quantity_across_experiments(
            plt,
            results_dict,
            linestyle='--',
            mode='val',
            diagnostic=diagnostic,
            prefix=None,
            show_early_stop=show_early_stop)
    else:
        _plot_quantity_across_experiments(
            plt,
            results_dict,
            linestyle='--',
            mode='test',
            diagnostic=diagnostic,
            prefix=None,
            show_early_stop=show_early_stop)
    # Add legends, including base train vs. test legend
    val_label = 'Val' if flags.FLAGS.validation else 'Test'
    custom_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Train'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label=val_label)
    ]
    if show_early_stop:
        custom_lines.append(
            Line2D(
                [
                    0,
                ], [0],
                color='black',
                lw=2,
                linestyle=':',
                label='Early Stop'))
    main_leg = plt.legend(loc='upper right')
    plt.legend(handles=custom_lines, bbox_to_anchor=(.7, 1), loc='upper right')
    plt.gca().add_artist(main_leg)

    outfile = flags.FLAGS.outfile
    _mkdir(outfile)
    plt.savefig(outfile, format='pdf', bbox_inches='tight')


def _plot_quantiles(plt, color, idx, values, label, linestyle='-'):
    values = np.vstack(values)
    q1, med, q3 = np.percentile(values, q=[25, 50, 75], axis=0)
    plt.plot(idx, q1, ls=':', color=color)
    plt.plot(idx, med, ls=linestyle, color=color, label=label)
    plt.plot(idx, q3, ls=':', color=color)
    plt.fill_between(idx, q1, q3, alpha=0.25, color=color)


def _get_early_stopping(val_loss):
    # equivalent to choosing minimum-validation model after training
    # with early stopping and a "keras patience" of 3
    stop_indices = []
    for seed_val_losses in val_loss:
        running_min = np.minimum.accumulate(seed_val_losses)
        perf_drop = running_min < seed_val_losses
        starts, runs = _get_runs(perf_drop)
        if np.any(runs >= 3):
            violation_idx = starts[np.argmax(runs >= 3)]
            stop_indices.append(np.argmin(seed_val_losses[:violation_idx]))
        else:
            stop_indices.append(len(seed_val_losses) - 1)
    return stop_indices


def _get_runs(arr):
    runs = [np.flatnonzero(arr[1:] ^ arr[:-1]) + 1]
    if arr[0]:
        runs.insert(0, [0])
    if arr[-1]:
        runs.append([len(arr)])
    runs = np.concatenate(runs)
    starts = runs[::2]
    ends = runs[1::2]
    return starts, ends - starts


def _main(_):
    groupby = flags.FLAGS.groupby
    if groupby:
        results = lbs.evaluate.all_experiment_results(
            flags.FLAGS.experiment_directory, groupby)
        if flags.FLAGS.final:
            _plot_final_quantity_across_experiments(
                results, flags.FLAGS.diagnostic, flags.FLAGS.early_stop)
        else:
            _plot_across_experiments(
                results,
                flags.FLAGS.diagnostic,
                show_early_stop=flags.FLAGS.early_stop)
    else:
        results = lbs.evaluate.single_experiment_result(
            flags.FLAGS.experiment_directory)
        batch_size = lbs.evaluate.parse_flag_key(
            flags.FLAGS.experiment_directory,
            'batch_size',
            module='lbs.training')
        _plot_experiment(results, batch_size, flags.FLAGS.diagnostic)


def _mkdir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


if __name__ == '__main__':
    flags.mark_flag_as_required('experiment_directory')
    app.run(_main)
