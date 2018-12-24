import numpy as np
import glob
import argparse
import logging
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_line(s):
    s = s[1:-2].strip().replace(' ','').replace('"','')
    fields = s.split(',')
    d = {}
    for field_val_pair in fields:
        k, v = field_val_pair.split(':')
        v = float(v)
        d[k] = v
    return d

def get_field_arr(dicts, field='test_acc'):
    result = [0.]*len(dicts)
    for d in dicts:
        epoch = int(d['epoch'])
        field_val = d[field]
        result[epoch - 1] = field_val
    return result

def plot_field(dicts, field, label):
    values = get_field_arr(dicts, field)
    plt.plot(values, label=label)

def aggregate_field(dicts, field, mode='max'):
    values = get_field_arr(dicts, field)
    if mode == 'max':
        return max(values)
    else:
        return min(values)

def parse_batch_size(file_str):
    bs_ind = file_str.index('bs_')
    after_bs = file_str[bs_ind + 3:]
    batch_size = int(after_bs[:after_bs.index('_')])
    return batch_size

def process_across_experiments(logdir, experiments, fields, export_dir, agg_modes):
    """
    For each (experiment, field, BS) tuple, calculate max value of field for that BS run of experiment.
    Then, for each experiment, plot a line of BS vs. max field val.
    Determine 'max' or 'min' aggeregate for each field via agg_modes dict.

    Returns:
        2D dictionary d where d[field][experiment] gives me results for each batch size.
    """
    assert len(experiments) > 0, "Need experiments to look at!"

    results = {}
    for field in fields:
        results[field] = {}
        logging.info('\tRunning for field {}'.format(field))
        for experiment in experiments:
            logging.info('\t\tCalculating BS stats for experiment {}'.format(experiment))
            log_file_strs = glob.glob(os.path.join(logdir, experiment + '_*/log.txt'))
            assert len(log_file_strs) > 0, "Need logs for experiment {}!".format(experiment)
            bs_log_file_pairs = sorted(list(map(lambda log_file_str: \
                    (parse_batch_size(log_file_str), log_file_str), log_file_strs)),
                    key=lambda x: x[0])

            for batch_size, log_file_str in bs_log_file_pairs:
                if not os.path.isfile(log_file_str):
                    raise IOError('Log file not found! {}'.format(log_file_str))

                with open(log_file_str) as log_file:
                    parsed_lines = list(map(parse_line, log_file.readlines()))
                    agg_field_value = aggregate_field(parsed_lines, field, agg_modes[field])
                    if experiment in results[field].keys():
                        results[field][experiment].append(agg_field_value)
                    else:
                        results[field][experiment] = [agg_field_value]
        return results

def plot_across_experiments(logdir, experiments, fields, export_dir, agg_modes, batch_sizes):
    results = process_across_experiments(logdir, experiments, fields, export_dir, agg_modes)
    logging.info('\tPlotting results for fields across all experiments')
    for field in fields:
        experiments = results[field].keys()
        for experiment in experiments:
            values = results[field][experiment]
            plt.plot(batch_sizes[:len(values)], values, label=experiment)
        plt.xlabel('Batch size')
        plt.ylabel('{0} {1}'.format(agg_modes[field], field))
        plt.legend(loc='upper right')
        export_file_str = os.path.join(export_dir, 'all_experiments_' + agg_modes[field] + '_' + field + '.png')
        plt.savefig(export_file_str)
        plt.clf()


def plot(logdir, experiment, fields, export_dir, calc_max=False):
    """
    For each field, look through all <logdir>/<experiment>_*/log.txt files
    and plot that field's value across all epochs, then save the plot to <export_dir>.

    logdir: save directory to look through for logs
    experiment: experiment name from general.name field in exp. YAML
    fields: list of fields to plot [e.g. ['train_loss','test_acc']]
    export_dir: save location for plots [e.g. ./plots/]
    """
    log_file_strs = glob.glob(os.path.join(logdir, experiment + '_*/log.txt'))
    assert len(log_file_strs) > 0, "Need to have logs to parse!"

    for field in fields:
        logging.info('\tPlotting field {}'.format(field))
        bs_log_file_pairs = sorted(list(map(lambda log_file_str: \
                (parse_batch_size(log_file_str), log_file_str), log_file_strs)),
                key=lambda x: x[0])

        for batch_size, log_file_str in bs_log_file_pairs:
            if not os.path.isfile(log_file_str):
                raise IOError('Log file not found! {}'.format(log_file_str))

            with open(log_file_str) as log_file:
                parsed_lines = list(map(parse_line, log_file.readlines()))
                if calc_max:
                    max_field_val = aggregate_field(parsed_lines, field, 'max')
                    print("BS: {0}\tMAX FOR {1}: {2}".format(str(batch_size), field, max_field_val))
                else:
                    plot_field(parsed_lines, field, "BS: {}".format(str(batch_size)))

        if not calc_max:
            plt.xlabel('epoch')
            plt.ylabel(field)
            plt.legend(loc='upper right')
            export_file_str = os.path.join(export_dir, experiment + '_' + field + '.png')
            plt.savefig(export_file_str)
            plt.clf()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', action='store_true')
    parser.add_argument('--across', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('experiment', type=str, help='./logs/prefix.... e.g. baseline for parsing ./logs/baseline*')
    parser.add_argument('--fields', type=str, help='fields to plot', default='test_acc')
    parser.add_argument('--export', type=str, default='./plots/', help='export directory for plots')
    parser.add_argument('--agg', default='min', type=str, help="min/max comma-separated list for each experiment")
    parser.add_argument('--experiments', default='baseline', type=str, help="comma-separated list of experiments to aggregate on")
    parser.add_argument('--batch_sizes', default='128', type=str)
    args = parser.parse_args()

    fields = args.fields.split(',')
    assert len(fields) > 0, "need some fields to plot"

    if args.across:
        batch_sizes = args.batch_sizes.split(',')
        experiments = args.experiments.split(',')
        agg_modes = {}
        for field, mode in zip(fields, args.agg.split(',')):
            agg_modes[field] = mode
        plot_across_experiments(args.logdir, experiments, fields, args.export, agg_modes, batch_sizes)
    else:
        plot(args.logdir, args.experiment, fields, args.export, args.max)
