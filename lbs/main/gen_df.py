"""
Generates a dataframe of all existing evaluated runs in the logroot
with keys specified by the flattened flags json.

The dataframe is then presumably used to make various visuals.
"""

import os
import pickle
import itertools
import json

from absl import app
from absl import flags
import numpy as np
import pandas as pd

flags.DEFINE_string('logroot', './logs', 'where to look for logs')
flags.DEFINE_string('outfile', 'results.pkl', 'pickled pandas output df')


def _main(_):
    rows = []
    for experiment in os.listdir(flags.FLAGS.logroot):
        experiment_dir = os.path.join(flags.FLAGS.logroot, experiment)
        eval_file = os.path.join(experiment_dir, 'evaluation.pkl')
        if not os.path.isfile(eval_file):
            continue
        with open(eval_file, 'rb') as result_file:
            results = pickle.load(result_file)
        seed2idx = results['batch_idxs']
        if not seed2idx:
            continue
        print('reading experiment', experiment)
        idx = next(iter(seed2idx.values()))
        for other_idx in seed2idx.values():
            np.testing.assert_equal(idx, other_idx)
        for seed in seed2idx:
            seed_dir = os.path.join(experiment_dir, 'seed-{}'.format(seed))
            with open(os.path.join(seed_dir, 'flags.json'), 'r') as flags_file:
                seed_flags = json.load(flags_file)
                row = dict(
                    itertools.chain(*(v.items() for v in seed_flags.values())))
            row['train'] = results['train'][seed]
            row['val'] = results['val'][seed]
            row['test'] = results['test'][seed]
            row['idx'] = idx
            rows.append(row)
    print('writing out', len(rows), 'rows')
    pd.DataFrame(rows).to_pickle(flags.FLAGS.outfile)


if __name__ == '__main__':
    app.run(_main)
