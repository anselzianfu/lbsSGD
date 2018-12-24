"""
Performs DNN training in a supervised learning setting.

Given a specification for:

* what data to use (lbs/dataset.py)
* what model to use (lbs/models/__init__.py)
* how long to train and how often to record metrics (lbs/training.py)
* how to train (lbs/optimization.py)

Running this will perform the specified optimization, using all
GPUs made available by CUDA_VISIBLE_DEVICES (or by default all GPUs).

The spec can be supplied with command-line flags or a flagfile
(use --flagfile).

Logs are written to {logroot}/{flags hash}/seed-{seed},
where the flag hash is is of the flags
after the seed value is removed.
"""

from absl import app
from absl import flags

import lbs
from lbs.utils import seed_all

flags.DEFINE_integer('seed', 1, 'random seed')

flags.DEFINE_string('dataset', 'mnist', 'what dataset to use')

flags.DEFINE_string('model', 'lenet', 'neural network architecture')


def main(argv):
    """see train.py module documentation"""
    seed_all(flags.FLAGS.seed)
    exp_name = lbs.log.flaghash_dirname([argv[0]], ['seed'])
    lbs.log.init(exp_name)

    lbs.log.debug('fetching dataset {}', flags.FLAGS.dataset)
    train, _, dataset_params = lbs.dataset.from_name(flags.FLAGS.dataset)

    lbs.log.debug('building model {}', flags.FLAGS.model)
    model, store_hidden = lbs.models.build_model(flags.FLAGS.model,
                                                 **dataset_params)

    num_parameters = int(sum(p.numel() for p in model.parameters()))
    lbs.log.debug('number of parameters in model {}', num_parameters)

    lbs.training.do_training(train, model, store_hidden)


if __name__ == '__main__':
    app.run(main)
