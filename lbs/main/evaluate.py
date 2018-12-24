"""
Loads several checkpointed DNNs from a single experiment across several seeds
and evaluates their training, validation, and test performance.
"""

from absl import app
from absl import flags

from lbs.evaluate import single_experiment_result
from lbs.utils import seed_all

flags.DEFINE_string('experiment_directory', './logs',
                    'location of a single experiment')


def _main(_):
    seed_all(1)  # any consistent seed should work
    single_experiment_result(flags.FLAGS.experiment_directory)


if __name__ == '__main__':
    app.run(_main)
