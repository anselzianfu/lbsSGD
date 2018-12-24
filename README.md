
# Large Batch Sizes for Neural Networks

## Setup

See `setup.py` for necessary python packages. Requires a linux x64 box.

```
conda create -y -n lbs-env python=3.5
source activate lbs-env
./scripts/install-pytorch.sh
pip install --no-cache-dir --editable .
```

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root in the `lbs-env`.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `tests.sh` | runs tests |
| `install-pytorch.sh` | infer python and cuda versions, use them to install pytorch |
| `format.sh` | auto-format the entire `lbs` directory |
| `where.sh` | finds the experiments whose params match the given ones on the command line |
| `annotate.sh` | annotates experiment metadata (when it was run, etc.) |

## Example

All mainfiles are documented. Run `python lbs/main/*.py --help` for any `*` for details.

```
for SEED in 1 2 3 ; do
  python lbs/main/train.py --flagfile experiments/wideresnet28_cifar10.flags --seed 1 --batch_size 64 --experiment_name wideresnet28_cifar10_bs64
done
python lbs/main/plot.py --experiment_name wideresnet28_cifar10_bs64
```

To, e.g., run a Facebook-style linear scaling rule with a warmup, we might do the following:

```
# Let's run the cifar10 experiment on our local 8-gpu machine.
# This still takes like a week.
GIT_HASH=$(git rev-parse HEAD)
python lbs/main/ray_experiment.py --experiment_name cifar10 --config experiments/cifar10.yaml --self_host 8

# here we can annotate the runs to see which ones we have available
./scripts/where.sh ./logs git_hash=$GIT_HASH dataset=cifar10 model=resnet20 batch_size=32 | ./scripts/annotate.sh

./scripts/where.sh ./logs git_hash=$GIT_HASH dataset=cifar10 | xargs -d '\n' -P $(nproc) -I {} bash -c 'python lbs/main/plot.py --outfile "./plots/cifar10/{model}-bs{batch_size}-lr{learning_rate}-linear{linear_lr}.pdf" --plot_key lbs.main.train:model --plot_key lbs.training:batch_size --plot_key lbs.optimizers:learning_rate --plot_key lbs.optimizers:linear_lr --experiment_directory {}'
```

## Code structure

Here are the important files.

* `lbs/main/train.py` - the main training invocation, trains a neural net see its `--help` for details
* `lbs/main/evaluate.py` - runs evaluation for all checkpoints of models in a directory on training val and test sets, see `--help` for details.
* `lbs/main/plot.py` - evaluates and plots train and test performance during the course of training for a list of checkpointed networks, see its `--help` for details
* `lbs/training.py` - the main training routine, loops over mini-batches, persisting and evaluating the network occasionally
* `lbs/models/` - model zoo, add a model to this directory and make sure `build_model()` in `lbs/models/__init__.py` creates it when the corresponding model name is passed int
* `lbs/dataset.py` - dataset zoo, like the model zoo but for datasets
