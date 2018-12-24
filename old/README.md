# Documentation for old Large-BS-Experiments

First, run `pip install requirements.txt`.

## Running experiments locally
To run a single experiment, do `python3.5 src/main.py --config $CONFIG_FILE --gpus $GPUS --batch_sizes $BATCH_SIZES`
 - `$CONFIG_FILE` is a YAML file in experiments/baselines
 - `$GPUS` is a comma-separated list of GPUs to use
 - `$BATCH_SIZES` is a comma-separated list of batch sizes to run

## Running all experiments
To run all experiments in parallel:
 - Install GNU parallel: `apt-get install parallel`
 - Edit `run.sh` to add the experiments you want to run
 - Do `./run.sh`
 - stdout/stderr for each process goes to the `STDOUT_DIR` directory 
