#!/bin/bash
# NOTE that this requires GNU parallel (apt-get install parallel)
# Config YAML for model, training details
EXP=experiments/baselines/resnet34.yaml
# Launch using this config
BASE="python3.5 src/main.py --config $EXP"
# Where GNU parallel should redirect stdout/stderr logs
STDOUT_LOGDIR=./stdout
# List of commands to run in paralllel
COMMANDS=(
	"$BASE --gpus 0 --batch_sizes 128"
	"$BASE --gpus 3 --batch_sizes 256"
)
parallel --results $STDOUT_LOGDIR ::: "${COMMANDS[@]}" 
