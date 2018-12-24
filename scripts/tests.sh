#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this.
#
#   ./scripts/tests.sh
#   ./scripts/tests.sh --dry-run

set -eo pipefail

set -u

if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "--dry-run" ] ; then
    echo 'usage: ./scripts/tests.sh [--dry-run]' 1>&2
    exit 1
fi

if [ $# -eq 1 ] ; then
    DRY_RUN="true"
else
    DRY_RUN="false"
fi

box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    cmd=""
    function note_failure {
        box "${cmd}"
    }
    trap note_failure EXIT

    cmds=()
    train_main="lbs/main/train.py"
    cmds+=("rm -rf ./test_logs/*")
    cmds+=("python $train_main --seed 3 --max_batches 1000 --persist_every 0 --evaluate_every 500 --logroot ./test_logs/_test")
    cmds+=("test -f ./test_logs/_test/*/seed-3/githash.txt")
    cmds+=("test -f ./test_logs/_test/*/seed-3/flags.json")
    cmds+=("test -f ./test_logs/_test/*/seed-3/flags.flags")
    cmds+=("test -f ./test_logs/_test/*/seed-3/invocation.txt")
    cmds+=("test -f ./test_logs/_test/*/seed-3/starttime.txt")
    cmds+=("python $train_main --seed 3 --max_batches 500 --persist_every 0 --evaluate_every 500 --max_samples_per_gpu 16 --batch_size 32 --logroot test_logs")
    cmds+=("rm -rf ./test_logs/_test_restore")
    cmds+=("python $train_main --seed 1 --max_batches 4 --evaluate_every 0 --persist_every 2 --logroot ./test_logs/_test_restore")
    cmds+=("python $train_main --seed 1 --max_batches 4 --evaluate_every 0 --restore_checkpoint ./test_logs/_test_restore/*/seed-1/checkpoints/2.pth --logroot test_logs")
    cmds+=("python $train_main --seed 1 --max_batches 1 --evaluate_every 1 --persist_every 0 --dataset cifar10 --model wideresnet28 --eval_batches 3 --logroot test_logs --batch_size 2 --max_samples_per_gpu 2")
    # Language modeling/LSTM tests
    cmds+=("python $train_main --seed 1 --max_batches 4 --evaluate_every 1 --persist_every 2 --dataset wikitext-2 --model lstm --eval_batches 3 --logroot test_logs")
    # Plotting tests
    cmds+=("rm -rf ./test_logs/*")
    cmds+=("python lbs/main/train.py --max_batches 10 --persist_every 5 --evaluate_every 0 --batch_size 32 --logroot test_logs/_test_plot_32")
    cmds+=("python lbs/main/train.py --max_batches 10 --persist_every 5 --evaluate_every 0 --batch_size 64 --logroot test_logs/_test_plot_64")
    cmds+=("python lbs/main/plot.py --experiment_directory test_logs/_test_plot_32/* --outfile test_logs/32test.pdf")
    cmds+=('python lbs/main/plot.py --experiment_directory test_logs/_test_plot_32/* --outfile test_logs/32test_acc.pdf --diagnostic accuracy')
    cmds+=("python lbs/main/plot.py --validation --experiment_directory test_logs/_test_plot_32/* --outfile test_logs/32val.pdf")
    cmds+=("mkdir -p test_logs/all_batches")
    cmds+=("cp -r test_logs/_test_plot_32/* test_logs/_test_plot_64/* test_logs/all_batches")
    cmds+=("python lbs/main/plot.py --groupby batch_size --experiment_directory test_logs/all_batches --outfile test_logs/allbatches.pdf")
    cmds+=("test -f ./test_logs/32test.pdf")
    cmds+=("test -f ./test_logs/32test_acc.pdf")
    cmds+=("test -f ./test_logs/32val.pdf")
    cmds+=('test -f ./test_logs/allbatches.pdf')
    # Resnet tests
    cmds+=("python $train_main --seed 1 --max_batches 1 --evaluate_every 1 --persist_every 0 --dataset cifar10 --model resnet34 --eval_batches 1 --logroot test_logs --batch_size 2 --max_samples_per_gpu 2")
    # Linear LR tests
    cmds+=("python $train_main --seed 1 --max_batches 10 --evaluate_every 1 --persist_every 0 --dataset mnist --model lenet --eval_batches 1 --logroot test_logs --linear_lr --warmup_batch_idx 3  --batch_size 2 --max_samples_per_gpu 2")
    # DRN tests
    cmds+=("python $train_main --use_fake_data True --seed 1 --max_batches 1 --evaluate_every 1 --persist_every 0 --dataset cityscapes --model drn_d_22 --eval_batches 1 --logroot test_logs --batch_size 2 --max_samples_per_gpu 2")

    # Check ray evaluation works
    cmds+=("rm -rf ./test_logs/test_rayoutput")
    cmds+=("python lbs/main/ray_experiment.py --experiment_name testray --config experiments/unit_test.yaml --logroot test_logs/test_rayoutput --self_host 1 --cpu")
    # unit_test.yaml should produce two experiments with 2 seeds each, so we need two distinct
    # output directories here
    cmds+=('test $(find ./test_logs/test_rayoutput -name seed-[12] | wc -l ) -eq 4')

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            bash -c "$cmd"
        fi
    done

    trap '' EXIT
}

main
