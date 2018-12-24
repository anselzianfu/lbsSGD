#!/bin/bash
# ./makeflags.sh baseline-lr
# Programatically make flag files for resnet-on-cifar10 examples

LR="$1"
LINEAR="False"
DATASIZE=60000
WARMUP_EPOCHS=8
# BS_ARR=(32 256 512 1024 2048)
#BATCHES_ARR=(100000 80000 60000 40000 15000)
# Small-batch version
BS_ARR=(8 16 64 128)
BATCHES_ARR=(120000 120000 100000 70000)

for ((i=0;i<${#BS_ARR[@]};++i)); do
    echo "DEVICE $i"
    BS="${BS_ARR[i]}"
    BATCHES="${BATCHES_ARR[i]}"
    PERSIST_EVERY=$(( $BATCHES / 50 ))
    EVAL_EVERY=$(( $BATCHES / 500 ))
    WARMUP_BATCH_IDX=$(( $WARMUP_EPOCHS * $DATASIZE / $BS ))

    FILE="$(dirname $0)/bs$(printf '%06d\n' $BS)-lr${LR}.flags"
    rm -f $FILE
    echo "--model=lstm" >> $FILE
    echo "--dataset=wikitext-2" >> $FILE
    echo "--learning_rate=$LR" >> $FILE
    echo "--batch_size=$BS" >> $FILE
    echo "--max_batches=$BATCHES" >> $FILE
    echo "--persist_every=$PERSIST_EVERY" >> $FILE
    echo "--evaluate_every=$EVAL_EVERY" >> $FILE
    echo "--eval_batches=4" >> $FILE
    echo "--max_samples_per_gpu=256" >> $FILE
    echo "--linear_lr=$LINEAR" >> $FILE
    echo "--baseline_bz=$BS" >> $FILE
    echo "--warmup_batch_idx=$WARMUP_BATCH_IDX" >> $FILE
    echo "--device=$i" >> $FILE
    echo "output flags into $FILE"
done
