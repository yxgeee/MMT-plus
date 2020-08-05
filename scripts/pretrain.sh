#!/bin/sh
SOURCE=$1
ARCH=$2
SEED=$3

python examples/source_pretrain.py -ds ${SOURCE} -a ${ARCH} --seed ${SEED} --margin 0.0 --height 384 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 120 --eval-step 10 \
  --logs-dir logs/source-pretrain/${SOURCE}-${ARCH}-${SEED}
