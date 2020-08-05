#!/bin/sh
ARCH=$1

python examples/camera_train.py -a ${ARCH} --seed 1 --margin 0.0 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 \
  --iters 200 --epochs 120 --eval-step 10 \
	--logs-dir logs/camera/${ARCH}
