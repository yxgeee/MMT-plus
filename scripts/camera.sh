#!/bin/sh
ARCH=resnest50
SEED=1

PARTITION=$1

srun --mpi=pmi2 -p ${PARTITION} \
      -n1 --gres=gpu:1 --ntasks-per-node=1 \
      --job-name=camera \
python examples/camera_train.py -a ${ARCH} --seed ${SEED} --margin 0.0 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 \
  --iters 200 --epochs 120 --eval-step 10 \
	--logs-dir logs/camera/${ARCH}
