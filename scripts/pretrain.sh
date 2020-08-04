#!/bin/sh
SOURCE=personx_sda
ARCH=$1
SEED=$2

PARTITION=$3


srun --mpi=pmi2 -p ${PARTITION} \
      -n1 --gres=gpu:4 --ntasks-per-node=1 \
      --job-name=pretrain \
python examples/source_pretrain.py -ds ${SOURCE} -a ${ARCH} --seed ${SEED} --margin 0.0 --height 384 \
	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 120 --eval-step 10 \
  --logs-dir logs/source-pretrain/${SOURCE}-${ARCH}-${SEED}
