#!/bin/sh
ARCH=$1

METRIC=arc
MS=64
MM=0.35

PARTITION=$2

srun --mpi=pmi2 -p $PARTITION \
      -n1 --gres=gpu:4 --ntasks-per-node=1 \
      --job-name=mmt \
python examples/mmt_train_uda.py \
  -ds personx -a ${ARCH} --iters 200 --height 384 --fp16 \
  --eps 0.6 --k1 20 --k2 6 --cluster-alg dbscan --min-samples 4 --moco-neg 1 \
  --margin 0.0 --soft-ce-weight 0.5 --soft-tri-weight 0.5 --alpha 0.999 --dropout 0 --mc-weight 0.1 \
  --init-1 logs/source-pretrain/personx_sda-${ARCH}-1/model_best.pth.tar \
  --init-2 logs/source-pretrain/personx_sda-${ARCH}-2/model_best.pth.tar \
  -m ${METRIC} -ms ${MS} -mm ${MM} \
  --logs-dir logs/mmt/${ARCH}-${METRIC}-s${MS}-m${MM}
