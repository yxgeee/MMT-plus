#!/bin/sh
ARCH=$1
MODEL=$2
PARTITION=$3

ARCHC=resnest50

srun --mpi=pmi2 -p $PARTITION \
      -n1 --gres=gpu:1 --ntasks-per-node=1 \
      --job-name=test \
python -u examples/test.py -d target_val -a ${ARCH} -ac ${ARCHC} --resume $MODEL \
    --camera logs/camera_id/${ARCHC}/model_best.pth.tar \
    --rerank --k1 30 --k2 6 --lambda-value 0.3 --dsbn
