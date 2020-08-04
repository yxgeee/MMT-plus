#!/bin/sh
PARTITION=$1

srun --mpi=pmi2 -p $PARTITION \
      -n1 --gres=gpu:1 --ntasks-per-node=1 \
      --job-name=test \
python -u examples/test_ensemble.py -d target_val \
    --rerank --k1 30 --k2 6 --lambda-value 0.3 --dsbn --flip \
    -ac resnest50 \
    --camera logs/camera/resnest50/model_best.pth.tar \
    -a \
    resnest50 resnest101 densenet_ibn169a resnext_ibn101a \
    --resume \
    logs/mmt/resnest50-arc-s64-m0.35/model_best.pth.tar \
    logs/mmt/resnest101-cos-s30-m0.4/model_best.pth.tar \
    logs/mmt/densenet_ibn169a-arc-s30-m0.3/model_best.pth.tar \
    logs/mmt/resnext_ibn101a-arc-s64-m0.35/model_best.pth.tar
