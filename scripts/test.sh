#!/bin/sh
ARCH=$1
MODEL=$2

ARCHC=$3
MODELC=$4

python -u examples/test.py -d target_val \
    -a ${ARCH} --resume ${MODEL} \
    -ac ${ARCHC} --camera ${MODELC} \
    --rerank --k1 30 --k2 6 --lambda-value 0.3 --dsbn
