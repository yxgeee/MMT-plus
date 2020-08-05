ARCH=$1
SOURCE_MODEL=$2
TARGET_MODEL=$3

python -u examples/sda_train.py --gan_mode lsgan --netD basic --model sda \
    -a ${ARCH} --lambda_rc 1.0 --eval-step 10 \
    --num-instances 4 -b 32 -j 4 --dropout 0 --display_id 0 --display_port 6089 --seed 1 \
    --lr_policy linear --niter 50 --niter_decay 50 --iters 200 --height 384 \
    --init-s ${SOURCE_MODEL} --init-t ${TARGET_MODEL} \
    --logs-dir logs/sda-${ARCH}
