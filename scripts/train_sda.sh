PARTITION=$1
ARCH=resnet_ibn50a

srun --mpi=pmi2 -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=sda \
python -u examples/sda_train.py --gan_mode lsgan --netD basic --model sda \
    -a ${ARCH} --lambda_rc 1.0 --eval-step 10 \
    --num-instances 4 -b 32 -j 4 --dropout 0 --display_id 0 --display_port 6089 --seed 1 \
    --lr_policy linear --niter 50 --niter_decay 50 --iters 200 --height 384 \
    --init-s logs/source-pretrain/personx-${ARCH}-1/model_best.pth.tar \
    --init-t logs/cluster-baseline/${ARCH}/model_best.pth.tar \
    --logs-dir logs/sda-${ARCH}
