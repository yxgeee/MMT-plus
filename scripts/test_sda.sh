RESUME=$1
PARTITION=$2

srun --mpi=pmi2 -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=infer \
python -u examples/sda_infer.py --model test -b 128 -j 8 --height 384 \
    --resume ${RESUME}
