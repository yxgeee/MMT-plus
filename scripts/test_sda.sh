RESUME=$1

python -u examples/sda_infer.py --model test -b 128 -j 8 --height 384 \
    --resume ${RESUME}
