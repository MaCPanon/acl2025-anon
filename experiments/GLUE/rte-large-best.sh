CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-large \
    --dataset rte \
    --task rte \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.005 \
    --fft_lr 0.08 \
    --num_epoch 60 \
    --bs 32  \
    --scale 90.0 \
    --seed 0 \
    --share_entry
    
