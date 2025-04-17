source /var/scratch/yshen/anaconda3/etc/profile.d/conda.sh
conda activate fourierft
module load gcc
module load cuDNN/cuda11.1/8.0.5
module load cuda11.1/toolkit/11.1.1

export HF_HOME=/var/scratch/yshen/.cache/huggingface
export PIP_CACHE_DIR=/var/scratch/yshen/.cache/pip

CUDA_VISIBLE_DEVICES=1 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset boolq \
    --task boolq \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.008 \
    --fft_lr 0.12 \
    --num_epoch 100 \
    --bs 32  \
    --scale 49.2 \
    --seed 0 \
    --share_entry
#   --dataset boolq \
#   --task boolq \
# "copa, record,wsc,wic,cb, rte, multirc"
