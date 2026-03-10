# 仓库默认基线 A
exp=gpt-m-new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
  --model_name openai-community/gpt2 \
  --use_lora \
  --model_save_dir ${exp}_repo \
  --log_file logs/${exp}_repo.log \
  --lr 4e-5 \
  --max_unlearn_steps 300 \
  --save_every 50 \
  --ster_batch_size 4 \
  --batch_size 28 \
  --ster_weight 0.5 \
  --anti_weight 0.3 \
  --kl_weight 0.2 \
  --mix_anti