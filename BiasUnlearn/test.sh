exp=gpt-m-new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1,2,3,4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch \
  --num_processes 4 \
  --mixed_precision fp16 \
  --multi_gpu \
  train.py \
  --ster_batch_size 2 \
  --batch_size 4 \
  --model_name openai-community/gpt2-large \
  --use_lora \
  --model_save_dir "${exp}" \
  --log_file logs/${exp}.log \
  --lr 2e-5 \
  --max_unlearn_steps 1000 \
  --save_every 100 \
  --ster_weight 0.5 \
  --anti_weight 0.3 \
  --kl_weight 0.2 \
  --mix_anti \
  2>&1 | tee "logs/${exp}.tee.log"