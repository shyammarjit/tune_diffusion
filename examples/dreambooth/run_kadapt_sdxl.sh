export MODEL_NAME="stabilityai/stable-diffusion-xl-base-0.9"
export INSTANCE_DIR="/home/ai/HSN/dataset/cat"
export OUTPUT_DIR="/home/ai/HSN/outputs"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of a cat" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2 \
  --adapter_type="krona" \
  --lora_rank=4 \
  --seed="0" \
  --diffusion_model="sdxl" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam

# python3 generator.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="fp16" \
#   --instance_prompt="a photo of a cat" \
#   --resolution=1024 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=500 \
#   --adapter_type="lora" \
#   --lora_rank=4 \
#   --seed="4" \
#   --diffusion_model="sdxl" \
#   --enable_xformers_memory_efficient_attention \
#   --gradient_checkpointing \
#   --use_8bit_adam