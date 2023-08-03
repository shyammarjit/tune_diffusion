# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
#export MODEL_NAME="stabilityai/stable-diffusion-xl-base-0.9"

export INSTANCE_DIR="/home/rtx4090/HSN/dataset/dog6"
export OUTPUT_DIR="dreambooth_output"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 