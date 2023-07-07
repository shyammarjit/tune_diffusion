#!/bin/bash

subjects=("monster_toy")
lora_rank=("8" "4")
LEARNING_RATE=("1e-4" "2e-3")
LEARNING_RATE_TEXT=("5e-5")
ALPHA_TEXT=("1")
ALPHA_UNET=("1")

export OUTPUT_DIR="/home/btech/ayush.singh/alpha_outputs"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

for dataset in "${subjects[@]}"; do
    export INSTANCE_DIR="/home/btech/ayush.singh/dataset/${dataset}"
    
    for rank in "${lora_rank[@]}"; do
        for lr in "${LEARNING_RATE[@]}"; do
            for lr_text in "${LEARNING_RATE_TEXT[@]}"; do
                accelerate launch train_lora_dreambooth.py \
                    --pretrained_model_name_or_path="$MODEL_NAME"  \
                    --instance_data_dir="$INSTANCE_DIR" \
                    --output_dir="$OUTPUT_DIR" \
                    --train_text_encoder \
                    --resolution=512 \
                    --train_batch_size=1 \
                    --gradient_accumulation_steps=1 \
                    --learning_rate="$lr" \
                    --learning_rate_text="$lr_text" \
                    --color_jitter \
                    --lr_scheduler="constant" \
                    --lr_warmup_steps=0 \
                    --max_train_steps=10 \
                    --lora_or_krona=1 \
                    --lora_rank="$rank" \
                    --alpha_text="$ALPHA_TEXT" \
                    --alpha_unet="$ALPHA_UNET"
            done
        done
    done
done
