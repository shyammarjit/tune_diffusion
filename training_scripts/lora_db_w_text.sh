#!/bin/bash

# subjects=("dog6")
# subjects=("clock")
subjects=("teapot")

# lora_rank=("8" "4" "16" "2")
lora_rank=("8" "4" "16")
LEARNING_RATE=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3")
LEARNING_RATE_TEXT=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3")
steps=("800" "1000" "1200" "1500")

export OUTPUT_DIR="/home/nmathur/outputs"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

for dataset in "${subjects[@]}"; do
    export INSTANCE_DIR="/home/nmathur/dataset/tune_diffusion/${dataset}"

    for rank in "${lora_rank[@]}"; do
        for lr in "${LEARNING_RATE[@]}"; do
            for lr_text in "${LEARNING_RATE_TEXT[@]}"; do
                for s in "${steps[@]}"; do 
                    accelerate launch train_lora_dreambooth.py \
                        --pretrained_model_name_or_path="$MODEL_NAME"  \
                        --instance_data_dir="$INSTANCE_DIR" \
                        --output_dir="$OUTPUT_DIR" \
                        --train_text_encoder \
                        --resolution=768 \
                        --train_batch_size=1 \
                        --gradient_accumulation_steps=1 \
                        --learning_rate="$lr" \
                        --learning_rate_text="$lr_text" \
                        --color_jitter \
                        --lr_scheduler="constant" \
                        --lr_warmup_steps=0 \
                        --max_train_steps="$s" \
                        --lora_or_krona=0 \
                        --lora_rank="$rank"
                done
            done
        done
    done
done
