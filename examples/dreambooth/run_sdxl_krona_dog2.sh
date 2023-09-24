subjects=("dog2")
lora_rank=("8" "4" "16" "2")
LEARNING_RATE=("1e-4" "5e-6" "2e-6" "1e-6" "5e-5" "2e-5" "1e-5" "5e-4")
steps=("400" "800" "1000" "1200")

for dataset in "${subjects[@]}"; do
    export INSTANCE_DIR="/home/btech/ayush.singh/dataset/${dataset}"
    for rank in "${lora_rank[@]}"; do
        for lr in "${LEARNING_RATE[@]}"; do
            for s in "${steps[@]}"; do
                export MODEL_NAME="stabilityai/stable-diffusion-xl-base-0.9"
                export OUTPUT_DIR="/home/btech/ayush.singh/outputs_sdxl"

                accelerate launch train_dreambooth_lora_sdxl.py \
                    --pretrained_model_name_or_path=$MODEL_NAME \
                    --instance_data_dir=$INSTANCE_DIR \
                    --output_dir=$OUTPUT_DIR \
                    --mixed_precision="fp16" \
                    --instance_prompt="a photo of sks${dataset}" \
                    --resolution=1024 \
                    --train_batch_size=1 \
                    --gradient_accumulation_steps=4 \
                    --learning_rate=$lr \
                    --lr_scheduler="constant" \
                    --lr_warmup_steps=0 \
                    --max_train_steps=$s \
                    --adapter_type="krona" \
                    --lora_rank=$rank \
                    --seed="0" \
                    --enable_xformers_memory_efficient_attention \
                    --gradient_checkpointing \
                    --use_8bit_adam 


                python3 generator.py \
                    --pretrained_model_name_or_path=$MODEL_NAME \
                    --instance_data_dir=$INSTANCE_DIR \
                    --output_dir=$OUTPUT_DIR \
                    --mixed_precision="fp16" \
                    --instance_prompt="a photo of sks${dataset}" \
                    --resolution=1024 \
                    --train_batch_size=1 \
                    --gradient_accumulation_steps=4 \
                    --learning_rate=$lr \
                    --lr_scheduler="constant" \
                    --lr_warmup_steps=0 \
                    --max_train_steps=$s \
                    --adapter_type="krona" \
                    --lora_rank=$rank \
                    --seed="0" \
                    --enable_xformers_memory_efficient_attention \
                    --gradient_checkpointing \
                    --use_8bit_adam 

            done
        done
    done
done
