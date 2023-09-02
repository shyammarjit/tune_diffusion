subjects="teapot"
rank=4
rank_a1=16
rank_a2=4
attn_update_unet="kqvo"
attn_update_text="kvqo"
lr=1e-4
steps=1
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/ai/HSN/test"
export INSTANCE_DIR="/home/ai/HSN/dataset/${subjects}"

accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks${subjects}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --adapter_type="lora" \
    --lora_rank=$rank \
    --seed="0" \
    --diffusion_model="sdxl" \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --gradient_checkpointing \
    --attn_update_unet=$attn_update_unet \
    --tune_mlp
    # --attn_update_text=$attn_update_text \
    # --train_text_encoder \

# python3 generator.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --instance_data_dir=$INSTANCE_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --mixed_precision="fp16" \
#     --instance_prompt="a photo of sks${subjects}" \
#     --resolution=1024 \
#     --train_batch_size=1 \
#     --gradient_accumulation_steps=4 \
#     --learning_rate=$lr \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=0 \
#     --max_train_steps=$steps \
#     --adapter_type="slice_lora" \
#     --lora_rank=$rank \
#     --seed="0" \
#     --diffusion_model="sdxl" \
#     --enable_xformers_memory_efficient_attention \
#     --use_8bit_adam \
#     --gradient_checkpointing \
#     --attn_update_unet=$attn_update_unet \
#     --attn_update_text=$attn_update_text \
#     --train_text_encoder \
#     # --delete_and_upload_drive