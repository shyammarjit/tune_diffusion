subjects="teapot"
attn_update_unet="kqvo"
attn_update_text="kvqo"
# unet parameters
unet_lora_rank_k=4
unet_lora_rank_q=4
unet_lora_rank_v=4
unet_lora_rank_o=4
unet_lora_rank_mlp=4

# text encoder parameters
text_lora_rank_k=4
text_lora_rank_q=4
text_lora_rank_v=4
text_lora_rank_o=4
text_lora_rank_mlp=4

lr=1e-4
steps=500
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/nmathur/diffusion_output"
export INSTANCE_DIR="/home/nmathur/dataset/tune_diffusion/${subjects}"

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
    --seed="0" \
    --diffusion_model="sdxl" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --attn_update_unet=$attn_update_unet \
    --enable_xformers_memory_efficient_attention \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_o \
    --unet_lora_rank_mlp=$unet_lora_rank_mlp \
    --unet_tune_mlp \
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
#     --adapter_type="lora" \
#     --seed="0" \
#     --diffusion_model="sdxl" \
#     --use_8bit_adam \
#     --gradient_checkpointing \
#     --attn_update_unet=$attn_update_unet \
#     --enable_xformers_memory_efficient_attention \
#     --unet_lora_rank_k=$unet_lora_rank_k \
#     --unet_lora_rank_q=$unet_lora_rank_q \
#     --unet_lora_rank_v=$unet_lora_rank_v \
#     --unet_lora_rank_out=$unet_lora_rank_o \
#     # --unet_lora_rank_mlp=$unet_lora_rank_mlp \
#     # --unet_tune_mlp \
# #     # --attn_update_text=$attn_update_text \
# #     # --train_text_encoder \
# #     # --delete_and_upload_drive