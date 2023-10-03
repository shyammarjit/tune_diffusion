# WARNING: Currently having issue! Don't Run!
subjects="teapot"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/nmathur/test_text"
export INSTANCE_DIR="/home/nmathur/dataset/tune_diffusion/${subjects}"

attn_update_unet="kqvo"
attn_update_text="kqvo"
# unet parameters
unet_lora_rank_k=4
unet_lora_rank_q=4
unet_lora_rank_v=4
unet_lora_rank_out=4
unet_lora_rank_mlp=4

# text encoder parameters
text_lora_rank_k=4
text_lora_rank_q=4
text_lora_rank_v=4
text_lora_rank_out=4

lr=1e-3
steps=2


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
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --attn_update_text=$attn_update_text \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --attn_update_text=$attn_update_text \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --train_text_encoder \
    # --unet_tune_mlp \    

python3 generator_test.py \
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
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --attn_update_text=$attn_update_text \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --attn_update_text=$attn_update_text \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --train_text_encoder \
    # --unet_tune_mlp \
    # --delete_and_upload_drive