subjects="teapot"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="/home2/shyammarjit/test"
export INSTANCE_DIR="/home2/shyammarjit/dataset/${subjects}"

attn_update_unet="kqvo"
# unet parameters
unet_lora_rank_k=4
unet_lora_rank_q=4
unet_lora_rank_v=4
unet_lora_rank_out=4
unet_lora_rank_mlp=4

lr=1e-4
steps=1000


accelerate launch train_dreambooth_lora.py \
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
    --diffusion_model="base" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --attn_update_unet=$attn_update_unet \
    --enable_xformers_memory_efficient_attention \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --unet_lora_rank_mlp=$unet_lora_rank_mlp \
    # --unet_tune_mlp \

python3 generator.py \
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
    --diffusion_model="base" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --unet_lora_rank_k=$unet_lora_rank_k \
    --unet_lora_rank_q=$unet_lora_rank_q \
    --unet_lora_rank_v=$unet_lora_rank_v \
    --unet_lora_rank_out=$unet_lora_rank_out \
    --unet_lora_rank_mlp=$unet_lora_rank_mlp \
#    # --unet_tune_mlp \
#    # --delete_and_upload_drive