subjects="cat"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/shyam/test"
export INSTANCE_DIR="/home/shyam/dataset/${subjects}"

attn_update_unet="kqvo"
# unet parameters
a1=64
a2=8
krona_unet_k_rank_a1=$a1 # k 
krona_unet_k_rank_a2=$a2 # k
krona_unet_q_rank_a1=$a1 # q
krona_unet_q_rank_a2=$a2 # q
krona_unet_v_rank_a1=$a1 # v
krona_unet_v_rank_a2=$a2 # v
krona_unet_o_rank_a1=$a1 # out
krona_unet_o_rank_a2=$a2 # out
krona_unet_ffn_rank_a1=$a1 # out
krona_unet_ffn_rank_a2=$a2 # out

lr=1e-3
steps=1000


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
    --adapter_type="krona" \
    --seed="0" \
    --diffusion_model="sdxl" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --krona_unet_k_rank_a1=$krona_unet_k_rank_a1 \
    --krona_unet_k_rank_a2=$krona_unet_k_rank_a2 \
    --krona_unet_q_rank_a1=$krona_unet_q_rank_a1 \
    --krona_unet_q_rank_a2=$krona_unet_q_rank_a2 \
    --krona_unet_v_rank_a1=$krona_unet_v_rank_a1 \
    --krona_unet_v_rank_a2=$krona_unet_v_rank_a2 \
    --krona_unet_o_rank_a1=$krona_unet_o_rank_a1 \
    --krona_unet_o_rank_a2=$krona_unet_o_rank_a2 \
    --krona_unet_ffn_rank_a1=$krona_unet_ffn_rank_a1 \
    --krona_unet_ffn_rank_a2=$krona_unet_ffn_rank_a2 \
    # --unet_tune_mlp \

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
#     --adapter_type="krona" \
#     --seed="0" \
#     --diffusion_model="sdxl" \
#     --use_8bit_adam \
#     --gradient_checkpointing \
#     --enable_xformers_memory_efficient_attention \
#     --attn_update_unet=$attn_update_unet \
#     --krona_unet_k_rank_a1=$krona_unet_k_rank_a1 \
#     --krona_unet_k_rank_a2=$krona_unet_k_rank_a2 \
#     --krona_unet_q_rank_a1=$krona_unet_q_rank_a1 \
#     --krona_unet_q_rank_a2=$krona_unet_q_rank_a2 \
#     --krona_unet_v_rank_a1=$krona_unet_v_rank_a1 \
#     --krona_unet_v_rank_a2=$krona_unet_v_rank_a2 \
#     --krona_unet_o_rank_a1=$krona_unet_o_rank_a1 \
#     --krona_unet_o_rank_a2=$krona_unet_o_rank_a2 \
#     --krona_unet_ffn_rank_a1=$krona_unet_ffn_rank_a1 \
#     --krona_unet_ffn_rank_a2=$krona_unet_ffn_rank_a2 \
#     # --unet_tune_mlp \
#     # --delete_and_upload_drive