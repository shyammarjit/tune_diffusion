subjects="teapot"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 423008, 32 min approx => without text encoder
# 823392, not stable  to run 
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# 
# 520928, with text encoder
# 398048, without text encoder

export OUTPUT_DIR="/home/nmathur/shyam"
export INSTANCE_DIR="/home/nmathur/dataset/tune_diffusion/${subjects}"

attn_update_unet="kqvo"
attn_update_text="kqvo"
# unet parameters
a1=64
a2=4
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
steps=50

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
    --adapter_type="krona" \
    --seed="0" \
    --diffusion_model="base" \
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
    --krona_text_k_rank_a1=$a1 \
    --krona_text_k_rank_a2=$a2 \
    --krona_text_q_rank_a1=$a1 \
    --krona_text_q_rank_a2=$a2 \
    --krona_text_v_rank_a1=$a1 \
    --krona_text_v_rank_a2=$a2 \
    --krona_text_o_rank_a1=$a1 \
    --krona_text_o_rank_a2=$a2 \
    --attn_update_text=$attn_update_text \
    --train_text_encoder \
    # --unet_tune_mlp \
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
#     --adapter_type="krona" \
#     --seed="0" \
#     --diffusion_model="base" \
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
#     # --attn_update_text=$attn_update_text \
#     # --train_text_encoder \
#     # --delete_and_upload_drive