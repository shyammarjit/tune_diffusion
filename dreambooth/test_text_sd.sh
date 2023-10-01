subjects="teapot"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="/home/nmathur/test_text_sd"
export INSTANCE_DIR="/home/nmathur/dataset/tune_diffusion/${subjects}"

attn_update_unet="kqvo"
attn_update_text="kqvo"
# unet parameters
a1=16
a2=32
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

# text encoder parameters
krona_text_k_rank_a1=$a1 # k 
krona_text_k_rank_a2=$a2 # k
krona_text_q_rank_a1=$a1 # q
krona_text_q_rank_a2=$a2 # q
krona_text_v_rank_a1=$a1 # v
krona_text_v_rank_a2=$a2 # v
krona_text_o_rank_a1=$a1 # out
krona_text_o_rank_a2=$a2 # out

lr=1e-3
steps=2


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
    --attn_update_text=$attn_update_text \
    --krona_unet_k_rank_a1=$krona_unet_k_rank_a1 \
    --krona_unet_k_rank_a2=$krona_unet_k_rank_a1 \
    --krona_unet_q_rank_a1=$krona_unet_q_rank_a1 \
    --krona_unet_q_rank_a2=$krona_unet_q_rank_a2 \
    --krona_unet_v_rank_a1=$krona_unet_v_rank_a1 \
    --krona_unet_v_rank_a2=$krona_unet_v_rank_a2 \
    --krona_unet_o_rank_a1=$krona_unet_o_rank_a1 \
    --krona_unet_o_rank_a2=$krona_unet_o_rank_a2 \
    --krona_unet_ffn_rank_a1=$krona_unet_ffn_rank_a1 \
    --krona_unet_ffn_rank_a2=$krona_unet_ffn_rank_a2 \
    --krona_text_k_rank_a1=$krona_text_k_rank_a1 \
    --krona_text_k_rank_a2=$krona_text_k_rank_a2 \
    --krona_text_q_rank_a1=$krona_text_q_rank_a1 \
    --krona_text_q_rank_a2=$krona_text_q_rank_a2 \
    --krona_text_v_rank_a1=$krona_text_v_rank_a1 \
    --krona_text_v_rank_a2=$krona_text_v_rank_a2 \
    --krona_text_o_rank_a1=$krona_text_o_rank_a1 \
    --krona_text_o_rank_a2=$krona_text_o_rank_a2 \
    --train_text_encoder \
    --unet_tune_mlp \
    # --attn_update_text=$attn_update_text \
    

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
    --adapter_type="krona" \
    --seed="0" \
    --diffusion_model="base" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --attn_update_text=$attn_update_text \
    --krona_unet_k_rank_a1=$krona_unet_k_rank_a1 \
    --krona_unet_k_rank_a2=$krona_unet_k_rank_a1 \
    --krona_unet_q_rank_a1=$krona_unet_q_rank_a1 \
    --krona_unet_q_rank_a2=$krona_unet_q_rank_a2 \
    --krona_unet_v_rank_a1=$krona_unet_v_rank_a1 \
    --krona_unet_v_rank_a2=$krona_unet_v_rank_a2 \
    --krona_unet_o_rank_a1=$krona_unet_o_rank_a1 \
    --krona_unet_o_rank_a2=$krona_unet_o_rank_a2 \
    --krona_unet_ffn_rank_a1=$krona_unet_ffn_rank_a1 \
    --krona_unet_ffn_rank_a2=$krona_unet_ffn_rank_a2 \
    --krona_text_k_rank_a1=$krona_text_k_rank_a1 \
    --krona_text_k_rank_a2=$krona_text_k_rank_a2 \
    --krona_text_q_rank_a1=$krona_text_q_rank_a1 \
    --krona_text_q_rank_a2=$krona_text_q_rank_a2 \
    --krona_text_v_rank_a1=$krona_text_v_rank_a1 \
    --krona_text_v_rank_a2=$krona_text_v_rank_a2 \
    --krona_text_o_rank_a1=$krona_text_o_rank_a1 \
    --krona_text_o_rank_a2=$krona_text_o_rank_a2 \
    --train_text_encoder \
    --unet_tune_mlp \
    # --delete_and_upload_drive