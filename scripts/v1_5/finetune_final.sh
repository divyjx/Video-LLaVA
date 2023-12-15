DATA_ROOT="data_root"
DATA_ROOT_IMAGE="images"
DATA_ROOT_VIDEO="videos"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path tuning_data/merged_200k.json \
    --video_folder ${DATA_ROOT_VIDEO} \
    --image_folder ${DATA_ROOT_IMAGE} \
    --X "Video" "Image" \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --image_tower LanguageBind/LanguageBind_Image \
    --pretrain_mm_mlp_adapter checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/output/Video-LLaVA-7B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"