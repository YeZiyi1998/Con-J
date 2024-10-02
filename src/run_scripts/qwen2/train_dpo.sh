export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=""
wandb login --relogin $WANDB_TOKENS

data_path="../../outputs/inference/zeroshot_llm_$1/dpo/dataset/"

EXP_NAME=dpo_critic_trained_$1
model_path="../../../open_models/Qwen2-7B-Instruct/"
BATCH_SIZE_PER_GPU=1
LR=5e-7

save_path="../../outputs/inference/zeroshot_llm_$1/dpo/model/$EXP_NAME"
ckpt_path="../../outputs/inference/zeroshot_llm_$1//dpo/model/$EXP_NAME/checkpoints"
MACHINE_SIZE=1
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
WORLD_SIZE=$[MACHINE_SIZE * 24]

GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/train_dpo.py \
     --save_path $save_path \
     --logging_steps 10 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --bf16 \
     --max_epochs 1 \
     --max_len 6400 \
     --zero_stage 3 \
     --l2 0.0001 \
     --eval_steps 500 \
     --save_steps 2000 \
     --beta 0.1 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --ckpt_path $ckpt_path \
     --flash_attn \
     --sft_loss 0.001 \
     --sft_loss_float 0.001 \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --max_eval_samples 5000 \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl_critic_test \
     --ref_offload
EOF

deepspeed $training_commands 2>&1 | tee $save_path/train_qwen_test.log
