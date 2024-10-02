export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXP_NAME=zeroshot_llm_$1

MODEL="../../../open_models/Qwen2-7B-Instruct/"
DATA="../../../data/sub_data/pairwise_critic_inference2_get_answer/$1"
save_path="../../outputs/inference/${EXP_NAME}/"
BATCH_SIZE_PER_GPU=3
LR=1e-5

MACHINE_SIZE=1
WORLD_SIZE=$[MACHINE_SIZE * 8]

File_NAME=".jsonl" 
read -r -d '' training_commands <<EOF
../../examples/batch_inference.py \
    --eval_task generate_vllm \
    --pretrain $MODEL \
    --bf16 \
    --max_len 6400 \
    --dataset $DATA \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --tp_size 4 \
    --micro_batch_size $BATCH_SIZE_PER_GPU \
    --flash_attn \
    --max_new_tokens 512 \
    --input_template Human \
    --tp_size 4 \
    --output_path $save_path/guide_reverse$File_NAME \
    --template_key guide_reverse \
    --llm_batch 5000
EOF

python $training_commands |tee $save_path/train_qwen1.log 

read -r -d '' training_commands <<EOF
../../examples/batch_inference.py \
    --eval_task generate_vllm \
    --pretrain $MODEL \
    --bf16 \
    --max_len 6400 \
    --dataset $DATA \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --tp_size 4 \
    --micro_batch_size $BATCH_SIZE_PER_GPU \
    --flash_attn \
    --max_new_tokens 512 \
    --input_template Human \
    --tp_size 4 \
    --output_path $save_path/guide$File_NAME \
    --template_key guide \
    --llm_batch 5000
EOF

python $training_commands |tee $save_path/train_qwen2.log

