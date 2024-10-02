export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXP_NAME=zeroshot_llm_$2

MODEL="../../../open_models/Qwen2-7B-Instruct/"
File_NAME="best_of_$1.jsonl" 
DATA="../../../data/sub_data/pairwise_critic_inference2_get_answer/$2"
save_path="../../outputs/inference/${EXP_NAME}/"
BATCH_SIZE_PER_GPU=3
LR=1e-5

MACHINE_SIZE=1
WORLD_SIZE=$[MACHINE_SIZE * 8]

n=$1
mkdir -p $save_path

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
    --temperature 1.2 \
    --num_beams 1 \
    --top_p 0.95 \
    --top_k 50 \
    --best_of_n $1 \
    --seed 1688 \
    --output_path $save_path/$File_NAME
EOF

export CUDA_VISIBLE_DEVICES=0,1,2,3; python $training_commands > $save_path/best_of_$1.3.log 2>&1& 
pid1=$!
wait $pid1
