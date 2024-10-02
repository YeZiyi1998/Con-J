export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EXP_NAME=trained_critic_llm_$1

MODEL="../../outputs/inference/zeroshot_llm_$1/dpo/model/dpo_critic_trained_$1/"
File_NAME="$2.jsonl" 
DATA="../../../data/sub_data/pairwise_critic_inference2_get_answer/$2/test"
save_path="../../outputs/inference/zeroshot_llm_$1/dpo/result/${EXP_NAME}"
# exist_prompt="$save_path/Qwen_Critic_machine.jsonl"
BATCH_SIZE_PER_GPU=4

MACHINE_SIZE=1
WORLD_SIZE=$[MACHINE_SIZE * 8]

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
    --input_template Human_reason \
    --tp_size 4 \
    --greedy_sampling \
    --output_path $save_path/$File_NAME \
    --evaluate \
    --llm_batch 5000
EOF

python $training_commands 2>&1 | tee $save_path/test_qwen2.log
# python /data2/rlhf/yzy/write_email.py --info Qwen_RM_critic_104_machine_done


