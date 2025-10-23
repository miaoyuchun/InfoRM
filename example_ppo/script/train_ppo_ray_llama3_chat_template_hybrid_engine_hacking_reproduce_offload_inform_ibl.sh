NODES=$1
GPUS=$2
TP=$3
VLLM_MEMORY=$4
ZERO_STAGE=$5
ACTOR_LR=$6
CRITIC_LR=$7
BS=$8
GBS=$9
RBS=${10}
GRBS=${11}
PRETRAIN=${12}
REWARD_PRETRAIN=${13}
KL=${14}
DATASET=${15}
INPUT_KEY=${16}
CLASS_KEY=${17}
EPOCH=${18}
SAMPLES=${19}
PROMPT_LEN=${20}
GENERATE_LEN=${21}
FREEZE=${22}
IBL_COEF=${23}
OUTPUT=${24}
PREFIX=${25}

mkdir -p $OUTPUT/$PREFIX

rm -rf ~/.cache/torch_extensions/
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
ray stop --force
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${GPUS}

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes ${NODES} \
   --ref_num_gpus_per_node ${GPUS} \
   --reward_num_nodes ${NODES} \
   --reward_num_gpus_per_node ${GPUS} \
   --critic_num_nodes ${NODES} \
   --critic_num_gpus_per_node ${GPUS} \
   --actor_num_nodes ${NODES} \
   --actor_num_gpus_per_node ${GPUS} \
   --vllm_num_engines $(expr $GPUS / $TP) \
   --vllm_tensor_parallel_size ${TP} \
   --colocate_all_models \
   --ref_reward_offload \
   --vllm_gpu_memory_utilization ${VLLM_MEMORY} \
   --pretrain  ${PRETRAIN}\
   --reward_pretrain  ${REWARD_PRETRAIN}\
   --critic_pretrain ${PRETRAIN}\
   --micro_train_batch_size ${BS} \
   --train_batch_size ${GBS} \
   --micro_rollout_batch_size ${RBS} \
   --rollout_batch_size ${GRBS} \
   --n_samples_per_prompt 1 \
   --freezing_actor_steps ${FREEZE} \
   --max_epochs ${EPOCH} \
   --num_episodes 1 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --temperature 0.0 \
   --prompt_max_len ${PROMPT_LEN} \
   --max_samples ${SAMPLES} \
   --generate_max_len ${GENERATE_LEN} \
   --zero_stage ${ZERO_STAGE} \
   --bf16 \
   --actor_learning_rate ${ACTOR_LR} \
   --critic_learning_rate ${CRITIC_LR} \
   --init_kl_coef ${KL} \
   --prompt_data ${DATASET} \
   --input_key ${INPUT_KEY} \
   --class ${CLASS_KEY} \
   --use_inform \
   --ibl_coef ${IBL_COEF} \
   --apply_chat_template \
   --tokenizer_chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --adam_offload \
   --vllm_sync_backend nccl \
   --use_tensorboard $OUTPUT/$PREFIX/tensorboard/ \
   --save_path $OUTPUT/$PREFIX/model/ 2>&1 | tee -a $OUTPUT/$PREFIX/training.log

ray stop --force