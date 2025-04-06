
ZERO_STAGE=$1
LR=$2
BS=$3
GBS=$4
PRETRAIN=$5
DATASET=$6
EPOCH=$7
SAMPLES=$8
LATENT_DIM=$9
KL_LOSS_COEF=${10}
OUTPUT=${11}
PREFIX=${12}

mkdir -p $OUTPUT/$PREFIX

rm -rf ~/.cache/torch_extensions/

deepspeed --module openrlhf.cli.train_rm \
   --max_len 2048 \
   --dataset $DATASET \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --apply_chat_template \
   --tokenizer_chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \
   --train_batch_size $GBS \
   --micro_train_batch_size $BS \
   --max_samples $SAMPLES \
   --pretrain $PRETRAIN \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage $ZERO_STAGE \
   --max_epochs $EPOCH \
   --bf16 \
   --flash_attn \
   --packing_samples \
   --learning_rate $LR \
   --use_inform \
   --latent_dim ${LATENT_DIM} \
   --kl_loss_coef ${KL_LOSS_COEF} \
   --gradient_checkpointing \
   --use_tensorboard $OUTPUT/$PREFIX/tensorboard/ \
   --save_path $OUTPUT/$PREFIX/model/ 2>&1 | tee -a $OUTPUT/$PREFIX/$HOSTNAME-training.log