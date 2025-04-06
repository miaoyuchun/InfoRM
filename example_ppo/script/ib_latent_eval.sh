ROOT=${1}
INFORM=${2}
TAG=${3}


mkdir -p ${ROOT}/${TAG}
deepspeed --module openrlhf.eval.eval_rlhf_ib_latent \
    --pretrain ${INFORM} \
    --out ${ROOT}/${TAG} \
    --batch_size 64 \
    --max_seq_len 2048 \
    --zero_stage 2 \
    --target_key latest_response \
    --dataset_path ${ROOT}/response_comparison/all_response \
    --flash_attn \
    --use_inform 2>&1 | tee -a $ROOT/inform_eval.log



