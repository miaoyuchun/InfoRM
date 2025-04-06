DATASET=${1}
ROOT=${2}
INI_MODEL_PATH=${3}
NUM=${4}
TP=${5}

CUDA_VISIBLE_DEVICES=0 python -m openrlhf.eval.generate_eval \
    --prompt_data ${DATASET} \
    --model_path ${ROOT}/model \
    --output_path ${ROOT} \
    --max_len 1024 \
    --num ${NUM} \
    --tensor_parallel_size ${TP} \
    --tag latest 2>&1 | tee -a $ROOT/generate_eval.log

CUDA_VISIBLE_DEVICES=0 python -m openrlhf.eval.generate_eval \
    --prompt_data ${DATASET} \
    --model_path ${INI_MODEL_PATH} \
    --output_path ${ROOT} \
    --max_len 1024 \
    --num ${NUM} \
    --tensor_parallel_size ${TP} \
    --tag num0 2>&1 | tee -a $ROOT/generate_eval.log

python -m openrlhf.eval.combine_responses \
    --root ${ROOT} 2>&1 | tee -a $ROOT/generate_eval.log
