WORKSPACE=/path/to/InfoRM
cd $WORKSPACE

prompt_harmless=/path/to/anthropic_harmless.jsonl
prompt_helpful=/path/to/anthropic_helpful.jsonl

model_path=./output_sft/step1_llama3_8b_sharegpt_lr5e-6_bs8_gbs64_epoch1_samples92223_chat_template_multi_turn_packing/model
reward_model_path=./output_rm/step2_llama3_8b_sharegptsft_hh105_lr9e-6_bs8_gbs64_epoch1_samples53684_inform_latent_512_beta_1.0_chat_template_packing/model

python -m openrlhf.eval.prepare_ib_representation \
    --prompt_data ${prompt_harmless} \
    --model_path ${model_path} \
    --reward_model_path ${reward_model_path}


python -m openrlhf.eval.prepare_ib_representation \
    --prompt_data ${prompt_helpful} \
    --model_path ${model_path} \
    --reward_model_path ${reward_model_path}