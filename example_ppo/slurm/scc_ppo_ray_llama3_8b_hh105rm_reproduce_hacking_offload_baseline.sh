export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ENABLE=1
export NCCL_P2P_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

WORKSPACE=/path/to/InfoRM
cd $WORKSPACE
nodes=1
gpus=8
tp=2
vllm_memory=0.5
zero_stage=3
actor_lr=5e-7
critic_lr=5e-6
bs=16
gbs=128
rbs=16
grbs=128
pretrain=./output_sft/step1_llama3_8b_sharegpt_lr5e-6_bs8_gbs64_epoch1_samples92223_chat_template_multi_turn_packing/model
reward_pretrain=./output_rm/step2_llama3_8b_sharegptsft_hh105_lr9e-6_bs8_gbs64_epoch1_samples53684_chat_template_packing/model
kl=0
dataset=/path/to/anthropic_hh11.jsonl #helpful:harmelss = 1:1
input_key=context_messages
epoch=1
samples=72288
prompt_length=512
generate_length=512
freezing_actor_steps=-1
output=${WORKSPACE}/output_ppo_ray
prefix=step3_llama3_8b_sharegptsft_packing_hh105rm_packing_hh11_actorlr${actor_lr}_criticlr${critic_lr}_bs${bs}_gbs${gbs}_rbs${rbs}_grbs${grbs}_kl${kl}_epoch${epoch}_samples${samples}_plen${prompt_length}_glen${generate_length}_freezing_actor${freezing_actor_steps}_chat_template_hacking_reproduce_tem0_offload_reproduce
bash ./example_ppo/script/train_ppo_ray_llama3_chat_template_hybrid_engine_hacking_reproduce_offload.sh $nodes $gpus $tp $vllm_memory $zero_stage $actor_lr $critic_lr $bs $gbs $rbs $grbs $pretrain $reward_pretrain $kl $dataset $input_key $epoch $samples $prompt_length $generate_length $freezing_actor_steps $output $prefix


dataset=/path/to/openrlhf_alpaca_farm.jsonl
root=${output}/${prefix}
samples=2000
tp=1
bash ./example_ppo/script/generate_eval.sh ${dataset} ${root} ${pretrain} ${samples} ${tp}

root=${output}/${prefix}
inform=./output_rm/step2_llama3_8b_sharegptsft_hh105_lr9e-6_bs8_gbs64_epoch1_samples53684_inform_latent_512_beta_1.0_chat_template_packing/model
tag=analysis_512_1.0_sft
bash ./example_ppo/script/ib_latent_eval.sh ${root} ${inform} ${tag}