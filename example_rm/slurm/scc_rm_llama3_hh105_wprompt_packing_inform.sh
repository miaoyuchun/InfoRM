export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ENABLE=1
export NCCL_P2P_ENABLE=1

WORKSPACE=/path/to/InfoRM
cd $WORKSPACE
zero_stage=2
lr=9e-6
epoch=1
bs=8
gbs=64
samples=53684
latent_dim=512
kl_loss_coef=1.0
pretrain=${WORKSPACE}/output_sft/step1_llama3_8b_sharegpt_lr5e-6_bs8_gbs64_epoch1_samples92223_chat_template_multi_turn_packing/model
dataset=/path/to/anthropic_hh105.jsonl #helpful:harmelss = 2:1
output_dir=${WORKSPACE}/output_rm
prefix=step2_llama3_8b_sharegptsft_hh105_lr${lr}_bs${bs}_gbs${gbs}_epoch${epoch}_samples${samples}_inform_latent_${latent_dim}_beta_${kl_loss_coef}_chat_template_packing
bash ./example_rm/script/train_rm_llama3_chat_template_wprompt_packing_inform.sh $zero_stage $lr $bs $gbs $pretrain $dataset $epoch $samples $latent_dim $kl_loss_coef $output_dir $prefix