export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ENABLE=1
export NCCL_P2P_ENABLE=1

WORKSPACE=/path/to/InfoRM
cd $WORKSPACE
zero_stage=3
lr=5e-6
epoch=1
bs=8
gbs=64
samples=92223
pretrain=/path/to/Llama-3-8B
dataset=/path/to/sharegpt.jsonl
output_dir=${WORKSPACE}/output_sft
prefix=step1_llama3_8b_sharegpt_lr${lr}_bs${bs}_gbs${gbs}_epoch${epoch}_samples${samples}_chat_template_multi_turn_packing
bash ./example_sft/script/train_sft_llama3_chat_template_multi_turn_packing.sh $zero_stage $lr $bs $gbs $pretrain $dataset $epoch $samples $output_dir $prefix