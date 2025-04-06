import argparse, os, torch, sys
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression, get_llm_for_sequence_regression_inform
from openrlhf.eval.rlhf_dataset_tsne import RLHF_Dataset_TSNE
from openrlhf.eval.reward_engine import RewardEngine
import deepspeed
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pretrain", type=str, default=None, help="Directory containing trained actor model")
    parser.add_argument('--out', type=str, default=None, help="specify the name of the output folder")
    parser.add_argument('--batch_size', type=int, default=64, help="specify the batch size for evaluation")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="The maximum sequence length.")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument('--dataset_path', type=str, default=None, help='The path of datasets')
    parser.add_argument('--target_key', type=str, default='latest_response')
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--use_inform", action="store_true", default=False)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    strategy = get_strategy(args)
    strategy.setup_distributed()
    ds_config = strategy.get_ds_eval_config()

    if args.use_inform:
        strategy.print("InfoRM is activated !!!")
        rm_model = get_llm_for_sequence_regression_inform(
            args.pretrain,
            "reward",
            use_flash_attention_2=args.flash_attn,
            ds_config=ds_config
        )
    else:
        strategy.print("Standard RM is activated !!!")
        rm_model = get_llm_for_sequence_regression(
            args.pretrain,
            "reward",
            use_flash_attention_2=args.flash_attn,
            ds_config=ds_config
        )

    tokenizer = get_tokenizer(args.pretrain, rm_model, "left", strategy, use_fast=True)

    strategy.print(rm_model)
    rm_model.eval()
    ds_engine = deepspeed.initialize(model=rm_model, config_params=ds_config)[0]
    ds_engine.module.eval()  # inference
    rm_model = ds_engine.module
    
    engine = RewardEngine(strategy, rm_model, tokenizer)
    data_process = RLHF_Dataset_TSNE(args.dataset_path, engine, tokenizer, strategy)
    data_process.eval()



    

if __name__ == "__main__":
    main()