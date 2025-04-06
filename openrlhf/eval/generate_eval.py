import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
from openrlhf.utils.logging_utils import init_logger
import jsonlines, json
import os
logger = init_logger(__name__)

def save_jsonl(data, file):
    with jsonlines.open(file, 'w') as w:
        for item in data:
            w.write(item)
    print(f'Save to {file}')

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {file}.')

def read_jsonl(file):
    datalist=[]
    with open(file, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            datalist.append(item)
    return datalist



def generate_eval(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_len)

    prompt_data_list = args.prompt_data.split(",")
    for file in prompt_data_list:
        logger.info(f"Processing {file} !!!")
        filename = os.path.basename(file)
        assert filename.endswith('.jsonl') and filename.startswith('openrlhf_')
        filename = filename[len("openrlhf_"):-len(".jsonl")]
        output_path = os.path.join(args.output_path, "response_comparison", f"{args.tag}_response", f"{filename}_response.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Output path: {output_path}")

        all_prompt_chat = read_jsonl(file)[:args.num]
        all_prompt_str = [tokenizer.apply_chat_template(prompt['context_messages'], tokenize=False, add_generation_prompt=True) for prompt in all_prompt_chat]
        all_response = llm.generate(all_prompt_str, sampling_params)
        all_response = [response.outputs[0].text for response in all_response]

        data_save = []
        index = 0
        for prompt, response in zip(all_prompt_str, all_response):
            data_save.append({"index": index, "prompt": prompt, f"{args.tag}_response": response})
            index = index + 1
        
        save_jsonl(data_save, output_path)
        logger.info(f"Save to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()

    generate_eval(args)