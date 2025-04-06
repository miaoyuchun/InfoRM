# # Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import argparse
import os
from openrlhf.utils.logging_utils import init_logger
import jsonlines, json
logger = init_logger(__name__)

def filter_existing_directories(dir_list):
    # 使用列表推导式来过滤存在的目录
    existing_dirs = [dir for dir in dir_list if os.path.exists(dir) and os.path.isdir(dir) and os.listdir(dir)]
    return existing_dirs

def save_jsonl(data, file):
    with jsonlines.open(file, 'w') as w:
        for item in data:
            w.write(item)
    logger.info(f'Save to {file}')

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f'Saved to {file}.')

def read_jsonl(file):
    datalist=[]
    with open(file, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            datalist.append(item)
    return datalist

def main(root, output_path, file_name):
    file_path = [os.path.join(root, file) for file in file_name]
    file_path.sort(reverse=True)
    data_name = [f for f in os.listdir(file_path[0]) if f.endswith('.jsonl')]
    for data in data_name:
        logger.info(f"processing {data}")
        data_path = [os.path.join(p, data) for p in file_path]
        save_path = os.path.join(output_path, data)
        all_data = [list(read_jsonl(file)) for file in data_path]
        assert len(set([len(data) for data in all_data])) == 1
        merged_data_list = []

        for id in range(len(all_data[0])):
            prompts = [data[id]['prompt'] for data in all_data]
            assert len(set(prompts)) == 1
            merged_data = {k: v for data in all_data for k, v in data[id].items()}
            merged_data_list.append(merged_data)

        logger.info(f"{id+1} samples are combined")
        save_jsonl(merged_data_list, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="root path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    args.root = os.path.join(args.root, "response_comparison")
    output_path = os.path.join(args.root, "all_response")
    os.makedirs(output_path, exist_ok=True)

    file_name = [name for name in os.listdir(args.root) if name.endswith("_response") and os.path.isdir(os.path.join(args.root, name)) and os.listdir(os.path.join(args.root, name))]
    logger.info(file_name)

    main(root=args.root, output_path=output_path, file_name=file_name)
