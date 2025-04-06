import yaml
import sys
import json
import pickle
import time
import os
import gzip

from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self,testset):
        self.testset = testset
        
    def __getitem__(self,idx):
        if len(self.testset[idx]) == 2:
            example_id, token = self.testset[idx]
            item = {'idx':example_id}
            item.update(token)
        else:
            example_id, token, category = self.testset[idx]
            item = {'idx':example_id, 'category_id':category}
            item.update(token)
        return item
        
    def __len__(self):
        return len(self.testset)
        
class Config:
    def __init__(self):
        pass

# read config file & load the param into the 'config'
def read_config(args):
    # read config file
    with open(args.config_path,'r') as imf:
        yaml_config = yaml.safe_load(imf.read())
    
    config = Config()
    
    # load args param
    for k,v in sorted(vars(args).items()):
        setattr(config, k, v)
    
    # load config param
    for k in yaml_config.keys():
        setattr(config,k,yaml_config[k])
     
    return config

# write args into file
def write_args(args):
    
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(args.out, 'config.log')
    with open(file_name, 'wt') as config_file:
        config_file.write(message)
        config_file.write('\n')

# get local time format: year_month_day_hour_minute_second
def get_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))

# copy from mmlu
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

# copy from mmlu
def read_txt(file: str):
    with open(file, 'r') as f:
        return f.readlines()
        
# copy from mmlu
def save_txt(data: list, file: str):
    with open(file, 'w') as f:
        f.writelines(data)
    print(f'Save to {file}')

# copy from mmlu
def read_pickle(file: str) -> list:
    with open(file, 'rb') as f:
        return pickle.load(f)

# copy from mmlu
def save_pickle(data, file: str):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')

# copy from mmlu
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


# copy from mmlu
def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {file}.')
    return

# dedup
def dedup(l:list):
    out = {}
    for i in l:
        out[i[0]] = i[1]
    return list(out.items())    



# from baopu

def truncate_output(results):

    count=0
    count1=0
    count_human = 0
    count_assistant = 0
    count_sep=0
    for idx, answer in enumerate(results):
        if answer.find('USER:') > 0:
            end_pos = answer.find('USER:')
            results[idx] = results[idx][:end_pos]
            count+=1
        elif answer.find("Human:\n") > 0:
            end_pos = answer.find('Human:\n')
            results[idx] = results[idx][:end_pos]
            count_human+=1
        elif answer.find("Assistant:\n") > 0:
            end_pos = answer.find('Assistant:\n')
            results[idx] = results[idx][:end_pos]
            count_assistant+=1
        elif answer.find('<|endoftext|>') > 0:
            end_pos = answer.find('<|endoftext|>')
            results[idx] = results[idx][:end_pos]
            count1+=1
        elif answer.find('</s>') > 0:
            end_pos = answer.find('</s>')
            results[idx] = results[idx][:end_pos]
            count_sep+=1
        else:
            pass
            
    print(f"find {count} 'USER' {count_human} 'human' need to be truncated , {count1} end of text, {count_sep} need to be ended" )

    return results