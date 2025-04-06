import os
import time
import numpy as np
import torch
from transformers import set_seed, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class RewardEngine():
    def __init__(self, strategy, rm_model, tokenizer):
        self.strategy = strategy
        self.model = rm_model
        self.tokenizer = tokenizer
        self.cfg = strategy.args
        
    def run(self,dataset,**kwargs):
        '''
        params: dataloader, cfg
        returns:
            {
                "probs":[(idx1,[prob1,prob2,prob3,prob4]),(idx2,[prob1,prob2,prob3,prob4])]
                "preds":[(idx1,A),(idx2,B)]
            }
        '''
        dataloader = self.strategy.setup_dataloader(dataset, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=False, drop_last=False, collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer))
        
        reward_values = []
        world_size = dist.get_world_size()
        device = torch.cuda.current_device()
        '''
        batch must be has 'idx','input_ids', 'attention_mask'
        '''
        for step,batch in enumerate(dataloader):

            with torch.no_grad():
                batch_input = {'input_ids': batch['input_ids'], \
                            'attention_mask': batch['attention_mask'],\
                            'return_output': True,
                        }
                # print(batch['idx'])
                batch_idx = batch['idx']
                batch_input = to_device(batch_input, device)
                reward, outputs = self.model(**batch_input)
                batch_len = batch['attention_mask'].sum(dim=1)

                outputs['ib_representation'] = [None]*reward.shape[0] if 'ib_representation' not in outputs else outputs['ib_representation']
                outputs['ib_representation_logvar'] = [None]*reward.shape[0] if 'ib_representation_logvar' not in outputs else outputs['ib_representation_logvar']

                # zip into tuple with id
                reward_value = list(zip(batch_idx, batch_len, reward.tolist(), outputs['ib_representation'], outputs['ib_representation_logvar']))
                reward_value = [(sample[0], sample[1:])  for sample in reward_value]

                rw_distributed = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(rw_distributed, reward_value)
                rw_distributed = [item for _rw in rw_distributed for item in _rw]
                reward_values.extend(rw_distributed)
                
        # dedup
        def dedup(l:list):
            out = {}
            for i in l:
                out[i[0].item()] = i[1]
            return list(out.items())
        
        reward_values = dedup(reward_values)
        
        # sort output values according to their indexes
        sorted_reward_values = sorted(reward_values, key=lambda x: x[0])
        return {'reward_values': sorted_reward_values}

