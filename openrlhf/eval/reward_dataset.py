import os
import numpy as np
import torch
import glob
from openrlhf.eval.utils import read_json, save_json, stream_jsonl, GeneratedDataset
from datasets import load_dataset, load_from_disk
from collections import Counter
import math
import time, random
from tqdm import tqdm, trange


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)


class RewardDataset():
    def __init__(self, path, engine, tokenizer, strategy):
        self.strategy = strategy
        self.engine = engine
        self.tokenizer = tokenizer
        self.path = path

    def eval(self):
        args = self.strategy.args
        overall_result = {'model_path':args.pretrain}
        short_overall_result = {'model_path':args.pretrain}
        overall_output_info = {'model_path':args.pretrain}

        #############################     Evaluating Alpaca Alighment           ###############################        
        # binary test sets
        overall_rw_values = []
        overall_label_list = []

        for dataset_type in ['gpt_rank_val', 'human_rank_val']:
            print_rank_0(f"begining to process alpaca_farm-{dataset_type}", args.local_rank)
            data_list,label_list,_ = self.get_data(os.path.join(self.path,'alpaca_farm'),type=dataset_type) # [(idx, (样本1， 样本2))]  [(idx, (1, 0))]
            print_rank_0("examples of alpaca_farm-{}:\n{}".format(dataset_type, data_list[0][1][0]), args.local_rank)
            dataset = self.create_dataset(data_list)  #{'idx': , 'input_ids': , 'attention mask': }
            print_rank_0(f"alpaca_farm-{dataset_type} has {len(data_list)} test examples", args.local_rank)
            reward_values = self.engine.run(dataset)['reward_values'] # [(idx, (length, reward, [mu], [logvar]))...]
            
            paired_rewards = self._convert2pair(reward_values) #[[idx, ((reward, length, [mu], [logvar]), (reward, length, [mu], [logvar]))]...]
            overall_rw_values.extend(paired_rewards) # [[0, (..., (...), (...))], [1, (..., (...), (...))], [2, (..., (...), (...))],    ]
            overall_label_list.extend(label_list) # [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
            accuracy = self._compute_binary_accuracy(paired_rewards,label_list) 
            
            dataset_name = f"alpaca_farm_{dataset_type}" 
            curr_infos = self._get_eval_info(paired_rewards,label_list, data_list, label_type='binary')
            overall_output_info[dataset_name] = curr_infos
                
            short_overall_result[dataset_name] = accuracy
            
        accuracy = self._compute_binary_accuracy(overall_rw_values,overall_label_list)

        overall_result['alpaca_farm_overall_accuracy'] = accuracy
        short_overall_result['alpaca_farm_overall_accuracy'] = accuracy
        
        print_rank_0("overall result of alpaca_farm is {}".format(short_overall_result), args.local_rank)

        #############################     Evaluating HHH Alighment           ###############################        
        # # binary test sets
        overall_rw_values = []
        overall_label_list = []

        for dataset_type in ['harmless', 'honest', 'helpful','other']:
            print_rank_0(f"begining to process hhh_alignment-{dataset_type}", args.local_rank)
            data_list,label_list,_ = self.get_data(os.path.join(self.path,'hhh_alignment'),type=dataset_type) # [(idx, (样本1， 样本2))]  [(idx, (1, 0))]
            print_rank_0("examples of hhh_alignment-{}:\n{}".format(dataset_type, data_list[0][1][0]), args.local_rank)
            dataset = self.create_dataset(data_list)  #{'idx': , 'input_ids': , 'attention mask': }
            print_rank_0(f"hhh_alignment-{dataset_type} has {len(data_list)} test examples", args.local_rank)
            reward_values = self.engine.run(dataset)['reward_values'] # [(idx, (length, reward, [mu], [logvar]))...]
            
            paired_rewards = self._convert2pair(reward_values) #[[idx, ((reward, length, [mu], [logvar]), (reward, length, [mu], [logvar]))]...]
            overall_rw_values.extend(paired_rewards) # [[0, (..., (...), (...))], [1, (..., (...), (...))], [2, (..., (...), (...))],    ]
            overall_label_list.extend(label_list) # [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
            accuracy = self._compute_binary_accuracy(paired_rewards,label_list) 
            
            dataset_name = f"hhh_{dataset_type}" 
            curr_infos = self._get_eval_info(paired_rewards,label_list, data_list, label_type='binary')
            overall_output_info[dataset_name] = curr_infos
                
            short_overall_result[dataset_name] = accuracy

        accuracy = self._compute_binary_accuracy(overall_rw_values,overall_label_list)

        overall_result['HHH_overall_accuracy'] = accuracy
        short_overall_result['HHH_overall_accuracy'] = accuracy
        
        print_rank_0("overall result of hhh_alignment is {}".format(short_overall_result), args.local_rank)
        

        #############################     Evaluating hh_rlhf           ###############################
        print_rank_0("begining to process hh_rlhf", args.local_rank)
        overall_rw_values = []
        overall_label_list = []

        for dataset_type in ['harmless-base', 'helpful-base']:
            print_rank_0(f"begining to process hh_rlhf-{dataset_type}", args.local_rank)
            data_list,label_list,_ = self.get_data(os.path.join(self.path,'hh_rlhf'),type=dataset_type) # [(idx, (样本1， 样本2))]  [(idx, (1, 0))]
            print_rank_0("examples of hh_rlhf-{}:\n{}".format(dataset_type, data_list[0][1][0]), args.local_rank)
            dataset = self.create_dataset(data_list)  #{'idx': , 'input_ids': , 'attention mask': }
            print_rank_0(f"hh_rlhf-{dataset_type} has {len(data_list)} test examples", args.local_rank)
            reward_values = self.engine.run(dataset)['reward_values'] # [(idx, (reward, [mu], [logvar]))...]
            
            paired_rewards = self._convert2pair(reward_values) #[[idx, ((reward, [mu], [logvar]), (reward, [mu], [logvar]))]...]
            overall_rw_values.extend(paired_rewards) # [[0, ((...), (...))], [0, ((...), (...))], [0, ((...), (...))],    ]
            overall_label_list.extend(label_list) # [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
            accuracy = self._compute_binary_accuracy(paired_rewards,label_list) 
            
            dataset_name = f"hh_rlhf_{dataset_type}"
            curr_infos = self._get_eval_info(paired_rewards,label_list, data_list, label_type='binary')
            overall_output_info[dataset_name] = curr_infos
                
            short_overall_result[dataset_name] = accuracy
        accuracy = self._compute_binary_accuracy(overall_rw_values,overall_label_list)

        overall_result['hh_rlhf_overall_accuracy'] = accuracy
        short_overall_result['hh_rlhf_overall_accuracy'] = accuracy
        print_rank_0("overall result of hh_rlhf is {}".format(short_overall_result), args.local_rank)


        #############################     Evaluating TruthfulQA           ###############################        
        for dataset_name in ['truthfulqa_mc']:
            print_rank_0("begining to process truthfulqa_mc", args.local_rank)
            data_list,label_list, count_candidates = \
                self.get_data(self.path, type='truthful_qa')
            print_rank_0("examples of truthfulqa_mc:\n{}".format(data_list[0][1][0]), args.local_rank)

            # print(count_candidates)
            dataset = self.create_dataset(data_list)
            print_rank_0(f"truthfulqa_mc has {len(data_list)} test examples", args.local_rank)
            reward_values = self.engine.run(dataset)['reward_values']

            paired_rewards = self._convert2pair(reward_values)
            accuracy = self._compute_mc_accuracy(paired_rewards,label_list,count_candidates) #_compute_factor_mc_accuracy
            short_overall_result[dataset_name] = accuracy
            
            curr_infos = self._get_eval_info(paired_rewards,label_list, data_list, label_type='binary')
            overall_output_info[f"{dataset_name}_accuracy"] = curr_infos     

        print_rank_0("overall result of truthfulqa_mc is {}".format(short_overall_result), args.local_rank)


        if args.local_rank == 0:
            save_json(data=short_overall_result,file=f"{args.out}/result_accuracy.json")
            # save_json(data=overall_output_info,file=f"{args.out}/detail_{args.pretrain.split('/')[-1]}.json")

    def create_dataset(self,data_list): # [(idx, (样本1， 样本2))]
        
        dataset = []
        for idx,data_pair in data_list:
            
            data_pair = list(data_pair)
            data_pair[0] = data_pair[0].rstrip("\n")
            if not data_pair[0].endswith(self.tokenizer.eos_token):
                data_pair[0] += " " + self.tokenizer.eos_token
            data_pair[1] = data_pair[1].rstrip("\n")
            if not data_pair[1].endswith(self.tokenizer.eos_token):
                data_pair[1] += " " + self.tokenizer.eos_token
            data_pair = tuple(data_pair)


            token = self.tokenizer(data_pair[0],return_tensors='pt')
            token['input_ids'] = token['input_ids'].squeeze() #torch.Size([75])
            token['attention_mask'] = token['attention_mask'].squeeze()#torch.Size([75])
            dataset.append((idx*2,token))

            token = self.tokenizer(data_pair[1],return_tensors='pt')
            token['input_ids'] = token['input_ids'].squeeze()
            token['attention_mask'] = token['attention_mask'].squeeze()
            dataset.append((idx*2+1,token)) # [(idx, {'input_ids': , 'attention_mask'})]
 
        return GeneratedDataset(dataset)

    
    def get_data(self,path,type=''):
        files = [file for file in os.listdir(os.path.join(path, type)) if file.startswith('openrlhf_') and file.endswith('.jsonl')]
        assert len(files) == 1, f"{os.path.join(path, type)} has {len(files)} files"
        all_data = [item for item in stream_jsonl(os.path.join(path, type, files[0]))]

        data_list, label_list, count_candidates = [], [], []
        global_idx=0
        cache = []
        for item in all_data:
            data_item = (self.tokenizer.apply_chat_template(item['prompt']+item['chosen'], tokenize=False), 
                         self.tokenizer.apply_chat_template(item['prompt']+item['rejected'], tokenize=False))   #(样本1， 样本2)        
            label_item = (1, 0)
            data_list.append((global_idx,data_item)) # [(idx, (样本1， 样本2))]
            label_list.append((global_idx, label_item)) # [(idx, (1, 0))]
            global_idx+=1

            cache.append(self.tokenizer.apply_chat_template(item['prompt'], tokenize=False))
            if len(set(cache)) != 1:
                 assert len(set(cache[:-1])) == 1
                 count_candidates.append(len(cache)-1)
                 cache = [cache[-1]]

        return data_list, label_list, count_candidates # [(idx, (样本1， 样本2))]  [(idx, (1, 0))]



    # def get_summarize_preference_data(self,path):
    #     summarize_valid = load_from_disk(os.path.join(path,'summarize_from_feedback'))['validation']
    #     print(f"Summarize dataset total: {len(summarize_valid)}")
    #     summarize_valid = summarize_valid[:200]
    #     input_prefix = "Human:\nPlease summarize the article below.\n"
    #     output_prefix = "Assistant:\nSummary:\n"
    #     num_samples = len(summarize_valid['info'])
    #     data_list = []
    #     label_list = []
    #     for idx in range(num_samples):
    #         article_info= summarize_valid['info'][idx]['post']
    #         candidate_summaries = summarize_valid['summaries'][idx]
    #         data_item = (input_prefix + article_info + output_prefix + candidate_summaries[0]['text'] + '<|endoftext|>', \
    #                      input_prefix + article_info + output_prefix + candidate_summaries[1]['text'] + '<|endoftext|>')
    #         label_item = (1, 0) if int(summarize_valid['choice'][idx]) == 0 else (0, 1)
    #         data_list.append((idx, data_item))
    #         label_list.append((idx, label_item))

    #     return data_list, label_list

    def _convert2pair(self, reward_values: list): # [(idx, (reward, length, [mu], [logvar]))...]
        assert(len(reward_values) % 2 == 0)
        paired_rewards = []
        for idx in range(0,len(reward_values),2):
            paired_rewards.append(
                [idx // 2, \
                (reward_values[idx][-1],reward_values[idx+1][-1])
                ]
            )
        
        return paired_rewards #[[idx, ((reward, [mu], [logvar]), (reward, [mu], [logvar]))] ... ]
   
    def _compute_binary_accuracy(self,paired_rewards, label_list):  #[[idx, ((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))]...]  [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
        # print(f"{len(label_list)}")
        catched_items = 0
        total_len = len(label_list)
        for pred, truth in zip(paired_rewards,label_list):# pred: [idx, ((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))]    truth:  (idx, (1, 0))
            assert(pred[0] == truth[0]) # ensure that idx matches
            pair_pred = pred[1] #((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))
            pair_truth = truth[1] # (1, 0)
            if (pair_pred[0][1] > pair_pred[1][1]) ^ (pair_truth[0] > pair_truth[1]):
                pass
            else:
                catched_items += 1
                
        return catched_items / total_len
    

    def _compute_mc_accuracy(self, paired_rewards, label_list,candidate_count):#[[idx, ((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))]...]  [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
        # print(f"{len(label_list)}")
        catched_items = 0
        total_len = len(candidate_count)
        global_idx = 0
        for answer_count in candidate_count:
            correct = True
            for temp_id in range(answer_count - 1):
                pred_idx, pair_pred = paired_rewards[global_idx + temp_id]
                truth_idx, pair_truth = label_list[global_idx + temp_id]
                assert(pred_idx == truth_idx)
                if (pair_pred[0][1] > pair_pred[1][1]) ^ (pair_truth[0] > pair_truth[1]): # pred与truth的大小关系不同
                    correct = False
                    break

            if correct:
                catched_items += 1
            global_idx += (answer_count - 1)

        return catched_items / total_len
    
    
    @staticmethod
    def _get_pos(a,b):
        max_pos = min(len(a),len(b))
        for idx in range(max_pos):
            if a[idx] != b[idx]:
                break
        return idx
    
    
    def _get_eval_info(self, paired_rewards, label_list, data_list, label_type): ##paired_rewards: [[idx, ((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))],  ... ] label_list: [(idx, (1,0)), ...] data_list: [(idx,(正样本，负样本)),(idx,(正样本，负样本)),(idx,(正样本，负样本))]

        outputs = []
        num_of_samples = len(paired_rewards)
        for idx in range(num_of_samples):
            answer_start_pos = self._get_pos(data_list[idx][-1][0],data_list[idx][-1][1])
            question = data_list[idx][-1][0][:answer_start_pos]
            response_a = data_list[idx][-1][0][answer_start_pos:]
            response_b = data_list[idx][-1][1][answer_start_pos:]
            ground_truth = label_list[idx][-1]
            reward_value = paired_rewards[idx][-1] # ((length, reward, [mu], [logvar]), (length, reward, [mu], [logvar]))
            if reward_value[0][2] == None:
                info_item = {'correct': not ((reward_value[0][1] > reward_value[1][1]) ^ (ground_truth[0] > ground_truth[1])),\
                    'question':question, \
                    'response': [response_a, response_b], \
                    'grouth_truth': ground_truth,\
                    'reward_value': [reward_value[0][1], reward_value[1][1]]
                }
            else:
                info_item = {'correct': not ((reward_value[0][1] > reward_value[1][1]) ^ (ground_truth[0] > ground_truth[1])),\
                    'question':question, \
                    'response': [response_a, response_b], \
                    'grouth_truth': ground_truth,\
                    'reward_value': [reward_value[0][1], reward_value[1][1]],\
                    'positive_mean_std': [[np.mean(ipm) for ipm in reward_value[0][2]], [np.mean(ips) for ips in reward_value[0][3]]],\
                    'reject_mean_std': [[np.mean(irm) for irm in reward_value[1][2]], [np.mean(irs)  for irs in reward_value[1][3]]]
                }
            outputs.append(info_item)

        return outputs
    

    def _get_eval_info_template(self, paired_rewards, label_list, data_list, label_type):

        outputs = []
        num_of_samples = len(paired_rewards)
        for idx in range(num_of_samples):
            answer_start_pos1 = data_list[idx][-1][0].find("Assistant:\n")
            answer_start_pos2 = data_list[idx][-1][1].find("Assistant:\n")
            question_a = data_list[idx][-1][0][:answer_start_pos1]
            question_b = data_list[idx][-1][0][:answer_start_pos2]
            response_a = data_list[idx][-1][0][answer_start_pos1:]
            response_b = data_list[idx][-1][1][answer_start_pos2:]
            info_item = {'question':[question_a, question_b], \
                'response': [response_a, response_b], \
                'grouth_truth': label_list[idx][-1],\
                'reward_value': paired_rewards[idx][-1]
            }
            outputs.append(info_item)

        return outputs