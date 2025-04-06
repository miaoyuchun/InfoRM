import os
import numpy as np
import torch
import glob
from openrlhf.eval.utils import read_json, save_json, stream_jsonl, GeneratedDataset

from datasets import load_dataset, load_from_disk
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import time, random
from tqdm import tqdm, trange
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)

class RLHF_Dataset_TSNE():
    def __init__(self, path, engine, tokenizer, strategy):
        self.strategy = strategy
        self.engine = engine
        self.tokenizer = tokenizer
        self.path = path

    def eval(self):
        args = self.strategy.args
        overall_result = {'model_path':args.pretrain, 'dataset_path': args.dataset_path}
        short_overall_result = {'model_path':args.pretrain, 'dataset_path': args.dataset_path}
        overall_output_info = {'model_path':args.pretrain, 'dataset_path': args.dataset_path}

        overall_rw_values = []
        overall_label_list = []
        
        for file_path in glob.glob(os.path.join(args.dataset_path, '*.jsonl')):
            file_name = os.path.basename(file_path)
            print_rank_0(f"begining to process {file_name}", args.local_rank)
            all_data = [item for item in stream_jsonl(file_path)]
                
            data_list, label_list = [], []
            global_idx=0

            for item in all_data:
                try:
                    data_item = (item['prompt'] + item[args.target_key].rstrip() + ' ' + self.tokenizer.eos_token, 
                                item['prompt'] + item['num0_response'].rstrip() + ' ' + self.tokenizer.eos_token)
                except:
                    data_item = (item['input'] + item[args.target_key].rstrip() + ' ' + self.tokenizer.eos_token, 
                                item['input'] + item['actor_num0_response'].rstrip() + ' ' + self.tokenizer.eos_token)
                label_item = (1, 0)
                data_list.append((global_idx,data_item)) # [(idx, (样本1， 样本2))]
                label_list.append((global_idx, label_item)) # [(idx, (1, 0))]
                global_idx+=1
                
            print_rank_0("examples of {}:\n{}".format(file_name, data_list[0][1][0]), args.local_rank)
            dataset = self.create_dataset(data_list)  #{'idx': , 'input_ids': , 'attention mask': }
            reward_values = self.engine.run(dataset)['reward_values'] # [(idx, (length, reward, [mu], [std]))...]
            paired_rewards = self._convert2pair(reward_values) #[[idx, ((reward, [mu], [std]), (reward, [mu], [std]))]...]
            overall_rw_values.extend(paired_rewards) # [[0, ((...), (...))], [0, ((...), (...))], [0, ((...), (...))],    ]
            overall_label_list.extend(label_list) # [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
            accuracy = self._compute_binary_accuracy(paired_rewards,label_list)  #[[idx, ((reward, length, [mu], [std]), (reward, length, [mu], [std]))]...]
            curr_infos = self._get_eval_info(paired_rewards,label_list, data_list, label_type='binary')
            overall_output_info[file_name] = curr_infos
            
            overall_result[file_name] = accuracy
                
            print_rank_0("Begin to draw T-SNE", args.local_rank)
            self._draw_dynamic_tsne(paired_rewards, save_path=f"{args.out}/dynamic_tsne_"+file_name+".html", num=500)
            self._draw_tsne(paired_rewards, save_path=f"{args.out}/tsne_"+file_name+".png", num=500)
                
            short_overall_result[file_name] = accuracy
            
        accuracy = self._compute_binary_accuracy(overall_rw_values,overall_label_list)
        overall_result['RLHF_overall_accuracy'] = accuracy
        short_overall_result['RLHF_overall_accuracy'] = accuracy
        
        print_rank_0("overall result of RLHF is {}".format(short_overall_result), args.local_rank)
        
        if args.local_rank == 0:
            save_json(data=overall_result,file=f"{args.out}/result_rlhf.json")
            save_json(data=overall_output_info,file=f"{args.out}/detail_result_rlhf.json")
            
            
                
            
                


    def create_dataset(self,data_list): # [(idx, (样本1， 样本2))]
        
        dataset = []
        for idx,data_pair in data_list:
            token = self.tokenizer(data_pair[0],return_tensors='pt')
            token['input_ids'] = token['input_ids'].squeeze() #torch.Size([75])
            token['attention_mask'] = token['attention_mask'].squeeze()#torch.Size([75])
            dataset.append((idx*2,token))

            token = self.tokenizer(data_pair[1],return_tensors='pt')
            token['input_ids'] = token['input_ids'].squeeze()
            token['attention_mask'] = token['attention_mask'].squeeze()
            dataset.append((idx*2+1,token)) # [(idx, {'input_ids': , 'attention_mask'})]

        return GeneratedDataset(dataset)


    def _convert2pair(self, reward_values: list): # [(idx, (reward, length, [mu], [std]))...]
        assert(len(reward_values) % 2 == 0)
        paired_rewards = []
        for idx in range(0,len(reward_values),2):
            paired_rewards.append(
                [idx // 2, \
                (reward_values[idx][-1],reward_values[idx+1][-1])
                ]
            )
        
        return paired_rewards #[[idx, ((reward, [mu], [std]), (reward, [mu], [std]))] ... ]
   
    def _compute_binary_accuracy(self,paired_rewards, label_list):  #[[idx, ((length, reward, [mu], [std]), (length, reward, [mu], [std]))]...]  [(idx, (1, 0)), (idx, (1, 0)), (idx, (1, 0))]
        # print(f"{len(label_list)}")
        catched_items = 0
        total_len = len(label_list)
        for pred, truth in zip(paired_rewards,label_list):# pred: [idx, ((length, reward, [mu], [std]), (length, reward, [mu], [std]))]    truth:  (idx, (1, 0))
            assert(pred[0] == truth[0]) # ensure that idx matches
            pair_pred = pred[1] #((length, reward, [mu], [std]), (length, reward, [mu], [std]))
            pair_truth = truth[1] # (1, 0)
            if (pair_pred[0][1] > pair_pred[1][1]) ^ (pair_truth[0] > pair_truth[1]):
                pass
            else:
                catched_items += 1
                
        return catched_items / total_len

    
    
    @staticmethod
    def _get_pos(a,b):
        max_pos = min(len(a),len(b))
        for idx in range(max_pos):
            if a[idx] != b[idx]:
                break
        return idx
    
    
    def _get_eval_info(self, paired_rewards, label_list, data_list, label_type): ##paired_rewards: [[idx, ((length, reward, [mu], [std]), (length, reward, [mu], [std]))],  ... ] label_list: [(idx, (1,0)), ...] data_list: [(idx,(正样本，负样本)),(idx,(正样本，负样本)),(idx,(正样本，负样本))]

        outputs = []
        num_of_samples = len(paired_rewards)
        for idx in range(num_of_samples):
            answer_start_pos = self._get_pos(data_list[idx][-1][0],data_list[idx][-1][1])
            question = data_list[idx][-1][0][:answer_start_pos]
            response_a = data_list[idx][-1][0][answer_start_pos:]
            response_b = data_list[idx][-1][1][answer_start_pos:]
            ground_truth = label_list[idx][-1]
            reward_value = paired_rewards[idx][-1] # ((length, reward, [mu], [std]), (length, reward, [mu], [std]))
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



    def _draw_tsne(self, paired_rewards, save_path, num):#[[idx, ((length, reward, [mu], [std]), (length, reward, [mu], [std]))]...] tsne_"+dataset_name+".png"
        random.seed(123)
        paired_rewards = random.sample(paired_rewards, min(len(paired_rewards), num))
        dataset_name = os.path.basename(save_path).split('.png')[0].split('tsne_')[-1]
        num_sample = len(paired_rewards)
        factor_num = 1; factor_width = 1
        dim_factor =len((paired_rewards[0][-1][0][-1]))
        label = np.tile(np.arange(factor_num*factor_width), num_sample)
        label_all = np.tile(np.arange(factor_num*factor_width*2), num_sample)
    
        
        all_mu_a_list = [np.array(paired_rewards[i][-1][0][2]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))] # [[1,2,3], [1,2,3],...] # self.factor_num, self.dim_factor
        all_mu_b_list = [np.array(paired_rewards[i][-1][1][2]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))]
        all_mu_a = np.concatenate(all_mu_a_list, axis=0) # self.factor_num*num_sample  self.dim_factor
        all_mu_b = np.concatenate(all_mu_b_list, axis=0) # self.factor_num*num_sample  self.dim_factor
        all_mu = np.hstack((all_mu_a, all_mu_b)).reshape(-1, dim_factor) # 2*self.factor_num*num_sample  self.dim_factor
        
        all_std_a_list = [np.array(paired_rewards[i][-1][0][-1]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))] # [[1,2,3], [1,2,3],...]
        all_std_b_list = [np.array(paired_rewards[i][-1][1][-1]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))] # [[1,2,3], [1,2,3],...]
        all_std_a = np.concatenate(all_std_a_list, axis=0) # self.factor_num*num_sample  self.dim_factor
        all_std_b = np.concatenate(all_std_b_list, axis=0) # self.factor_num*num_sample  self.dim_factor
        all_std = np.hstack((all_std_a, all_std_b)).reshape(-1, dim_factor) # 2*self.factor_num*num_sample  self.dim_factor

        assert label.shape[0] == all_mu_a.shape[0] == all_mu_b.shape[0] == all_std_a.shape[0] == all_std_b.shape[0] == num_sample*factor_num*factor_width, "big error!!!"
        
        # draw overall tsne
        print("Begining to initialize TSNE")
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        start = time.time()
        result_all_mu_a = tsne.fit_transform(all_mu_a); result_all_mu_a = (result_all_mu_a - np.min(result_all_mu_a, 0)) / (np.max(result_all_mu_a, 0) - np.min(result_all_mu_a, 0))
        result_all_mu_b = tsne.fit_transform(all_mu_a); result_all_mu_b = (result_all_mu_b - np.min(result_all_mu_b, 0)) / (np.max(result_all_mu_b, 0) - np.min(result_all_mu_b, 0))
        result_all_std_a = tsne.fit_transform(all_std_a); result_all_std_a = (result_all_std_a - np.min(result_all_std_a, 0)) / (np.max(result_all_std_a, 0) - np.min(result_all_std_a, 0))
        result_all_std_b = tsne.fit_transform(all_std_b); result_all_std_b = (result_all_std_b - np.min(result_all_std_b, 0)) / (np.max(result_all_std_b, 0) - np.min(result_all_std_b, 0))
        result_all_mu = tsne.fit_transform(all_mu); result_all_mu = (result_all_mu - np.min(result_all_mu, 0)) / (np.max(result_all_mu, 0) - np.min(result_all_mu, 0))
        result_all_std = tsne.fit_transform(all_std); result_all_std = (result_all_std - np.min(result_all_std, 0)) / (np.max(result_all_std, 0) - np.min(result_all_std, 0))
        consume = time.time() - start
        print("finish TSNE initialization with {}".format(consume))

        fig, axes = plt.subplots(1,2, figsize=(12, 6))
        ax = axes[0]
        for i in tqdm(range(result_all_mu.shape[0]), desc="Processing Mean of All {} {} Sample".format(dataset_name, result_all_mu.shape[0])):
            ax.scatter(result_all_mu[i, 0], result_all_mu[i, 1], marker='o', color=plt.cm.Set1(label_all[i] / float(factor_num*2)), s=0.1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title('Mean of All Sample', fontsize=10)
            
        ax = axes[1]
        for i in tqdm(range(result_all_std.shape[0]), desc="Processing Std of All {} {} Sample".format(dataset_name, result_all_mu.shape[0])):
            ax.scatter(result_all_std[i, 0], result_all_std[i, 1], marker='o', color=plt.cm.Set1(label_all[i] / float(factor_num*2)), s=0.1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title('Std of All Sample', fontsize=10)

        print(f"begin to save")
        start = time.time()
        fig.suptitle("TSNE of {}".format(dataset_name), fontsize=16)
        fig.savefig(save_path, dpi=500, bbox_inches='tight')
        consume = time.time() - start
        print(f"finish saving at {save_path} costs {consume}")

        # draw mean factor-level tsne
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        fig, axes = plt.subplots(int(factor_num), int(factor_width))# , figsize=(12, 8)
        fig_pure, axes_pure = plt.subplots(int(factor_num), int(factor_width))# , figsize=(12, 8)
        custom_colors, custom_cmap = ['red', 'blue'], ['Reds', 'Blues']
        all_a_length = [paired_rewards[i][-1][0][0].item() for i in range(len(paired_rewards))]   # num_sample
        all_b_length = [paired_rewards[i][-1][1][0].item() for i in range(len(paired_rewards))]    # num_sample
        # all_length = np.column_stack((all_a_length, all_b_length)).reshape(-1) # 2*num_sample
        # normalized_all_length = [(x - min(all_length)) / (max(all_length) - min(all_length)) for x in all_length] # 2*num_sample
        all_length = np.log10(np.concatenate((all_a_length, all_b_length), axis=0))# 2*num_sample 
        normalized_all_length = np.array([(x - min(all_length)) / (max(all_length) - min(all_length)) for x in all_length]) # 2*num_sample
        normalized_all_a_length, normalized_all_b_length = normalized_all_length[:num_sample], normalized_all_length[num_sample:] # num_sample
    
        axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes
        axes_pure = np.array([axes_pure]) if not isinstance(axes_pure, np.ndarray) else axes_pure

        for i, (ax, ax_pure) in enumerate(zip(axes.flatten(), axes_pure.flatten())):
            all_mu_a_i = np.stack([sample[i,:] for sample in all_mu_a_list], axis=0) # num_sample, self.dim_factor
            all_mu_b_i = np.stack([sample[i,:] for sample in all_mu_b_list], axis=0) # num_sample, self.dim_factor
            diff = np.abs(all_mu_a_i.mean() - all_mu_b_i.mean())
            all_mu_i = np.concatenate((all_mu_a_i, all_mu_b_i), axis=0)# 2*num_sample  self.dim_factor
            label_factor = np.tile(np.arange(2), num_sample)
            assert all_mu_i.shape[0] == label_factor.shape[0] == len(all_length) == 2*num_sample, "big error!!!"
            result_all_mu_i = tsne.fit_transform(all_mu_i)# 2*num_sample  2
            result_all_mu_i = (result_all_mu_i - np.min(result_all_mu_i, 0)) / (np.max(result_all_mu_i, 0) - np.min(result_all_mu_i, 0)) # 2*num_sample  2
            result_all_mu_a, result_all_mu_b = result_all_mu_i[:num_sample, :], result_all_mu_i[num_sample:, :] # num_sample, 2
            sc1 = ax.scatter(result_all_mu_a[:,0].squeeze(), result_all_mu_a[:,1].squeeze(), c=normalized_all_a_length.squeeze(), cmap=custom_cmap[0], vmin=-0.2, vmax=1.0, alpha=0.5, marker='o', s=0.1)
            sc2 = ax.scatter(result_all_mu_b[:,0].squeeze(), result_all_mu_b[:,1].squeeze(), c=normalized_all_b_length.squeeze(), cmap=custom_cmap[1], vmin=-0.2, vmax=1.0 , alpha=0.5, marker='o', s=0.1)
            sc3 = ax_pure.scatter(result_all_mu_a[:,0].squeeze(), result_all_mu_a[:,1].squeeze(), color=custom_colors[0], alpha=0.5, marker='o', s=0.1)
            sc4 = ax_pure.scatter(result_all_mu_b[:,0].squeeze(), result_all_mu_b[:,1].squeeze(), color=custom_colors[1], alpha=0.5, marker='o', s=0.1)

            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'Mean of Factor {i}-{diff:.3f}', fontsize=3)
            ax_pure.set_xticks([]); ax_pure.set_yticks([]); ax_pure.set_title(f'Mean of Factor {i}-{diff:.3f}', fontsize=3)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle("Factor-TSNE of {} Mean".format(dataset_name), fontsize=16)
        fig_pure.suptitle("Factor-TSNE of {} Mean".format(dataset_name), fontsize=16)
        print(f"begin to save")
        start = time.time()
        fig.savefig(save_path.replace("tsne_", "factor_tsne_mean_"), dpi=1200, bbox_inches='tight')
        fig_pure.savefig(save_path.replace("tsne_", "pure_factor_tsne_mean_"), dpi=1200, bbox_inches='tight')
        print("finish saving at {}".format(save_path.replace("tsne_", "factor_tsne_mean_")))
        consume = time.time() - start
        print("finish saving at {} costs {}".format(save_path.replace("tsne_", "factor_tsne_mean_"), consume))
        
        return 0



    
    def _draw_dynamic_tsne(self, paired_rewards, save_path, num):#[[idx, ((length, reward, [mu], [std]), (length, reward, [mu], [std]))]...] dynamic_tsne_"+dataset_name+".png"
        random.seed(123)
        paired_rewards = random.sample(paired_rewards, min(len(paired_rewards), num))
        dataset_name = os.path.basename(save_path).split('.html')[0].split('dynamic_tsne_')[-1]
        num_sample = len(paired_rewards)
        factor_num = 1; factor_width = 1
        dim_factor =len((paired_rewards[0][-1][0][-1]))
        
        all_mu_a_list = [np.array(paired_rewards[i][-1][0][2]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))]  # [self.factor_num, self.dim_factor] * num_sample
        all_mu_b_list = [np.array(paired_rewards[i][-1][1][2]).reshape(factor_num*factor_width, dim_factor) for i in range(len(paired_rewards))]
        # [self.factor_num, self.dim_factor] * num_sample
        all_id_list = [paired_rewards[i][0] for i in range(len(paired_rewards))]

        all_a_length = [paired_rewards[i][-1][0][0].item() for i in range(len(paired_rewards))]
        all_b_length = [paired_rewards[i][-1][1][0].item() for i in range(len(paired_rewards))]
        all_length = np.concatenate((all_a_length, all_b_length), axis=0)# 2*num_sample 
        all_length_log = np.log10(all_length) # = all_length 
        normalized_all_length_log = np.array([(x - min(all_length_log)) / (max(all_length_log) - min(all_length_log)) for x in all_length_log]) # 2*num_sample
    #     normalized_all_length_log[normalized_all_length_log < 0.2] = 0.2
        normalized_all_a_length_log, normalized_all_b_length_log = normalized_all_length_log[:num_sample], normalized_all_length_log[num_sample:] # num_sample

        assert len(all_id_list) == len(all_mu_b_list) == len(all_mu_a_list) == normalized_all_a_length_log.shape[0] == normalized_all_b_length_log.shape[0]
        
        subplot_titles = [f"Factor {i}" for i in range(factor_num*factor_width)]
        fig = make_subplots(rows=factor_num, cols=factor_width, subplot_titles=subplot_titles)
        fig_pure = make_subplots(rows=factor_num, cols=factor_width, subplot_titles=subplot_titles)
        
        
        index = 0
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        for row in range(1, factor_num+1):
            for col in range(1, factor_width+1):
                all_mu_a_i = np.stack([sample[index,:] for sample in all_mu_a_list], axis=0) # num_sample, self.dim_factor
                all_mu_b_i = np.stack([sample[index,:] for sample in all_mu_b_list], axis=0) # num_sample, self.dim_factor
                all_mu_i = np.concatenate((all_mu_a_i, all_mu_b_i), axis=0)# 2*num_sample  self.dim_factor
                all_id = np.concatenate((all_id_list, all_id_list), axis=0).tolist()# 2*num_sample 
                all_id = [f"sample id {ii} length {len}" for ii, len in zip(all_id, all_length.tolist()) ]
                all_id_a, all_id_b = all_id[:num_sample], all_id[num_sample:]
                
                assert all_mu_i.shape[0] == len(all_id) == 2*num_sample, "big error!!!"
                result_all_mu_i = tsne.fit_transform(all_mu_i) 
                result_all_mu_i = (result_all_mu_i - np.min(result_all_mu_i, 0)) / (np.max(result_all_mu_i, 0) - np.min(result_all_mu_i, 0)) # 2*num_sample, 2
                result_all_mu_i_a, result_all_mu_i_b = result_all_mu_i[:num_sample, :], result_all_mu_i[num_sample:, :] # num_sample, 2

                fig.add_trace(go.Scatter(x=result_all_mu_i_a[:,0], y=result_all_mu_i_a[:,1], mode='markers',
                        marker=dict(size=5, opacity=0.5, colorscale='Reds',color=normalized_all_a_length_log, cmin=-0.2, cmax=1.0),
                        showlegend=False, text=all_id_a), row=row, col=col)
                fig.add_trace(go.Scatter(x=result_all_mu_i_b[:,0], y=result_all_mu_i_b[:,1], mode='markers',
                        marker=dict(size=5, opacity=0.5, colorscale='Blues',color=normalized_all_b_length_log, cmin=-0.2, cmax=1.0),
                        showlegend=False, text=all_id_b), row=row, col=col)
                
                fig_pure.add_trace(go.Scatter(x=result_all_mu_i_a[:,0], y=result_all_mu_i_a[:,1], mode='markers',
                        marker=dict(size=5, opacity=0.5), showlegend=False, text=all_id_a, marker_color='red'), row=row, col=col)
                fig_pure.add_trace(go.Scatter(x=result_all_mu_i_b[:,0], y=result_all_mu_i_b[:,1], mode='markers',
                        marker=dict(size=5, opacity=0.5), showlegend=False, text=all_id_b, marker_color='blue'), row=row, col=col)
                
                fig.update_xaxes(row=row, col=col, visible=False); fig.update_yaxes(row=row, col=col, visible=False)
                fig_pure.update_xaxes(row=row, col=col, visible=False); fig_pure.update_yaxes(row=row, col=col, visible=False)
                index = index + 1
                
        fig.update_layout(height=1800, width=1800, title_text="Factor-TSNE of {} Mean".format(dataset_name))
        fig_pure.update_layout(height=1800, width=1800, title_text="Factor-TSNE of {} Mean".format(dataset_name))
        print(f"begin to save")
        fig.write_html(save_path)
        fig_pure.write_html(save_path.replace("dynamic_tsne", "pure_dynamic_tsne"))
        print("finish saving at {}".format(save_path))

