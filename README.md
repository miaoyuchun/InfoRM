# InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling
This repository contains the official implementation of this article

**[[NeurIPS 2024] InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling][1]**

[Yuchun Miao][myc], [Sen Zhang][zs], [Liang Ding][dl], [Rong Bao][br], [Lefei Zhang][zlf], [Dacheng Tao][tdc]


## Installation
```
conda create -n inform python=3.12 -y
conda activate inform

pip install vllm==0.7.2
git clone https://github.com/miaoyuchun/InfoRM.git
cd InfoRM
pip install -e .
```

# Prepare Datasets
The data format used in this project is fully consistent with that of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

 To reproduce this work, you can process the [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) dataset for SFT and the [Anthropic HH](https://arxiv.org/pdf/2204.05862.) dataset for RM and PPO training following the data format specifications provided by OpenRLHF.

## Supervised Fine-tuning

```
bash ./example_sft/slurm/scc_sft_llama3_8b_sharegpt_packing.sh
```
Before running the above commands, you should replace `WORKSPACE`, `pretrain`, and `dataset` with the corresponding project path, pretrained model path, and dataset path, respectively.

## RM Training

### Information-Theoretic Reward Model
```
bash ./example_rm/slurm/scc_rm_llama3_hh105_wprompt_packing_inform.sh
```

### Standard Reward Model
```
bash ./example_rm/slurm/scc_rm_llama3_hh105_wprompt_packing_baseline.sh
```

Before running the above commands, you should replace `WORKSPACE`, `pretrain`, and `dataset` with the corresponding project path, sft model path, and dataset path, respectively.

## PPO Training

### PPO with InfoRM
```
bash ./example_ppo/slurm/scc_ppo_ray_llama3_8b_hh105rm_reproduce_hacking_offload_inform.sh
```

### PPO with Standard-RM
```
bash ./example_ppo/slurm/scc_ppo_ray_llama3_8b_hh105rm_reproduce_hacking_offload_baseline.sh
```
* Before running the above commands, you should replace `WORKSPACE`, `pretrain`, `reward_pretrain`, and `dataset` with the corresponding project path, sft model path, rm model path, and dataset path, respectively.

* `./example_ppo/script/generate_eval.sh` is used to generate responses using the SFT and RLHF models.  This step is already included in the PPO training script above.

* `./example_ppo/script/ib_latent_eval.sh` is used to generate T-SNE visualizations based on the representations of samples in the latent space of InfoRM. You can assess the extent of reward hacking by identifying outliers in these plots. This step is also included in the PPO training script above.

## Citation
If you find our work useful in your research, please cite:

```
@inproceedings{
miao2024inform,
title={Info{RM}: Mitigating Reward Hacking in {RLHF} via Information-Theoretic Reward Modeling},
author={Yuchun Miao and Sen Zhang and Liang Ding and Rong Bao and Lefei Zhang and Dacheng Tao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=3XnBVK9sD6}
}
```


[1]: https://arxiv.org/abs/2402.09345
[myc]: https://scholar.google.com/citations?user=-ec3mwUAAAAJ&hl=en
[zs]: https://scholar.google.com/citations?user=-bJJNV0AAAAJ&hl=en
[dl]: https://scholar.google.com/citations?user=lFCLvOAAAAAJ&hl=en
[br]: https://scholar.google.com/citations?user=teGqP3kAAAAJ
[zlf]: https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=en
[tdc]: https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en


# Thanks
This project is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Thanks for this wonderful work!<br>