from typing import Optional, Union
import numpy as np
import deepspeed
import torch
import torch.nn as nn
from torch.autograd import Variable
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from openrlhf.utils.logging_utils import init_logger
from scipy.spatial import distance
from .ring_attn_utils import convert_ring_attn_params
from .utils import reset_position_ids

logger = init_logger(__name__)


def kl_divergence(mu, logvar):
    assert  mu.data.ndimension() == logvar.data.ndimension() == 1

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()) # torch.Size([bs])
    kl_loss = klds.mean() # torch.Size([1])
    
    return kl_loss

def cal_mahalanobis(buffer, ib_representation):
    mean_data = np.mean(buffer, axis=0)
    cov_matrix = np.cov(buffer.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distance_list = []
    for i in range(ib_representation.shape[0]):
        mahalanobis_distance = distance.mahalanobis(ib_representation[i], mean_data, inv_cov_matrix)
        mahalanobis_distance_list.append(mahalanobis_distance)

    return torch.tensor(mahalanobis_distance_list).cuda()

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.intermediate_size = self.latent_dim * 4
        self.gate_proj = nn.Linear(self.latent_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.latent_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, 1, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


# class MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.latent_dim = config.latent_dim
#         self.network = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim*8, bias=True), nn.Sigmoid(), 
#                                      nn.Linear(self.latent_dim*8, self.latent_dim*16, bias=True), nn.Sigmoid(), 
#                                      nn.Linear(self.latent_dim*16, self.latent_dim*32, bias=True), nn.Sigmoid(), 
#                                      nn.Linear(self.latent_dim*32, self.latent_dim*16, bias=True), nn.Sigmoid(), 
#                                      nn.Linear(self.latent_dim*16, self.latent_dim*8, bias=True), nn.Sigmoid(), 
#                                      nn.Linear(self.latent_dim*8, 1, bias=True))


#     def forward(self, x):
#         score = self.network(x)
#         return score
    
# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression_inform(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    encode_head_prefix="encode_head",
    decode_head_prefix="decode_head",
    device_map=None,
    packing_samples=False,
    latent_dim=128,
    ibl_coef=None,
    ibl_path=None,
    use_complex_decoder=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.a
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # InfoRM
    config.use_inform = True
    config.latent_dim = latent_dim if 'latent_dim' not in config else config.latent_dim
    config.encode_head_prefix = encode_head_prefix if 'encode_head_prefix' not in config else config.encode_head_prefix
    config.decode_head_prefix = decode_head_prefix if 'decode_head_prefix' not in config else config.decode_head_prefix
    config.use_complex_decoder = use_complex_decoder if 'use_complex_decoder' not in config else config.use_complex_decoder

    config.ibl_coef, config.ibl_path = ibl_coef, ibl_path

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, packing_samples=packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, packing_samples=packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        raise NotImplementedError("This function is not implemented yet.")
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        raise NotImplementedError("This function is not implemented yet.")

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        raise NotImplementedError("This function is not implemented yet.")
    
    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        raise NotImplementedError("This function is not implemented yet.")
        # encode_head_prefix = getattr(model, encode_head_prefix)
        # if dschf is not None:
        #     logger.info("initialize encode_head_prefix for ZeRO-3 reward model training.")
        #     with deepspeed.zero.GatheredParameters([encode_head_prefix.weight], modifier_rank=0):
        #         if torch.distributed.get_rank() == 0:
        #             encode_head_prefix.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size*config.latent_dim + 1))
        # else:
        #     encode_head_prefix.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size*config.latent_dim + 1))

        # decode_head_prefix = getattr(model, decode_head_prefix)
        # if dschf is not None:
        #     logger.info("initialize decode_head_prefix for ZeRO-3 reward model training.")
        #     with deepspeed.zero.GatheredParameters([decode_head_prefix.weight], modifier_rank=0):
        #         if torch.distributed.get_rank() == 0:
        #             decode_head_prefix.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim + 1))
        # else:        #     decode_head_prefix.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim + 1))

    return model

# base_llm_model corresponds to LlamaModel; base_pretrained_model corresponds to LlamaPreTrainedModel
def _get_reward_model(base_pretrained_model, base_llm_model, packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.encode_head_prefix = config.encode_head_prefix
            self.decode_head_prefix = config.decode_head_prefix

            self.latent_dim = config.latent_dim
            self.use_complex_decoder = config.use_complex_decoder

            if self.use_complex_decoder:
                decoder = MLP(config)
                setattr(self, self.decode_head_prefix, decoder)
            else:
                setattr(self, self.decode_head_prefix, nn.Linear(self.latent_dim, 1, bias=False))
      
            setattr(self, self.encode_head_prefix, nn.Linear(config.hidden_size, self.latent_dim*2, bias=False))
            
            self.post_init()
            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

            # whether print the reward obtained position 
            self.print_reward_position=True
            # self.variables_list = []

            self.ibl_coef = config.ibl_coef
            if self.ibl_coef:
                assert len(self.ibl_coef) == 2
                ibl_path_list = config.ibl_path.split(',')
                for ibl_path in ibl_path_list:
                    if 'harmless' in ibl_path:
                        self.harmless_ib_buffer = np.array(np.load(ibl_path))
                    elif 'helpful' in ibl_path:
                        self.helpful_ib_buffer = np.array(np.load(ibl_path))
                    else:
                        self.helpful_ib_buffer, self.harmless_ib_buffer = None, None
            
                # assert self.helpful_ib_buffer and self.harmless_ib_buffer


        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
            prompts=None,
            classes=None
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # convert attention_mask to position_ids
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    position_ids = reset_position_ids(attention_mask)
                # explicitly ignore attention_mask for packing_samples
                attention_mask = None

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            encode_result = getattr(self, self.encode_head_prefix)(last_hidden_states) # torch.Size([4, 418, self.latent_dim*2])
            mu = encode_result[:, :, :self.latent_dim] # torch.Size([4, 418, self.latent_dim])
            logvar = encode_result[:, :, self.latent_dim:] # torch.Size([4, 418, self.latent_dim])
            std = logvar.div(2).exp() # torch.Size([4, 418, self.latent_dim])
            eps = Variable(std.data.new(std.size()).normal_()) # torch.Size([4, 418, self.latent_dim])
            decode_input = mu + std * eps if self.training else mu # torch.Size([4, 418, self.latent_dim])
            values = getattr(self, self.decode_head_prefix)(decode_input).squeeze(-1) # torch.Size([4, 418]) [2*bs, seq]

            if self.packing_samples:
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                if ring_attn_group is not None:
                    raise NotImplementedError("This function is not implemented yet.")
                else:
                    reward = values
                reward = reward.squeeze(0).gather(dim=0, index=eos_indices)

                ib_representation = mu.squeeze(0)[eos_indices, :]  # [2*bs, latent_dim]
                mu_mean = mu.squeeze(0)[eos_indices, :].mean(dim=-1) # [2*bs]
                ib_representation_logvar = logvar.squeeze(0)[eos_indices, :]  # [2*bs, latent_dim]
                logvar_mean = logvar.squeeze(0)[eos_indices, :].mean(dim=-1) # [2*bs]
                kl_loss = kl_divergence(mu_mean, logvar_mean)

                if self.print_reward_position:
                    print("Reward is obtained at {}".format(input_ids[0,eos_indices].squeeze().tolist()))
                    print("Before {}".format(input_ids[0,eos_indices-1].squeeze().tolist()))
                    self.print_reward_position = False
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                ib_representation = mu[torch.arange(input_ids.shape[0]), eos_indices.squeeze(-1), :]  # [2*bs, latent_dim]
                mu_mean = mu[torch.arange(input_ids.shape[0]), eos_indices.squeeze(-1), :].mean(-1) # [2*bs]
                ib_representation_logvar = logvar[torch.arange(input_ids.shape[0]), eos_indices.squeeze(-1), :]  # [2*bs, latent_dim]
                logvar_mean = logvar[torch.arange(input_ids.shape[0]), eos_indices.squeeze(-1), :].mean(-1) # [2*bs]
                kl_loss = kl_divergence(mu_mean, logvar_mean)

                if self.print_reward_position:
                    print("Reward is obtained at {}".format(input_ids.gather(dim=1, index=eos_indices).squeeze().tolist()))
                    print("Before {}".format(input_ids.gather(dim=1, index=eos_indices-1).squeeze().tolist()))

                    self.print_reward_position = False

            if not self.training and self.ibl_coef:
                assert classes and len(classes) == ib_representation.shape[0]
                helpful_mask = torch.tensor([1 if class_ == 'helpful' else 0 for class_ in classes], dtype=torch.bool).cuda()
                # print("helpful_mask is {}".format(helpful_mask))
                harmless_mask = torch.tensor([1 if class_ == 'harmless' else 0 for class_ in classes], dtype=torch.bool).cuda()
                # print("harmless_mask is {}".format(harmless_mask))

                ib_representation_helpful = ib_representation[helpful_mask]
                # print("the shape of ib_representation_helpful is {}".format(ib_representation_helpful.shape))
                ib_representation_harmless = ib_representation[harmless_mask]
                # print("the shape of ib_representation_harmless is {}".format(ib_representation_harmless.shape))

                helpful_mahalanobis_distance = cal_mahalanobis(self.helpful_ib_buffer, ib_representation_helpful.to(torch.float32).detach().cpu().numpy())
                # print("the shape of helpful_mahalanobis_distance is {}".format(helpful_mahalanobis_distance.shape))
                harmless_mahalanobis_distance = cal_mahalanobis(self.harmless_ib_buffer, ib_representation_harmless.to(torch.float32).detach().cpu().numpy())
                # print("the shape of harmless_mahalanobis_distance is {}".format(harmless_mahalanobis_distance.shape))

                all_mahalanobis_distance = torch.full_like(reward, float('inf')).cuda()
                all_mahalanobis_distance[helpful_mask] = helpful_mahalanobis_distance.to(torch.bfloat16) * self.ibl_coef[0] 
                all_mahalanobis_distance[harmless_mask] = harmless_mahalanobis_distance.to(torch.bfloat16) * self.ibl_coef[1]
                # print("the shape of all_mahalanobis_distance is {}".format(all_mahalanobis_distance.shape))
                assert (~torch.isinf(all_mahalanobis_distance)).all()

                # mahalanobis_distance = cal_mahalanobis(self.ib_buffer, ib_representation.to(torch.float32).detach().cpu().numpy())
                # reward = reward - self.ibl_coef * all_mahalanobis_distance if self.ibl_coef > 0 else reward
                reward = reward - all_mahalanobis_distance
            else:
                helpful_mahalanobis_distance = torch.zeros_like(reward)
                harmless_mahalanobis_distance = torch.zeros_like(reward)


            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            outputs["kl_loss"] = kl_loss
            outputs["ib_representation"] = ib_representation.tolist()
            outputs["ib_representation_logvar"] = ib_representation_logvar.tolist()
            outputs["mu_mean"] = mu_mean.tolist()
            outputs["logvar_mean"] = logvar_mean.tolist()
            outputs["mahalanobis_helpful"] = [torch.mean(helpful_mahalanobis_distance).item()] * len(reward)
            outputs["mahalanobis_harmless"] = [torch.mean(harmless_mahalanobis_distance).item()] * len(reward)

            # outputs["mahalanobis"] = mahalanobis_distance.tolist()
            # print("miao debug info: mahalanobis_distance is {}".format(mahalanobis_distance.tolist()))
#             self.variables_list.append(eps)
#             print("length of random variable is {}".format(len(set(tuple(var.flatten().tolist()) for var in self.variables_list)
# )))

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.encode_head_prefix = config.encode_head_prefix
            self.decode_head_prefix = config.decode_head_prefix

            self.latent_dim = config.latent_dim
            self.use_complex_decoder = config.use_complex_decoder

            if self.use_complex_decoder:
                decoder = MLP(config)
                setattr(self, self.decode_head_prefix, decoder)
            else:
                setattr(self, self.decode_head_prefix, nn.Linear(self.latent_dim, 1, bias=False))
      
            setattr(self, self.encode_head_prefix, nn.Linear(config.hidden_size, self.latent_dim*2, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # convert attention_mask to position_ids
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    position_ids = reset_position_ids(attention_mask)
                # explicitly ignore attention_mask for packing_samples
                attention_mask = None

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            encode_result = getattr(self, self.encode_head_prefix)(last_hidden_states) # torch.Size([4, 418, self.latent_dim*2])
            mu = encode_result[:, :, :self.latent_dim] # torch.Size([4, 418, self.latent_dim])
            logvar = encode_result[:, :, self.latent_dim:] # torch.Size([4, 418, self.latent_dim])
            std = logvar.div(2).exp() # torch.Size([4, 418, self.latent_dim])
            decode_input = mu # torch.Size([4, 418, self.latent_dim])
            values = getattr(self, self.decode_head_prefix)(decode_input).squeeze(-1) # torch.Size([4, 418]) [2*bs, seq]

            if ring_attn_group is not None and values_allgather:
                values = all_gather(values, ring_attn_group).reshape(values.shape[0], -1)[:, :-1]
            else:
                values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if num_actions is None:
                assert return_output
                return outputs

            if not self.packing_samples:
                action_values = values[:, -num_actions:]
            else:
                assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
                action_values = []
                offset = 0
                for num_action, seq_len in zip(num_actions, packed_seq_lens):
                    start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                    action_values.append(values[:, start:end])
                    offset += seq_len
                action_values = torch.cat(action_values, dim=1)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
