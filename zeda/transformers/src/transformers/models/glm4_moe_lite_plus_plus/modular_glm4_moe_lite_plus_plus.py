# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteModel,
    Glm4MoeLiteMoE,
    Glm4MoeLitePreTrainedModel,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteRotaryEmbedding,
    Glm4MoeLiteTopkRouter,
)
from .configuration_glm4_moe_lite_plus_plus import Glm4MoeLitePlusPlusConfig


class CopyExpert(nn.Module):
    def __init__(self, hidden_size):
        super(CopyExpert, self).__init__()
        pass

    def forward(self, inputs):
        return inputs


class ZeroExpert(nn.Module):
    def __init__(self, hidden_size):
        super(ZeroExpert, self).__init__()
        pass

    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(nn.Module):
    def __init__(self, hidden_size):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(torch.empty((hidden_size)))
        torch.nn.init.normal_(self.constant)
        self.wg = torch.nn.Linear(hidden_size, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        weight = self.wg(inputs)
        weight = self.softmax(weight)
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
            'b,d->bd', [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)])


class Glm4MoeLitePlusPlusRotaryEmbedding(Glm4MoeLiteRotaryEmbedding):
    pass


class Glm4MoeLitePlusPlusAttention(Glm4MoeLiteAttention):
    pass


class Glm4MoeLitePlusPlusMLP(Glm4MoeLiteMLP):
    pass


class Glm4MoeLitePlusPlusRMSNorm(Glm4MoeLiteRMSNorm):
    pass


class Glm4MoeLitePlusPlusTopkRouter(nn.Module):
    """Router that supports both routed and ZCE experts."""

    def __init__(self, config: Glm4MoeLitePlusPlusConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.total_num_experts = config.n_routed_experts + sum(config.zce_nums)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.total_num_experts, config.hidden_size)))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros((self.total_num_experts), dtype=torch.float32),
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class Glm4MoeLitePlusPlusMoE(nn.Module):
    """MoE module with shared experts and ZCE expert support."""

    def __init__(self, config: Glm4MoeLitePlusPlusConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.zce_nums = config.zce_nums
        self.zce_types = config.zce_types
        self.use_zce_mask = config.use_zce_mask
        self.total_num_experts = config.n_routed_experts + sum(config.zce_nums)

        # Routed MLP experts
        self.experts = nn.ModuleList(
            [
                Glm4MoeLitePlusPlusMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        # ZCE experts
        for zce_type, zce_num in zip(self.zce_types, self.zce_nums):
            self.experts.extend(
                [self._get_zce_class(zce_type)(config.hidden_size) for _ in range(zce_num)]
            )

        assert self.total_num_experts == len(self.experts), (
            "The total number of experts should be equal to the sum of routed experts and ZCE experts"
        )

        self.gate = Glm4MoeLitePlusPlusTopkRouter(config)
        self.shared_experts = Glm4MoeLitePlusPlusMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def _get_zce_class(self, zce_type: str):
        if zce_type == "copy":
            return CopyExpert
        elif zce_type == "zero":
            return ZeroExpert
        elif zce_type == "constant":
            return ConstantExpert
        else:
            raise ValueError(f"Invalid ZCE type: {zce_type}")

    def route_tokens_to_experts(self, router_logits):
        # ZCE Mask: mask ZCE expert logits before sigmoid
        if self.use_zce_mask:
            zce_mask = torch.zeros_like(router_logits).bool()
            zce_mask[:, self.n_routed_experts:] = True
            router_logits = router_logits.masked_fill(zce_mask, float('-inf'))

        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        # Group routing (n_group=1 makes this straightforward)
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.total_num_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.total_num_experts // self.n_group)
            .reshape(-1, self.total_num_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        """Route tokens to experts (including ZCE) and compute weighted sum."""
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.total_num_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(self.total_num_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class Glm4MoeLitePlusPlusDecoderLayer(Glm4MoeLiteDecoderLayer, nn.Module):
    def __init__(self, config: Glm4MoeLitePlusPlusConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4MoeLitePlusPlusAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm4MoeLitePlusPlusMoE(config)
        else:
            self.mlp = Glm4MoeLitePlusPlusMLP(config)

        self.input_layernorm = Glm4MoeLitePlusPlusRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLitePlusPlusRMSNorm(config.hidden_size, config.rms_norm_eps)


class Glm4MoeLitePlusPlusPreTrainedModel(Glm4MoeLitePreTrainedModel):
    pass


class Glm4MoeLitePlusPlusModel(Glm4MoeLiteModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]


class Glm4MoeLitePlusPlusForCausalLM(Glm4MoeLiteForCausalLM):
    pass


__all__ = [
    "Glm4MoeLitePlusPlusConfig",
    "Glm4MoeLitePlusPlusPreTrainedModel",
    "Glm4MoeLitePlusPlusModel",
    "Glm4MoeLitePlusPlusForCausalLM",
]
