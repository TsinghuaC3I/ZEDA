# Adapted from glm4_moe_lite.py
# ==============================================================================

"""Inference-only GLM-Lite PlusPlus model with Zero-Compute Expert (ZCE) support.

Extends GLM4-MoE-Lite by adding zero-compute experts to the MoE routing.
Only the Gate, SparseMoeBlock, DecoderLayer, Model, and ForCausalLM are overridden.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.single_batch_overlap import SboFlags
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsWNA16MoEMethod,
)
from sglang.srt.layers.moe.ep_moe.kernels import zero_experts_compute_triton
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat, StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.glm4_moe_lite import (
    Glm4MoeLiteMLP,
    Glm4MoeLiteGate,
    Glm4MoeLiteSparseMoeBlock,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteModel,
    Glm4MoeLiteForCausalLM,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_device_sm,
    is_cuda,
    log_info_on_rank0,
    make_layers,
)

_is_cuda = is_cuda()
_device_sm = get_device_sm()

logger = logging.getLogger(__name__)


class Glm4MoeLitePlusPlusGate(Glm4MoeLiteGate):
    """Gate with expanded weight/bias for n_routed_experts + sum(zce_nums)."""

    def __init__(self, config, prefix: str = "", is_nextn: bool = False):
        nn.Module.__init__(self)
        self.is_nextn = is_nextn
        total_experts = config.n_routed_experts + sum(
            getattr(config, "zce_nums", [0])
        )
        self.weight = nn.Parameter(
            torch.empty((total_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((total_experts,), dtype=torch.float32)
        )

    def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class Glm4MoeLitePlusPlusSparseMoeBlock(Glm4MoeLiteSparseMoeBlock):
    """MoE block with Zero-Compute Expert (ZCE) support."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        # ZCE config
        self.zce_nums = getattr(config, "zce_nums", [0])
        self.zce_types = getattr(config, "zce_types", ["zero"])
        self.n_routed_experts = config.n_routed_experts

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Expanded gate for routed + ZCE experts
        self.gate = Glm4MoeLitePlusPlusGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )

        # Experts: only routed + fused shared (no ZCE MLP experts)
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        # TopK with expanded correction_bias
        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        # Shared experts (same as base)
        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            )
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype
                == torch.float8_e4m3fn
            )

        self.top_k = config.num_experts_per_tok

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def _handle_zce(self, topk_output, hidden_states):
        """Handle ZCE: mask out ZCE indices and compute ZCE contributions."""
        topk_weights, topk_ids, router_logits = topk_output
        zce_results = []
        for zce_type in self.zce_types:
            if zce_type == "zero":
                zce_mask = topk_ids >= self.n_routed_experts
                topk_weights[zce_mask] = 0.0
                topk_ids[zce_mask] = -1
            elif zce_type == "copy":
                zce_results.append(
                    zero_experts_compute_triton(
                        expert_indices=topk_ids,
                        expert_scales=topk_weights,
                        num_experts=self.n_routed_experts,
                        zero_expert_type=zce_type,
                        hidden_states=hidden_states,
                    )
                )
            else:
                raise ValueError(f"Invalid ZCE type: {zce_type}")
        return StandardTopKOutput(topk_weights, topk_ids, router_logits), zce_results

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        zce_results = []
        if hidden_states.shape[0] > 0:
            if not self._fuse_shared_experts_inside_sbo:
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits)
            # Handle ZCE before passing to real experts
            if self.zce_types is not None:
                topk_output, zce_results = self._handle_zce(
                    topk_output, hidden_states
                )
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if self._fuse_shared_experts_inside_sbo:
            shared_output = None

            def _forward_shared_experts_and_put_results():
                nonlocal shared_output
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )

        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
            **(
                dict(
                    forward_shared_experts=_forward_shared_experts_and_put_results,
                    alt_stream=self.alt_stream,
                )
                if self._fuse_shared_experts_inside_sbo
                else {}
            ),
        )
        if (
            not _is_cuda
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
            or isinstance(self.experts.quant_method, CompressedTensorsWNA16MoEMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states += shared_output

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        # Add ZCE contributions
        if zce_results and hidden_states.shape[0] > 0:
            for zce_result in zce_results:
                final_hidden_states += zce_result.to(final_hidden_states.device)

        return final_hidden_states

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(
            hidden_states, gemm_output_zero_allocator
        )

        zce_results = []
        with torch.cuda.stream(self.alt_stream):
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits)
            # Handle ZCE
            if self.zce_types is not None:
                topk_output, zce_results = self._handle_zce(
                    topk_output, hidden_states
                )
            final_hidden_states = self.experts(hidden_states, topk_output)
            if (
                not _is_cuda
                or isinstance(self.experts.quant_method, KTEPWrapperMethod)
                or isinstance(
                    self.experts.quant_method, CompressedTensorsWNA16MoEMethod
                )
            ):
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)
        final_hidden_states += shared_output

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        
        # Add ZCE contributions
        if zce_results and hidden_states.shape[0] > 0:
            for zce_result in zce_results:
                final_hidden_states += zce_result.to(final_hidden_states.device)
        
        return final_hidden_states

    def forward_deepep(self, hidden_states, forward_batch):
        raise NotImplementedError(
            "GLM4-MoE-Lite-PlusPlus is not supported for DeepEP."
        )


class Glm4MoeLitePlusPlusDecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config

        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        rope_theta = 1000000
        rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 202752)
        self.layer_id = layer_id

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            reduce_results=False,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Glm4MoeLitePlusPlusSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
        )


class Glm4MoeLitePlusPlusModel(Glm4MoeLiteModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()

        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.cp_size = get_attention_tp_size() if self.nsa_enable_prefill_cp else None
        self.gemm_output_zero_allocator_size = 0
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm4MoeLitePlusPlusDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []


class Glm4MoeLitePlusPlusForCausalLM(Glm4MoeLiteForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config.moe_layer_freq = 1
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("Glm4MoeLitePlusPlusForCausalLM")
        self.model = Glm4MoeLitePlusPlusModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Glm4MoeLitePlusPlusSparseMoeBlock)
            }
        )
        self.capture_aux_hidden_states = False

        from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_rank,
                get_attention_tp_size,
            )

            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_rank = self.cp_size = None


EntryClass = [Glm4MoeLitePlusPlusForCausalLM]
