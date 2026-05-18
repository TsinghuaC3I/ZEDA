import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from megatron.core import parallel_state

class MoERouterTracker:
    """
    A lightweight, hook-based tracker for MoE Router statistics.
    Designed for Megatron-LM backends within the Slime framework.

    It tracks:
    1. Expert Selection Counts (How many tokens were routed to each expert).
    
    And computes the following metrics:
    1. Max/Average Violation
    2. Average/Standard Deviation of Zero Expert Ratio
    """

    def __init__(
        self,
        args,
    ):
        """
        Initialize the tracker.
        """
        self.num_experts = args.num_experts
        self.num_zero_experts = getattr(args, 'num_zero_experts', None)
        self.total_experts = self.num_experts
        if self.num_zero_experts is not None:
            self.total_experts += self.num_zero_experts
        self.top_k = args.moe_router_topk
        
        # Storage for hooks handles to allow removal
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Accumulators (Tensor buffers on GPU)
        # Structure: {layer_name: Tensor(shape=[total_experts])}
        self.expert_counts: Dict[str, torch.Tensor] = {}
        self.step_count = 0

    def _get_or_create_buffers(self, layer_name: str, device: torch.device):
        """
        Lazy initialization of GPU buffers for a specific layer.
        """
        if layer_name not in self.expert_counts:
            # Initialize with zeros on the correct device
            self.expert_counts[layer_name] = torch.zeros(
                self.total_experts, dtype=torch.long, device=device
            )

    def _hook_fn(self, layer_name: str):
        """
        Factory to create the actual hook function with a closure over layer_name.
        """
        def hook(module: nn.Module, input: Tuple, output: Tuple):
            """
            The forward hook executed after the router's forward pass.
            
            Expected Output Signature from Megatron Router:
                (probs, routing_map)
                probs: [batch_size * seq_len, total_experts] or [batch, seq, total_experts]
                routing_map: [batch_size * seq_len, total_experts] or [batch, seq, total_experts]
            """
            # Ensure we are in no_grad context to save memory and computation
            with torch.no_grad():
                # Unpack output. Adjust this unpacking logic based on specific Megatron version
                if isinstance(output, tuple):
                    probs, routing_map = output[0], output[1]
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")
                
                device = routing_map.device
                self._get_or_create_buffers(layer_name, device)

                counts = routing_map.detach().view(-1, routing_map.shape[-1]).sum(dim=0)
                
                if counts.shape[0] != self.total_experts:
                    raise ValueError(f"Counts shape {counts.shape} mismatches total_experts {self.total_experts}")
                
                self.expert_counts[layer_name] += counts

        return hook

    def register_hooks(self, model: nn.Module):
        """
        Traverses the model and registers hooks on identified Router modules.
        
        Args:
            model (nn.Module): The root model to traverse.
        """

        def _extract_module(m):
            if hasattr(m, 'module'):
                if hasattr(m.module, 'module'):
                    return m.module.module
                return m.module
            return m

        self.reset()
        self.hooks = []

        if not isinstance(model, list):
            model_list = [model]
        else:
            model_list = model
        
        for model in model_list:
            extracted_model = _extract_module(model)
            if hasattr(extracted_model, 'decoder') and hasattr(extracted_model.decoder, 'layers'):
                for idx, layer in enumerate(extracted_model.decoder.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                        router = layer.mlp.router
                        handle = router.register_forward_hook(self._hook_fn(f"layer_{idx}"))
                        self.hooks.append(handle)
                
        print(f"[MoERouterTracker] Registered {len(self.hooks)} hooks.")

    def reset(self):
        """
        Resets internal statistics. Call this at the start of an epoch or after logging.
        """
        # Zero out existing tensors instead of deleting to save allocation time
        for k in self.expert_counts:
            self.expert_counts[k].zero_()
        self.step_count = 0

    def get_metrics(self) -> Dict[str, float]:
        """
        Aggregates metrics from GPU, synchronizes across distributed ranks (if needed),
        and returns a dictionary of scalar metrics suitable for logging.
        
        Args:
            group (dist.ProcessGroup, optional): PyTorch distributed group for reduction. 
                                                 If None, results are local.
        
        Returns:
            Dict[str, float]: Flat dictionary of metrics.
        """
        metrics = {}
        
        # Guard: If no data collected
        if not self.expert_counts:
            return metrics

        # Iterate over layers
        for layer_name in self.expert_counts.keys():
            counts = self.expert_counts[layer_name]

            # 1. Distributed Synchronization
            # We must clone before reducing if we want to keep accumulating, 
            # or we can reduce in-place if we reset immediately after.
            # Here we clone to ensure safety.
            global_counts = counts.clone()

            if parallel_state.is_initialized() and parallel_state.get_data_parallel_world_size() > 1:
                dp_group = parallel_state.get_data_parallel_group()
                if dp_group is not None:
                    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=dp_group)

            if parallel_state.is_initialized() and parallel_state.get_expert_model_parallel_world_size() > 1:
                ep_group = parallel_state.get_expert_model_parallel_group()
                if ep_group is not None:
                    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=ep_group)

            # Move to CPU for metric calculation
            # Note: This syncs the CPU, so only call this when logging (e.g. every 100 steps)
            global_counts_cpu = global_counts.float().cpu()
            clean_name = layer_name.replace('model.', '')

            # -- Metric: Max/Average Violation --
            global_counts_normal_experts = global_counts_cpu[:self.num_experts]
            mean_counts = global_counts_normal_experts.mean()
            max_counts = global_counts_normal_experts.max()
            max_violation = (max_counts - mean_counts) / (mean_counts + 1e-8)
            avg_violation = torch.abs(global_counts_normal_experts - mean_counts).mean() / (mean_counts + 1e-8)
            metrics[f"{clean_name}/max_violation"] = max_violation.item()
            metrics[f"{clean_name}/avg_violation"] = avg_violation.item()

            # -- Metric: Zero Expert Ratio --
            if self.num_zero_experts:
                zero_expert_counts = global_counts_cpu[self.num_experts:].sum()
                metrics[f"{clean_name}/zero_expert_ratio"] = zero_expert_counts / (global_counts_cpu.sum() + 1e-8)

        # Across All Layers
        max_violation = torch.tensor([metrics[f"{layer_name}/max_violation"] for layer_name in self.expert_counts.keys()]).mean()
        avg_violation = torch.tensor([metrics[f"{layer_name}/avg_violation"] for layer_name in self.expert_counts.keys()]).mean()
        metrics["max_violation"] = max_violation.item()
        metrics["avg_violation"] = avg_violation.item()

        if self.num_zero_experts:
            zero_expert_ratio = torch.tensor([metrics[f"{layer_name}/zero_expert_ratio"] for layer_name in self.expert_counts.keys()]).mean()        
            metrics["zero_expert_ratio"] = zero_expert_ratio.item()

        return metrics

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []