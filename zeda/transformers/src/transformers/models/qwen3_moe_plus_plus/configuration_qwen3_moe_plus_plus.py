from ..qwen3_moe import Qwen3MoeConfig
from typing import List

class Qwen3MoePlusPlusConfig(Qwen3MoeConfig):
    r"""
    Configuration class for Qwen3 MoE++.
    """
    model_type = "qwen3_moe_plus_plus"
    def __init__(
        self,
        zce_nums: List[int] = [64],
        zce_types: List[str] = ["copy"],
        use_zce_mask: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.zce_nums = zce_nums
        self.zce_types = zce_types
        self.use_zce_mask = use_zce_mask
    @property
    def total_zce(self): return sum(self.zce_nums)

__all__ = ["Qwen3MoePlusPlusConfig"]
