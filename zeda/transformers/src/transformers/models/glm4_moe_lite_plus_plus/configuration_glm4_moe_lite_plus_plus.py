from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from typing import List


class Glm4MoeLitePlusPlusConfig(Glm4MoeLiteConfig):
    r"""
    Configuration class for GLM4 MoE Lite++.
    Extends Glm4MoeLiteConfig with zero-compute expert (ZCE) support.
    """
    model_type = "glm4_moe_lite_plus_plus"

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
    def total_zce(self):
        return sum(self.zce_nums)


__all__ = ["Glm4MoeLitePlusPlusConfig"]
