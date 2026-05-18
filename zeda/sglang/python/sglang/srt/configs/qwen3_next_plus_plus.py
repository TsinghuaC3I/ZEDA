# coding=utf-8

from typing import List

from sglang.srt.configs.qwen3_next import Qwen3NextConfig


class Qwen3NextPlusPlusConfig(Qwen3NextConfig):
    model_type = "qwen3_next_plus_plus"

    def __init__(
        self,
        zce_nums: List[int] = [64],
        zce_types: List[str] = ["copy"],
        use_zce_mask: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.zce_nums = zce_nums
        self.zce_types = zce_types
        self.use_zce_mask = use_zce_mask

    @property
    def total_zce(self):
        return sum(self.zce_nums)
