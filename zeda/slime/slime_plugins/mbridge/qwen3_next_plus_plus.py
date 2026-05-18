import torch
from mbridge.core import register_model
from .qwen3_next import Qwen3NextBridge


@register_model("qwen3_next_plus_plus")
class Qwen3NextPlusPlusBridge(Qwen3NextBridge):
    pass