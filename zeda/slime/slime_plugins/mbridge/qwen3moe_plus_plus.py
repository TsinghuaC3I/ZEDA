import torch
from mbridge.core import register_model
from mbridge.models import Qwen3MoEBridge


@register_model("qwen3_moe_plus_plus")
class Qwen3MoEPlusPlusBridge(Qwen3MoEBridge):
    pass