from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("glm4_moe_lite_plus_plus")
class GLM4MoELitePlusPlusBridge(DeepseekV3Bridge):
    pass