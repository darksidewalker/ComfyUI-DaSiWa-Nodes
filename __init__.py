from .nodes.scaling_nodes import DaSiWa_ResolutionScaleCalculator
from .nodes.node_status_switch import DaSiWa_NodeStatusSwitch
from .nodes.rtx_upscaler_refiner import DaSiWa_RTX_UpscalerRefiner

NODE_CLASS_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator,
    "DaSiWa_NodeStatusSwitch": DaSiWa_NodeStatusSwitch,
    "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": "DaSiWa Resolution Scale Calculator",
    "DaSiWa_NodeStatusSwitch": "DaSiWa Node Status Switch",
     "DaSiWa_RTX_UpscalerRefiner": "DaSiWa RTX Upscaler & Refiner",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
