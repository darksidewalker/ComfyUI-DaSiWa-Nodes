from .nodes.scaling_nodes import DaSiWa_ResolutionScaleCalculator
from .nodes.node_status_switch import DaSiWa_NodeStatusSwitch

NODE_CLASS_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator,
    "DaSiWa_NodeStatusSwitch": DaSiWa_NodeStatusSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": "DaSiWa Resolution Scale Calculator",
    "DaSiWa_NodeStatusSwitch": "DaSiWa Node Status Switch",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
