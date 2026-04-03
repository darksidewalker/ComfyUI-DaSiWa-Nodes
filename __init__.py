from .nodes.scaling_nodes import DaSiWa_ResolutionScaleCalculator

NODE_CLASS_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": "DaSiWa Resolution Scale Calculator"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]