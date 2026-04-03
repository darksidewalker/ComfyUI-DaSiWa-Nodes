from .py.scaling_nodes import DaSiWa_ResolutionScaler

NODE_CLASS_MAPPINGS = {
    "DaSiWa-ResolutionScaler": DaSiWa_ResolutionScaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa-ResolutionScaler": "DaSiWa Resolution Scaler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]