from .nodes.scaling_nodes import DaSiWa_ResolutionScaleCalculator
from .nodes.node_status_switch import DaSiWa_NodeStatusSwitch
from .nodes.rtx_upscaler_refiner import DaSiWa_RTX_UpscalerRefiner
from .nodes.metadata_nodes import DaSiWa_MetadataImageSaver, DaSiWa_MetadataImageSaverFull, DaSiWa_MetadataConfig, DaSiWa_CreateExtraMetadata

NODE_CLASS_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator,
    "DaSiWa_NodeStatusSwitch": DaSiWa_NodeStatusSwitch,
    "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
    "DaSiWa_MetadataImageSaver": DaSiWa_MetadataImageSaver,
    "DaSiWa_MetadataImageSaverFull": DaSiWa_MetadataImageSaverFull,
    "DaSiWa_MetadataConfig": DaSiWa_MetadataConfig,
    "DaSiWa_CreateExtraMetadata": DaSiWa_CreateExtraMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_ResolutionScaleCalculator": "DaSiWa Resolution Scale Calculator",
    "DaSiWa_NodeStatusSwitch": "DaSiWa Node Status Switch",
    "DaSiWa_RTX_UpscalerRefiner": "DaSiWa RTX Upscaler & Refiner",
    "DaSiWa_MetadataImageSaver": "DaSiWa Metadata Image Saver",
    "DaSiWa_MetadataImageSaverFull": "DaSiWa Metadata Image Saver (Full)",
    "DaSiWa_MetadataConfig": "DaSiWa Metadata Config",
    "DaSiWa_CreateExtraMetadata": "DaSiWa Create Extra Metadata",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
