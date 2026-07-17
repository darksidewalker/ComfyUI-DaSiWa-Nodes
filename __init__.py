if __package__:
    from .nodes.scaling_nodes import DaSiWa_ResolutionScaleCalculator, DaSiWa_TorchResize
    from .nodes.node_status_switch import DaSiWa_NodeStatusSwitch
    from .nodes.rtx_upscaler_refiner import DaSiWa_RTX_UpscalerRefiner
    from .nodes.metadata_nodes import DaSiWa_MetadataImageSaver, DaSiWa_MetadataImageSaverFull, DaSiWa_MetadataConfig, DaSiWa_CreateExtraMetadata
    from .nodes.ltx2_loader import DaSiWa_LTX2LoraLoader
    from .nodes.watermark_nodes import DaSiWa_Watermark
    from .nodes.random_string_picker import DaSiWa_RandomStringPicker
    from .nodes.llm_nodes import DaSiWa_LLMModelSelector, DaSiWa_LLMAnalyze
    from .nodes import system_monitor

    NODE_CLASS_MAPPINGS = {
        "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator,
        "DaSiWa_TorchResize": DaSiWa_TorchResize,
        "DaSiWa_NodeStatusSwitch": DaSiWa_NodeStatusSwitch,
        "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
        "DaSiWa_MetadataImageSaver": DaSiWa_MetadataImageSaver,
        "DaSiWa_MetadataImageSaverFull": DaSiWa_MetadataImageSaverFull,
        "DaSiWa_MetadataConfig": DaSiWa_MetadataConfig,
        "DaSiWa_CreateExtraMetadata": DaSiWa_CreateExtraMetadata,
        "DaSiWa_LTX2LoraLoader": DaSiWa_LTX2LoraLoader,
        "DaSiWa_Watermark": DaSiWa_Watermark,
        "DaSiWa_RandomStringPicker": DaSiWa_RandomStringPicker,
        "DaSiWa_LLMModelSelector": DaSiWa_LLMModelSelector,
        "DaSiWa_LLMAnalyze": DaSiWa_LLMAnalyze,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "DaSiWa_ResolutionScaleCalculator": "DaSiWa Resolution Scale Calculator",
        "DaSiWa_TorchResize": "DaSiWa Torch Resize",
        "DaSiWa_NodeStatusSwitch": "DaSiWa Node Status Switch",
        "DaSiWa_RTX_UpscalerRefiner": "DaSiWa RTX Upscaler & Refiner",
        "DaSiWa_MetadataImageSaver": "DaSiWa Metadata Image Saver",
        "DaSiWa_MetadataImageSaverFull": "DaSiWa Metadata Image Saver (Full)",
        "DaSiWa_MetadataConfig": "DaSiWa Metadata Config",
        "DaSiWa_CreateExtraMetadata": "DaSiWa Create Extra Metadata",
        "DaSiWa_LTX2LoraLoader": "DaSiWa LTX-2 LoRA Loader",
        "DaSiWa_Watermark": "DaSiWa Watermark Overlay",
        "DaSiWa_RandomStringPicker": "DaSiWa Random String Picker",
        "DaSiWa_LLMModelSelector": "DaSiWa LLM Model Selector",
        "DaSiWa_LLMAnalyze": "DaSiWa LLM Analyze",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
