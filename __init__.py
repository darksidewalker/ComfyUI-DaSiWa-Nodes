if __package__:
    from .nodes.nodes_scaling import DaSiWa_ResolutionScaleCalculator, DaSiWa_TorchResize
    from .nodes.nodes_status_switch import DaSiWa_NodeStatusSwitch
    from .nodes.nodes_rtx_upscaler_refiner import DaSiWa_RTX_UpscalerRefiner
    from .nodes.nodes_metadata import DaSiWa_MetadataImageSaver, DaSiWa_MetadataImageSaverFull, DaSiWa_MetadataConfig, DaSiWa_CreateExtraMetadata
    from .nodes.nodes_advanced_lora_loader import DaSiWa_AdvancedLoRALoader
    from .nodes.nodes_comfy_kitchen_attention import PathchComfyKitchenAttentionDaSiWa
    from .nodes.nodes_watermark import DaSiWa_Watermark
    from .nodes.nodes_random_string_picker import DaSiWa_RandomStringPicker
    from .nodes.nodes_seed_control import DaSiWa_SeedControl
    from .nodes.nodes_wildcard_preset_prompt_builder import DaSiWa_WildcardPresetPromptBuilder
    from .nodes.nodes_llm import DaSiWa_LLMModelSelector, DaSiWa_LLMAnalyze
    from .nodes.nodes_inpaint import DaSiWa_InpaintCropPrep, DaSiWa_InpaintComposite
    from .nodes.nodes_enhanced_video_combine import DaSiWa_EnhancedVideoCombine
    from .nodes.nodes_minimax_h3_director import MiniMaxH3Director
    from .nodes.nodes_minimax_h3_director_guide import MiniMaxH3DirectorGuide
    from .nodes.nodes_minimax_h3_cache import MiniMaxH3Cache
    from .nodes.nodes_minimax_h3_latent_upscaler import DaSiWa_MiniMaxH3LatentUpscaler

    from .nodes import nodes_system_monitor
    from .nodes import input_images  # registers /dasiwa/input-images route
    from .nodes import refmod_library
    refmod_library.register_routes(__import__("server").PromptServer.instance)
    from .nodes import lora_info  # registers /dasiwa/ltx2/lorainfo + /dasiwa/ltx2/loraimg
    from .nodes.helper_logging import log_startup_summary

    NODE_CLASS_MAPPINGS = {
        "DaSiWa_ResolutionScaleCalculator": DaSiWa_ResolutionScaleCalculator,
        "DaSiWa_TorchResize": DaSiWa_TorchResize,
        "DaSiWa_NodeStatusSwitch": DaSiWa_NodeStatusSwitch,
        "DaSiWa_RTX_UpscalerRefiner": DaSiWa_RTX_UpscalerRefiner,
        "DaSiWa_MetadataImageSaver": DaSiWa_MetadataImageSaver,
        "DaSiWa_MetadataImageSaverFull": DaSiWa_MetadataImageSaverFull,
        "DaSiWa_MetadataConfig": DaSiWa_MetadataConfig,
        "DaSiWa_CreateExtraMetadata": DaSiWa_CreateExtraMetadata,
        "DaSiWa_LTX2LoraLoader": DaSiWa_AdvancedLoRALoader,
        "PathchComfyKitchenAttentionDaSiWa": PathchComfyKitchenAttentionDaSiWa,
        "DaSiWa_Watermark": DaSiWa_Watermark,
        "DaSiWa_RandomStringPicker": DaSiWa_RandomStringPicker,
        "DaSiWa_SeedControl": DaSiWa_SeedControl,
        "DaSiWa_WildcardPresetPromptBuilder": DaSiWa_WildcardPresetPromptBuilder,
        "DaSiWa_LLMModelSelector": DaSiWa_LLMModelSelector,
        "DaSiWa_LLMAnalyze": DaSiWa_LLMAnalyze,
        "DaSiWa_EnhancedVideoCombine": DaSiWa_EnhancedVideoCombine,
        "DaSiWa_InpaintCropPrep": DaSiWa_InpaintCropPrep,
        "DaSiWa_InpaintComposite": DaSiWa_InpaintComposite,
        "MiniMaxH3Director": MiniMaxH3Director,
        "MiniMaxH3DirectorGuide": MiniMaxH3DirectorGuide,
        "MiniMaxH3Cache": MiniMaxH3Cache,
        "DaSiWa_MiniMaxH3LatentUpscaler": DaSiWa_MiniMaxH3LatentUpscaler,

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
        "DaSiWa_LTX2LoraLoader": "Advanced LoRA Loader",
        "PathchComfyKitchenAttentionDaSiWa": "Patch Comfy Kitchen Attention",
        "DaSiWa_Watermark": "DaSiWa Watermark Overlay",
        "DaSiWa_RandomStringPicker": "DaSiWa Random String Picker",
        "DaSiWa_SeedControl": "Seed Control",
        "DaSiWa_WildcardPresetPromptBuilder": "DaSiWa Wildcard & Preset Prompt Builder",
        "DaSiWa_LLMModelSelector": "DaSiWa LLM Model Selector",
        "DaSiWa_LLMAnalyze": "DaSiWa LLM Analyze",
        "DaSiWa_EnhancedVideoCombine": "DaSiWa Enhanced Video Combine",
        "DaSiWa_InpaintCropPrep": "DaSiWa Inpaint Crop Prep",
        "DaSiWa_InpaintComposite": "DaSiWa Inpaint Composite",
        "MiniMaxH3Director": "MiniMax H3 Director",
        "MiniMaxH3DirectorGuide": "MiniMax H3 Director Guide",
        "MiniMaxH3Cache": "MiniMax H3 Cache",
        "DaSiWa_MiniMaxH3LatentUpscaler": "Minimax H3 Latent Upscaler",

    }
    log_startup_summary(len(NODE_CLASS_MAPPINGS))
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
