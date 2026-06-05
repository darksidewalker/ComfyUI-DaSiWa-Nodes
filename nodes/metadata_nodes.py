import os
import json
import re
import numpy as np
import torch
from PIL import Image, PngImagePlugin
from datetime import datetime
import hashlib
import folder_paths

# Global cache for model hashes to avoid re-calculating large files
_MODEL_HASH_CACHE = {}

METADATA_INPUT_DESCRIPTIONS = {
    "save_workflow": "Include the ComfyUI workflow JSON so the saved image can be loaded back into ComfyUI.",
    "model_hash": "Optional model hash to embed in A1111/Civitai-style metadata. Leave empty to auto-detect from the model file when possible.",
    "node_positive": "Connect the positive conditioning or prompt path so the saver can auto-detect the positive prompt.",
    "node_negative": "Connect the negative conditioning or prompt path so the saver can auto-detect the negative prompt.",
    "node_model": "Connect the model path so the saver can auto-detect checkpoint or diffusion model name and hash.",
    "node_latent": "Connect the latent/sampler path so the saver can trace steps, CFG, sampler, scheduler, and seed.",
    "node_noise": "Optional noise source used to auto-detect seed when it is separate from the sampler path.",
    "node_sigmas": "Optional sigmas/scheduler source used to auto-detect steps and scheduler when they are separate.",
    "node_sampler": "Optional sampler source used to auto-detect sampler name when it is separate from the latent path.",
    "extra_metadata": "Optional custom key/value metadata bundle created by DaSiWa Create Extra Metadata.",
    "text_positive": "Manual positive prompt override. Connect a string to force this exact prompt into saved metadata.",
    "text_negative": "Manual negative prompt override. Connect a string to force this exact negative prompt into saved metadata.",
    "text_steps": "Manual steps override. Use 0 to prefer auto-detected steps when available.",
    "text_cfg": "Manual CFG scale override. Use 0.0 to prefer auto-detected CFG when available.",
    "text_sampler": "Manual sampler override. Leave blank to prefer auto-detected sampler when available.",
    "text_scheduler": "Manual scheduler override. Leave blank to prefer auto-detected scheduler when available.",
    "text_seed": "Manual seed override. Use 0 to prefer auto-detected seed when available.",
    "text_model": "Manual model filename override. Leave blank to prefer auto-detected model when available.",
    "save_output": "When connected, overrides whether files are saved to the output folder or temp preview folder.",
}

class DaSiWa_MetadataConfig:
    """
    Companion node for DaSiWa_MetadataImageSaver.
    Gathers all metadata-related inputs into a single 'config' bundle
    to declutter the main saver node.
    """
    DESCRIPTION = (
        "DaSiWa Metadata Config: collects workflow metadata settings and optional "
        "auto-detection links into one compact config output for the image saver."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_workflow": ("BOOLEAN", {"default": True, "description": METADATA_INPUT_DESCRIPTIONS["save_workflow"]}),
                "model_hash": ("STRING", {"default": "", "placeholder": "Optional: Civitai uses this to link models", "description": METADATA_INPUT_DESCRIPTIONS["model_hash"]}),
            },
            "optional": {
                "node_positive": ("CONDITIONING", {"description": METADATA_INPUT_DESCRIPTIONS["node_positive"]}),
                "node_negative": ("CONDITIONING", {"description": METADATA_INPUT_DESCRIPTIONS["node_negative"]}),
                "node_model": ("MODEL", {"description": METADATA_INPUT_DESCRIPTIONS["node_model"]}),
                "node_latent": ("LATENT", {"description": METADATA_INPUT_DESCRIPTIONS["node_latent"]}),
                "node_noise": ("NOISE", {"description": METADATA_INPUT_DESCRIPTIONS["node_noise"]}),
                "node_sigmas": ("SIGMAS", {"description": METADATA_INPUT_DESCRIPTIONS["node_sigmas"]}),
                "node_sampler": ("SAMPLER", {"description": METADATA_INPUT_DESCRIPTIONS["node_sampler"]}),
                "extra_metadata": ("EXTRA_METADATA", {"description": METADATA_INPUT_DESCRIPTIONS["extra_metadata"]}),
                "text_positive": ("STRING", {"multiline": True, "placeholder": "Positive Prompt (Manual Override)", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_positive"]}),
                "text_negative": ("STRING", {"multiline": True, "placeholder": "Negative Prompt (Manual Override)", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_negative"]}),
                "text_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_steps"]}),
                "text_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_cfg"]}),
                "text_sampler": (["", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "ddim", "uni_pc"], {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_sampler"]}),
                "text_scheduler": (["", "normal", "karras", "exponential", "simple", "ddim_uniform"], {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_scheduler"]}),
                "text_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_seed"]}),
                "text_model": ("STRING", {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_model"]}),
                "save_output": ("BOOLEAN", {"forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["save_output"]}),
            }
        }

    RETURN_TYPES = ("METADATA_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "gather"
    CATEGORY = "DaSiWa/IO"

    def gather(self, **kwargs):
        return (kwargs,)

class DaSiWa_MetadataImageSaver:
    """
    Compact version of the Metadata Image Saver.
    Designed to be used with DaSiWa_MetadataConfig to keep the workflow clean.
    """
    DESCRIPTION = (
        "DaSiWa Metadata Image Saver: saves PNG or WebP images with ComfyUI "
        "workflow data and A1111/Civitai-compatible generation metadata."
    )

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"description": "The image(s) to save."}),
                "filename_prefix": ("STRING", {"default": "DaSiWa_%date:yyyyMMdd%_%seed%", "description": "The name of the file. Placeholders: %seed%, %model%, %width%, %height%, %date%."}),
                "file_format": (["webp", "png"], {"default": "webp", "description": "WebP (modern/small) or PNG (lossless/high compatibility)."}),
                "compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "description": "0=Best Quality, 100=Max Compression. PNG maps 0-100 to level 0-9. WebP maps 0-100 to quality 100-0."}),
                "save_output": ("BOOLEAN", {"default": True, "description": "If False, images are saved to the temp folder for preview only."}),
            },
            "optional": {
                "metadata_config": ("METADATA_CONFIG", {"description": "Config bundle from DaSiWa Metadata Config. Values in this config override or fill saver metadata settings."}),
                "extra_metadata": ("EXTRA_METADATA", {"description": METADATA_INPUT_DESCRIPTIONS["extra_metadata"]}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename", "metadata")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "DaSiWa/IO"

    def format_filename(self, prefix, seed, model_name, width, height):
        now = datetime.now()
        # Handle %date:format%
        def replace_date(match):
            fmt = match.group(1) if match.group(1) else "yyyyMMdd_HHmmss"
            # Map common tokens to strftime
            fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y").replace("MM", "%m").replace("dd", "%d").replace("DD", "%d")
            fmt = fmt.replace("HH", "%H").replace("hh", "%H").replace("mm", "%M").replace("ss", "%S")
            return now.strftime(fmt)

        prefix = re.sub(r"%date(?::([^%]+))?%", replace_date, prefix)
        prefix = prefix.replace("%seed%", str(seed))
        prefix = prefix.replace("%model%", os.path.splitext(os.path.basename(model_name))[0])
        prefix = prefix.replace("%width%", str(width))
        prefix = prefix.replace("%height%", str(height))
        return prefix

    def get_sha256(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        cache_key = (file_path, stat.st_mtime, stat.st_size)
        
        if cache_key in _MODEL_HASH_CACHE:
            return _MODEL_HASH_CACHE[cache_key]
            
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        h = hash_sha256.hexdigest()
        _MODEL_HASH_CACHE[cache_key] = h
        return h

    def resolve_value(self, workflow_prompt, value, search_keys=None):
        """Recursively follows links in the workflow graph to find literal values."""
        if isinstance(value, list) and len(value) == 2:
            node_id = str(value[0])
            node = workflow_prompt.get(node_id)
            
            # Handle nested/grouped node IDs (e.g., "290:325")
            if not node and ":" in node_id:
                node = workflow_prompt.get(node_id.split(":")[0])

            if node:
                inputs = node.get("inputs", {})
                
                # 1. Follow links through Reroute nodes
                if node.get("class_type") in ["Reroute", "NodeStyles_Reroute", "Reroute (rgthree)"]:
                    for v in inputs.values():
                        if isinstance(v, list) and len(v) == 2:
                            return self.resolve_value(workflow_prompt, v, search_keys=search_keys)

                # 2. Prioritize search_keys if provided (e.g., look for 'steps' in a Scheduler node)
                if search_keys:
                    keys = search_keys if isinstance(search_keys, list) else [search_keys]
                    for k in keys:
                        if k in inputs:
                            res = inputs[k]
                            # If it's a link, follow it. If it's a value, return it.
                            if isinstance(res, list) and len(res) == 2:
                                return self.resolve_value(workflow_prompt, res, search_keys=keys)
                            return res

                # 3. Fallback: Check for generic value keys ONLY
                # We do not want to fallback to 'steps' if we are looking for 'sampler'.
                generic_fallbacks = ["text", "string", "value", "val", "ckpt_name", "unet_name", "vae_name", "stop_at_clip_layer", "lora_name", "lora", "lora_model", "model_name", "sampler_name", "scheduler"]
                for k in generic_fallbacks:
                    if k in inputs:
                        res = inputs[k]
                        if isinstance(res, list) and len(res) == 2:
                            return self.resolve_value(workflow_prompt, res, search_keys)
                        return res
                
                # 4. Last resort fallback: if there is only one input and it's not a list, it's likely our value
                # (Common for simple "String" or "Int" custom nodes)
                non_list_inputs = [v for v in inputs.values() if not isinstance(v, list)]
                if len(non_list_inputs) == 1:
                    return non_list_inputs[0]

        return value

    def save_images(self, images, filename_prefix, file_format, compression, save_output,
                    node_positive=None, node_negative=None, node_model=None, node_latent=None,
                    node_noise=None, node_sigmas=None, node_sampler=None,
                    extra_metadata=None,
                    metadata_config=None,
                    text_positive="", text_negative="",
                    text_steps=0, text_cfg=0.0, text_sampler="",
                    text_scheduler="", text_seed=0, text_model="",
                    prompt=None, extra_pnginfo=None, **kwargs):

        # Determine output directory and type based on save_output toggle
        output_dir = self.output_dir if save_output else folder_paths.get_temp_directory()
        current_type = "output" if save_output else "temp"

        # Initialize metadata flags with defaults
        # "if no config node is connected, it should just save the workflow"
        save_workflow = kwargs.get("save_workflow", True)
        model_hash = kwargs.get("model_hash", "")

        # If metadata_config is provided, use its values to override or fill defaults
        if metadata_config is not None:
            save_workflow = metadata_config.get("save_workflow", save_workflow)
            model_hash = metadata_config.get("model_hash", model_hash)
            
            # These might be in config if using the Full/Legacy setup or custom routing
            filename_prefix = metadata_config.get("filename_prefix", filename_prefix)
            file_format = metadata_config.get("file_format", file_format)
            compression = metadata_config.get("compression", compression)
            save_output = metadata_config.get("save_output", save_output)

            # Re-evaluate output directory if overridden by config
            output_dir = self.output_dir if save_output else folder_paths.get_temp_directory()
            current_type = "output" if save_output else "temp"

            node_positive = metadata_config.get("node_positive", node_positive)
            node_negative = metadata_config.get("node_negative", node_negative)
            node_model = metadata_config.get("node_model", node_model)
            node_latent = metadata_config.get("node_latent", node_latent)
            node_noise = metadata_config.get("node_noise", node_noise)
            node_sigmas = metadata_config.get("node_sigmas", node_sigmas)
            node_sampler = metadata_config.get("node_sampler", node_sampler)
            extra_metadata = metadata_config.get("extra_metadata", extra_metadata)
            text_positive = metadata_config.get("text_positive", text_positive)
            text_negative = metadata_config.get("text_negative", text_negative)
            text_steps = metadata_config.get("text_steps", text_steps)
            text_cfg = metadata_config.get("text_cfg", text_cfg)
            text_sampler = metadata_config.get("text_sampler", text_sampler)
            text_scheduler = metadata_config.get("text_scheduler", text_scheduler)
            text_seed = metadata_config.get("text_seed", text_seed)
            text_model = metadata_config.get("text_model", text_model)

        filename_prefix += self.prefix_append
        
        # Automatically detect LoRAs used in the workflow
        lora_strings = []
        current_node_id = None
        detected_vae_name = ""
        detected_clip_skip = 0
        if prompt is not None:
            # Find the current node's ID for tracing
            if extra_pnginfo is not None:
                current_node_id = extra_pnginfo.get("output_node_id")

            if current_node_id is None: # Fallback for older ComfyUI or if not set
                for nid, ninfo in prompt.items():
                    if ninfo and ninfo.get("class_type") in ["DaSiWa_MetadataImageSaver", "DaSiWa_MetadataImageSaverFull"]:
                        current_node_id = nid
                        break

            for node_id in prompt:
                node = prompt[node_id]
                if not node: continue
                class_type = node.get("class_type", "")
                inputs = node.get("inputs", {})

                # --- 1. VAE Detection ---
                if class_type == "VAELoader":
                    v_name = self.resolve_value(prompt, inputs.get("vae_name", ""))
                    if v_name and isinstance(v_name, str) and v_name != "None":
                        detected_vae_name = v_name
                
                # --- 2. Clip Skip Detection ---
                if class_type == "CLIPSetLastLayer":
                    c_skip = self.resolve_value(prompt, inputs.get("stop_at_clip_layer", 0))
                    try:
                        detected_clip_skip = abs(int(c_skip))
                    except:
                        pass
                
                # --- 3. LoRA Detection ---
                # Aggressive detection: scan all nodes for inputs that look like LoRA filenames
                if "Checkpoint" not in class_type:
                    for k, v in inputs.items():
                        k_lower = k.lower()
                        if ("lora" in k_lower and ("name" in k_lower or "model" in k_lower or k_lower.split('_')[-1].isdigit())):
                            l_name = self.resolve_value(prompt, v)
                            if l_name and isinstance(l_name, str) and l_name != "None" and any(l_name.lower().endswith(ext) for ext in [".safetensors", ".pt", ".ckpt"]):
                                # Determine strength key based on the name key (e.g. lora_1 -> lora_1_strength)
                                base_key = k.replace("_name", "").replace("_model", "")
                                strength_key = next((sk for sk in [f"{base_key}_strength", f"{k}_strength", "strength", "strength_model", "model_strength"] if sk in inputs), None)
                                
                                l_strength = 1.0
                                if strength_key:
                                    l_strength = self.resolve_value(prompt, inputs[strength_key])
                                
                                # Check state/toggle for multi-slot loaders
                                l_state = self.resolve_value(prompt, inputs.get(f"{base_key}_state", "on"))
                                if str(l_state).lower() in ["on", "true", "1", "enabled"] or l_state is True:
                                    clean_name = os.path.splitext(os.path.basename(l_name))[0]
                                    entry = f"<lora:{clean_name}:{l_strength}>"
                                    if entry not in lora_strings:
                                        lora_strings.append(entry)

        results = list()
        saved_paths = []
        all_metadata = []

        # --- Auto-detect parameters from workflow graph ---
        # Initialize with empty/default values
        detected_positive_prompt = ""
        detected_negative_prompt = ""
        detected_steps = 0
        detected_cfg = 0.0
        detected_sampler_name = ""
        detected_scheduler = ""
        detected_seed = 0
        detected_model_name = ""

        if prompt is not None and current_node_id is not None:
            # Helper to find the source node's data connected to a specific input of the current node
            def get_source_node_data(target_input_name, current_node_id, workflow_prompt, config_node_id=None):
                # Check primary node first, then check config node if target not found there
                node_data = workflow_prompt.get(str(current_node_id))
                input_link_info = None
                if node_data:
                    input_link_info = node_data.get("inputs", {}).get(target_input_name)
                
                # If not connected to primary, check config node
                if not input_link_info and config_node_id:
                    cfg_node_data = workflow_prompt.get(str(config_node_id))
                    if cfg_node_data:
                        input_link_info = cfg_node_data.get("inputs", {}).get(target_input_name)
                
                if not input_link_info:
                    return None
                    
                # Follow links through Reroutes to find the actual source node
                while isinstance(input_link_info, list) and len(input_link_info) == 2:
                    source_node_id = str(input_link_info[0])
                    source_node = workflow_prompt.get(source_node_id)
                    if not source_node:
                        break
                    
                    if source_node.get("class_type") == "Reroute":
                        reroute_inputs = source_node.get("inputs", {})
                        found_next = False
                        for v in reroute_inputs.values():
                            if isinstance(v, list) and len(v) == 2:
                                input_link_info = v
                                found_next = True
                                break
                        if not found_next:
                            return source_node
                    else:
                        return source_node
                return None

            # Find the ID of the config node if connected
            config_node_id = None
            node_data = prompt.get(str(current_node_id))
            if node_data:
                link = node_data.get("inputs", {}).get("metadata_config")
                if isinstance(link, list):
                    config_node_id = str(link[0])

            # --- Positive Conditioning ---
            pos_cond_source_node = get_source_node_data("node_positive", current_node_id, prompt, config_node_id)
            if pos_cond_source_node:
                inputs = pos_cond_source_node.get("inputs", {})
                raw_text = inputs.get("text") or inputs.get("string") or inputs.get("value")
                if raw_text is not None:
                    detected_positive_prompt = self.resolve_value(prompt, raw_text)

            # --- Negative Conditioning ---
            neg_cond_source_node = get_source_node_data("node_negative", current_node_id, prompt, config_node_id)
            if neg_cond_source_node:
                inputs = neg_cond_source_node.get("inputs", {})
                raw_text = inputs.get("text") or inputs.get("string") or inputs.get("value")
                if raw_text is not None:
                    detected_negative_prompt = self.resolve_value(prompt, raw_text)

            # --- Model ---
            model_source_node = get_source_node_data("node_model", current_node_id, prompt, config_node_id)
            if model_source_node:
                inputs = model_source_node.get("inputs", {})
                # Support CheckpointLoaderSimple (ckpt_name) and UNETLoader (unet_name)
                ckpt_name = inputs.get("ckpt_name") or inputs.get("unet_name") or inputs.get("model_name") or inputs.get("checkpoint")
                if ckpt_name:
                    detected_model_name = self.resolve_value(prompt, ckpt_name)

            # --- Sampler Parameters (from KSampler) ---
            latent_source_node = get_source_node_data("node_latent", current_node_id, prompt, config_node_id)
            if latent_source_node:
                sampler_inputs = latent_source_node.get("inputs", {})
                
                # Aggressive search for sampler keys to support custom/advanced samplers
                def get_val(keys, default):
                    for k in keys:
                        if k in sampler_inputs:
                            return self.resolve_value(prompt, sampler_inputs[k], search_keys=keys)
                    return default

                detected_steps = get_val(["steps", "num_steps", "iterations"], 0)
                detected_cfg = get_val(["cfg", "cfg_scale", "scale"], 0.0)
                detected_sampler_name = get_val(["sampler_name", "sampler"], "")
                detected_scheduler = get_val(["scheduler", "scheduler_name"], "")
                detected_seed = get_val(["seed", "noise_seed", "random_seed"], 0)

        # --- Final values (auto-detected or manual override) ---
        def finalize(manual, input_name, detected, default, cast_type=str, search_keys=None, config_node_id=None):
            val = None
            # 1. Try to resolve directly from the workflow graph link
            nodes_to_check = [str(current_node_id)]
            if config_node_id:
                nodes_to_check.append(str(config_node_id))

            for nid in nodes_to_check:
                if prompt is not None and nid is not None:
                    node_data = prompt.get(nid)
                    if node_data:
                        link_info = node_data.get("inputs", {}).get(input_name)
                    
                        # Redirect to dedicated noise/sigmas slots if manual input is not connected
                        if not isinstance(link_info, list):
                            curr_inputs = node_data.get("inputs", {})
                            if input_name == "text_seed" and isinstance(curr_inputs.get("node_noise"), list):
                                link_info = curr_inputs["node_noise"]
                            elif input_name in ["text_steps", "text_scheduler"] and isinstance(curr_inputs.get("node_sigmas"), list):
                                link_info = curr_inputs["node_sigmas"]
                            elif input_name == "text_sampler" and isinstance(curr_inputs.get("node_sampler"), list):
                                link_info = curr_inputs["node_sampler"]

                        if isinstance(link_info, list):
                            val = self.resolve_value(prompt, link_info, search_keys=search_keys)
                            if val is not None: break
            
            # Fallback to metadata_config if still not found (handles widget constants on config node)
            if val is None and metadata_config:
                val = metadata_config.get(input_name)
            
            # 2. Fallback to the value passed to the function (manual override widgets)
            # Filter out booleans and empty strings for text-based fields
            if val in [None, ""] or isinstance(val, (list, dict, torch.Tensor, bool)):
                if manual is not None and not isinstance(manual, (list, dict, torch.Tensor)):
                    # Boolean check: prevent 'False' strings in prompts
                    if isinstance(manual, bool):
                        if cast_type is not bool:
                            val = None
                        else:
                            val = manual
                    elif manual != "" or (cast_type is int and manual is not None):
                        val = manual
            
            # 3. Last fallback to detected or default if still invalid
            if val in [None, ""] or isinstance(val, (list, dict, torch.Tensor, bool)):
                if detected not in [None, ""] and not isinstance(detected, (list, dict, torch.Tensor, bool)):
                    val = detected
                else:
                    val = default
            
            try:
                if cast_type is int and isinstance(val, float): return int(val)
                if cast_type is str and str(val).lower() in ["true", "false", "none"]:
                    return default
                if val is None: return default
                return cast_type(val)
            except:
                return default

        positive_prompt_final = finalize(text_positive, "text_positive", detected_positive_prompt, "positive_prompt_not_found", search_keys=["text", "string", "value"], config_node_id=config_node_id)
        negative_prompt_final = finalize(text_negative, "text_negative", detected_negative_prompt, "negative_prompt_not_found", search_keys=["text", "string", "value"], config_node_id=config_node_id)
        steps_final = finalize(text_steps, "text_steps", detected_steps, "steps_not_found", int, search_keys=["steps", "num_steps", "iterations"], config_node_id=config_node_id)
        cfg_final = finalize(text_cfg, "text_cfg", detected_cfg, "cfg_not_found", float, search_keys=["cfg", "cfg_scale", "scale"], config_node_id=config_node_id)
        sampler_name_final = finalize(text_sampler, "text_sampler", detected_sampler_name, "sampler_not_found", search_keys=["sampler_name", "sampler"], config_node_id=config_node_id)
        scheduler_final = finalize(text_scheduler, "text_scheduler", detected_scheduler, "scheduler_not_found", search_keys=["scheduler", "scheduler_name"], config_node_id=config_node_id)
        seed_final = finalize(text_seed, "text_seed", detected_seed, "seed_not_found", int, search_keys=["seed", "noise_seed", "random_seed"], config_node_id=config_node_id)
        model_name_final = finalize(text_model, "text_model", detected_model_name, "model_not_found.safetensors", search_keys=["ckpt_name", "model_name", "checkpoint"], config_node_id=config_node_id)

        # Auto-calculate model hash if not provided
        if not model_hash and model_name_final != "unknown.safetensors":
            full_path = folder_paths.get_full_path("checkpoints", model_name_final)
            if not full_path: # Check diffusion_models (Flux/UNET)
                full_path = folder_paths.get_full_path("diffusion_models", model_name_final)
            
            if full_path:
                full_hash = self.get_sha256(full_path)
                if full_hash:
                    # Civitai and A1111 use the first 10 characters for 'Model hash' usually, 
                    # but they accept the full SHA256 in metadata.
                    model_hash = full_hash[:10]

        # Append LoRAs to the final positive prompt
        final_positive_with_loras = positive_prompt_final
        if lora_strings:
            final_positive_with_loras += ", " + ", ".join(lora_strings)

        # Construct A1111-style parameters string
        base_params = f"{final_positive_with_loras}\nNegative prompt: {negative_prompt_final}\n"
        base_params += f"Steps: {steps_final}, Sampler: {sampler_name_final}{' ' + scheduler_final if scheduler_final != 'normal' else ''}, "
        base_params += f"CFG scale: {cfg_final}, Seed: {seed_final}, "
        
        if detected_clip_skip > 0:
            base_params += f"Clip skip: {detected_clip_skip}, "
            
        if detected_vae_name:
            base_params += f"VAE: {os.path.basename(detected_vae_name)}, "
        
        if model_hash:
            base_params += f"Model hash: {model_hash}, "
        
        for index, image in enumerate(images):
            # Convert tensor to PIL
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Finalize string with per-image dimensions
            params = base_params + f"Size: {img.width}x{img.height}, Model: {model_name_final.replace(',', '')}"
            all_metadata.append(params)
            
            # Add batch info if needed
            if len(images) > 1:
                params += f", Batch index: {index}, Batch size: {len(images)}"

            # Add extra metadata if provided
            if extra_metadata:
                for k, v in extra_metadata.items():
                    if k and v:
                        params += f", {k}: {str(v).replace(',', '/')}"

            # Dynamic Filename
            current_prefix = self.format_filename(filename_prefix, seed_final, model_name_final, img.width, img.height)
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(current_prefix, output_dir, img.width, img.height)

            metadata = PngImagePlugin.PngInfo()
            
            # 1. Standard ComfyUI Workflow (so the image is still 'loadable' in ComfyUI)
            if save_workflow:
                try:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                except Exception as e:
                    print(f"[DaSiWa] Warning: Failed to embed workflow metadata: {e}")
            
            # 2. Civitai/A1111 Compatibility Field
            metadata.add_text("parameters", params)

            file = f"{filename}_{counter:05}_.{file_format}"
            file_path = os.path.join(full_output_folder, file)
            
            if file_format == "webp":
                # WebP Quality: 100 is best (least compression)
                webp_quality = max(0, min(100, 100 - compression))
                
                # Create EXIF data for ComfyUI and A1111 compatibility
                exif = img.getexif()
                
                # Civitai / A1111 compatibility (Parameters in ImageDescription and UserComment)
                exif[0x010e] = params # Tag 270
                exif[0x9286] = b"ASCII\0\0\0" + params.encode("utf-8") # Tag 37510

                # ComfyUI "Drop-In" workflow support for WebP (Matching native core logic)
                if save_workflow:
                    try:
                        if prompt is not None:
                            exif[272] = f"prompt:{json.dumps(prompt)}" # 0x0110 (Model tag)
                        if extra_pnginfo is not None:
                            # Ensure 'workflow' is specifically in Tag 271 (Make) for UI restoration
                            if "workflow" in extra_pnginfo:
                                exif[271] = f"workflow:{json.dumps(extra_pnginfo['workflow'])}"
                            
                            # Store other metadata in tags starting at 269 to avoid clobbering 270 (Params) or 271/272
                            other_exif_tag = 269
                            for x in extra_pnginfo:
                                if x == "workflow": continue
                                # Safeguard: don't overwrite tags we explicitly set
                                if other_exif_tag == 270: other_exif_tag -= 1 
                                exif[other_exif_tag] = f"{x}:{json.dumps(extra_pnginfo[x])}"
                                other_exif_tag -= 1
                    except Exception as e:
                        print(f"[DaSiWa] Warning: Failed to embed WebP workflow EXIF: {e}")

                img.save(file_path, quality=webp_quality, lossless=(webp_quality == 100), exif=exif)
            else:
                # PNG Compression: 0 is fastest, 9 is smallest.
                # Map 0-100 to 0-9. 100 compression = level 9.
                png_level = max(0, min(9, int(compression / 11)))
                img.save(file_path, pnginfo=metadata, compress_level=png_level)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": current_type
            })
            saved_paths.append(file_path)
            counter += 1

        return { "ui": { "images": results }, "result": (", ".join(saved_paths), "\n---\n".join(all_metadata)) }

class DaSiWa_MetadataImageSaverFull(DaSiWa_MetadataImageSaver):
    """
    Full version of the Metadata Image Saver with all ports exposed.
    """
    DESCRIPTION = (
        "DaSiWa Metadata Image Saver Full: advanced saver variant with every "
        "metadata detection and manual override port exposed directly on the node."
    )

    @classmethod
    def INPUT_TYPES(s):
        types = DaSiWa_MetadataImageSaver.INPUT_TYPES()
        types["optional"].update({
            "save_workflow": ("BOOLEAN", {"default": True, "description": METADATA_INPUT_DESCRIPTIONS["save_workflow"]}),
            "model_hash": ("STRING", {"default": "", "description": METADATA_INPUT_DESCRIPTIONS["model_hash"]}),
            "node_positive": ("CONDITIONING", {"description": METADATA_INPUT_DESCRIPTIONS["node_positive"]}),
            "node_negative": ("CONDITIONING", {"description": METADATA_INPUT_DESCRIPTIONS["node_negative"]}),
            "node_model": ("MODEL", {"description": METADATA_INPUT_DESCRIPTIONS["node_model"]}),
            "node_latent": ("LATENT", {"description": METADATA_INPUT_DESCRIPTIONS["node_latent"]}),
            "node_noise": ("NOISE", {"description": METADATA_INPUT_DESCRIPTIONS["node_noise"]}),
            "node_sigmas": ("SIGMAS", {"description": METADATA_INPUT_DESCRIPTIONS["node_sigmas"]}),
            "node_sampler": ("SAMPLER", {"description": METADATA_INPUT_DESCRIPTIONS["node_sampler"]}),
            "text_positive": ("STRING", {"multiline": True, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_positive"]}),
            "text_negative": ("STRING", {"multiline": True, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_negative"]}),
            "text_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_steps"]}),
            "text_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_cfg"]}),
            "text_sampler": (["", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "ddim", "uni_pc"], {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_sampler"]}),
            "text_scheduler": (["", "normal", "karras", "exponential", "simple", "ddim_uniform"], {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_scheduler"]}),
            "text_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_seed"]}),
            "text_model": ("STRING", {"default": "", "forceInput": True, "description": METADATA_INPUT_DESCRIPTIONS["text_model"]}),
        })
        return types

    CATEGORY = "DaSiWa/IO/Advanced"

class DaSiWa_CreateExtraMetadata:
    """
    A helper node to inject custom metadata keys into the save node.
    """
    DESCRIPTION = "DaSiWa Create Extra Metadata: creates or extends a custom metadata key/value bundle for the image saver."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"default": "CustomField", "description": "The label for the custom metadata field."}),
                "value": ("STRING", {"default": "", "description": "The text content for the custom metadata field."}),
            },
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {"description": "Link another metadata node to chain multiple custom fields."}),
            }
        }
    RETURN_TYPES = ("EXTRA_METADATA",)
    FUNCTION = "add_metadata"
    CATEGORY = "DaSiWa/IO"

    def add_metadata(self, key, value, extra_metadata=None):
        data = extra_metadata if extra_metadata is not None else {}
        data[key] = value
        return (data,)

NODE_CLASS_MAPPINGS = {
    "DaSiWa_MetadataImageSaver": DaSiWa_MetadataImageSaver,
    "DaSiWa_MetadataConfig": DaSiWa_MetadataConfig,
    "DaSiWa_CreateExtraMetadata": DaSiWa_CreateExtraMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaSiWa_MetadataImageSaver": "DaSiWa Metadata Image Saver",
    "DaSiWa_MetadataConfig": "DaSiWa Metadata Config",
    "DaSiWa_CreateExtraMetadata": "DaSiWa Create Extra Metadata",
}
