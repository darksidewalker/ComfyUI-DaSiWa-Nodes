import math

class DaSiWa_ResolutionScaleCalculator:
    # --- DATA ARRAYS ---
    # Resolution presets: value = real_world_pixel_count / (1024*1024)
    # Matches ComfyUI native MP convention (1 MP = 1,048,576 pixels)
    RESOLUTION_PRESETS = {
        "144p": 0.0352,
        "240p": 0.0977,
        "360p": 0.22,
        "480p": 0.391,
        "540p": 0.494,
        "576p": 0.396,
        "720p": 0.879,
        "900p": 1.373,
        "1080p": 1.978,
        "1152p": 2.25,
        "1440p": 3.516,
        "2160p": 7.91,
        "2K": 3.906,
        "4K": 7.91,
    }

    PRECISION_PRESETS = {
        "0.26 MP - Preview": 0.26,
        "0.36 MP - Small": 0.36,
        "0.52 MP - SD": 0.52,
        "0.65 MP - Balanced": 0.65,
        "0.83 MP - HD": 0.83,
        "1.05 MP - HD+": 1.05,
        "1.20 MP - HD++": 1.20,
        "1.35 MP - 2K lite": 1.35,
        "1.55 MP - 2K": 1.55,
        "1.65 MP - 2K+": 1.65,
        "1.75 MP - QHD": 1.75,
        "2.10 MP - FHD": 2.10,
        "3.30 MP - QHD+": 3.30,
        "4.75 MP - 2K Pro": 4.75,
        "6.50 MP - Production": 6.50,
        "8.30 MP - UHD": 8.30,
    }

    PRESETS = {**RESOLUTION_PRESETS, **PRECISION_PRESETS}

    ASPECT_PRESETS = {
        "1:1 - Square": (1, 1),
        "2:3 - Classic": (2, 3),
        "3:4 - Photo": (3, 4),
        "5:8 - Tall": (5, 8),
        "9:16 - Social": (9, 16),
        "9:21 - Cinema": (9, 21),
        "CUSTOM": (0, 0),
    }

    DESCRIPTION = """
    DaSiWa Resolution Scale Calculator
    
    Calculates mathematically precise resolutions based on a target Megapixel area.
    
    - Standard Mode: Pure mathematical scaling.

    - WAN/LTX Mode: Snaps to 32-pixel boundaries (mandatory for WAN/LTX VAEs).

    - No Scale: Overrides all math and outputs the source image dimensions directly.
    
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_preset": (list(cls.PRESETS.keys()), {"default": "0.52 MP - SD", "description": "Target visual resolution / pixel budget. Pick either a standard resolution label or an optimized megapixel tier."}),
                
                "no_scale": ("BOOLEAN", {"default": False, "label_on": "ON (Source Dims)", "label_off": "OFF (Calculated)", "description": "Bypass all calculations and output the source dimensions exactly."}),
                
                "scale_from_image": ("BOOLEAN", {"default": True, "label_on": "IMAGE ASPECT", "label_off": "USE ASPECT BELOW", "description": "IMAGE ASPECT ignores the aspect controls below and uses the connected image shape. USE ASPECT BELOW uses the visible aspect controls."}),
                "aspect_preset_when_not_image": (list(cls.ASPECT_PRESETS.keys()), {"default": "9:16 - Social", "description": "Used only when scale_from_image is USE ASPECT BELOW. Ignored while IMAGE ASPECT is selected."}),
                "swap_aspect_when_not_image": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "description": "Used only when scale_from_image is USE ASPECT BELOW. Flip width and height."}),
                "custom_aspect_width": ("INT", {"default": 16, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio width, not final pixels."}),
                "custom_aspect_height": ("INT", {"default": 9, "min": 1, "max": 8192, "description": "Used only when scale_from_image is USE ASPECT BELOW and aspect preset is CUSTOM. Ratio height, not final pixels."}),
                "mode": (["Standard", "WAN/LTX (Div32)", "LTX 2-Stage (Div64)", "CUSTOM"], {"default": "WAN/LTX (Div32)", "description": "Snapping engine. Use WAN/LTX (Div32) for modern video models."}),
                "custom_divisor": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1, "description": "Custom pixel boundary snapping."}),
            },
            "optional": {
                "image": ("IMAGE", {"description": "The source image used to calculate the target aspect ratio."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width_int", "height_int", "width_float", "height_float")
    FUNCTION = "calculate"
    CATEGORY = "DaSiWa/Scaling"

    def calculate(
        self,
        resolution_preset=None,
        no_scale=False,
        scale_from_image=True,
        aspect_preset_when_not_image="9:16 - Social",
        swap_aspect_when_not_image=False,
        custom_aspect_width=16,
        custom_aspect_height=9,
        mode="WAN/LTX (Div32)",
        custom_divisor=8,
        image=None,
        method=None,
        preset=None,
        precision_presets=None,
        resolution_presets=None,
        aspect_preset=None,
        swap_aspect=None,
        manual_aspect_width=None,
        manual_aspect_height=None,
    ):
        if resolution_preset is None:
            resolution_preset = preset
        aspect_preset = aspect_preset if aspect_preset is not None else aspect_preset_when_not_image
        swap_aspect = swap_aspect if swap_aspect is not None else swap_aspect_when_not_image
        manual_aspect_width = manual_aspect_width if manual_aspect_width is not None else custom_aspect_width
        manual_aspect_height = manual_aspect_height if manual_aspect_height is not None else custom_aspect_height

        # 1. GET SOURCE DIMENSIONS (From Image or Manual)
        if scale_from_image:
            if image is None:
                raise ValueError("DaSiWa Scaler: 'scale_from_image' is set to YES, but no image is connected.")
            try:
                # Get shape from first frame
                _, h, w, _ = image.shape
                source_w, source_h = float(w), float(h)
            except Exception:
                raise ValueError("DaSiWa Scaler: Invalid image input format.")
        else:
            if aspect_preset == "CUSTOM":
                source_w, source_h = float(manual_aspect_width), float(manual_aspect_height)
            else:
                source_w, source_h = self.ASPECT_PRESETS.get(aspect_preset, (1, 1))
            
            if swap_aspect:
                source_w, source_h = source_h, source_w

        # 2. HANDLE NO-SCALE TOGGLE (PASS-THROUGH)
        if no_scale:
            final_w, final_h = int(source_w), int(source_h)
            return (final_w, final_h, float(final_w), float(final_h))

        # 3. GET MP TARGET
        aspect_ratio = source_w / source_h
        if resolution_preset is None:
            if method == "Use Resolution Presets":
                resolution_preset = resolution_presets
            else:
                resolution_preset = precision_presets
        a = self.PRESETS.get(resolution_preset, 0.52)

        # 4. CALCULATE
        # ComfyUI-native MP convention: 1 MP = 1024*1024 = 1,048,576 pixels
        # (matches Scale Image to Total Pixels node)
        target_total_pixels = a * 1024 * 1024
        calc_w = math.sqrt(target_total_pixels * aspect_ratio)
        calc_h = math.sqrt(target_total_pixels / aspect_ratio)
        
        # 5. MODE HANDLING
        if mode == "CUSTOM":
            d = max(1, int(custom_divisor))
            final_w = int(round(calc_w / d) * d)
            final_h = int(round(calc_h / d) * d)
            floor = d
        elif mode == "LTX 2-Stage (Div64)":
            final_w = int(round(calc_w / 64.0) * 64)
            final_h = int(round(calc_h / 64.0) * 64)
            floor = 64
        elif mode == "WAN/LTX (Div32)":
            final_w = int(round(calc_w / 32.0) * 32)
            final_h = int(round(calc_h / 32.0) * 32)
            floor = 32
        else:
            final_w = int(round(calc_w))
            final_h = int(round(calc_h))
            floor = 1

        return (max(final_w, floor), max(final_h, floor), float(final_w), float(final_h))
