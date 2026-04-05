import math

class DaSiWa_ResolutionScaleCalculator:
    # --- DATA ARRAYS ---
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

    RESOLUTION_PRESETS = {
        "360p": 0.23,
        "480p": 0.38,
        "720p": 0.92,
        "1080p": 2.07,
        "1440p": 3.68,
        "2K": 4.19,
        "4K": 8.29,
    }

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
                "method": (["Use Precision Presets", "Use Resolution Presets"], {"default": "Use Precision Presets"}),
                "precision_presets": (list(cls.PRECISION_PRESETS.keys()), {"default": "0.52 MP - SD"}),
                "resolution_presets": (list(cls.RESOLUTION_PRESETS.keys()), {"default": "1080p"}),
                
                "no_scale": ("BOOLEAN", {"default": False, "label_on": "ON (Source Dims)", "label_off": "OFF (Calculated)"}),
                
                "scale_from_image": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
                "aspect_preset": (list(cls.ASPECT_PRESETS.keys()), {"default": "9:16 - Social"}),
                "swap_aspect": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "manual_aspect_width": ("INT", {"default": 16, "min": 1, "max": 8192}),
                "manual_aspect_height": ("INT", {"default": 9, "min": 1, "max": 8192}),
                "mode": (["Standard", "WAN/LTX (Div32)"], {"default": "WAN/LTX (Div32)"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width_int", "height_int", "width_float", "height_float")
    FUNCTION = "calculate"
    CATEGORY = "DaSiWa/Scaling"

    def calculate(self, method, precision_presets, resolution_presets, no_scale, scale_from_image, aspect_preset, swap_aspect, manual_aspect_width, manual_aspect_height, mode, image=None):
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
        if method == "Use Resolution Presets":
            a = self.RESOLUTION_PRESETS.get(resolution_presets, 2.07)
        else:
            a = self.PRECISION_PRESETS.get(precision_presets, 0.52)

        # 4. CALCULATE
        target_total_pixels = a * 1000000
        calc_w = math.sqrt(target_total_pixels * aspect_ratio)
        calc_h = math.sqrt(target_total_pixels / aspect_ratio)
        
        # 5. MODE HANDLING
        if mode == "WAN/LTX (Div32)":
            final_w = int(round(calc_w / 32.0) * 32)
            final_h = int(round(calc_h / 32.0) * 32)
        else:
            final_w = int(round(calc_w))
            final_h = int(round(calc_h))

        return (max(final_w, 32), max(final_h, 32), float(final_w), float(final_h))