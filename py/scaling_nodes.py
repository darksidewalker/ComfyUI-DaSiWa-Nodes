import math

# --- DA SI WA CONFIGURATION PANEL ---
PRECISION_TIERS = {
    "0.26 MP - Preview": 0.26,
    "0.36 MP - Small": 0.36,
    "0.52 MP - Standard": 0.52,
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

RES_PRESETS = {
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

class DaSiWa_ResolutionScaler:
    DESCRIPTION = """

    DaSiWa Resolution Scaler
    Calculates mathematically precise resolutions based on Megapixel area.
    
    - Standard Mode: Pure mathematical scaling.
    - WAN/LTX Mode: Snaps to 32-pixel boundaries for VAE compatibility.
    - Scale From Image: 
        - YES: Uses aspect ratio from the connected 'image' input.
        - NO: Uses 'Aspect Preset' or 'Manual Aspect' values.

    """

def calculate(self, method, pixel_precision, res_preset, scale_from_image, aspect_preset, swap_aspect, manual_aspect_width, manual_aspect_height, mode, image=None):
        # 1. ASPECT LOGIC & BATCH HANDLING
        if scale_from_image:
            if image is None:
                raise ValueError("DaSiWa Scaler: 'scale_from_image' is set to YES, but no image or video is connected to the input.")
            
            # ComfyUI Images/Videos are [Batch, Height, Width, Channels]
            # We explicitly grab only the first frame [0] to prevent batch-processing crashes
            try:
                batch_size, h, w, _ = image.shape
                source_w, source_h = float(w), float(h)
            except (AttributeError, IndexError, ValueError):
                raise ValueError("DaSiWa Scaler: Received an invalid image tensor format.")
        else:
            # MANUAL MODE
            if aspect_preset == "CUSTOM":
                source_w, source_h = float(manual_aspect_width), float(manual_aspect_height)
            else:
                source_w, source_h = ASPECT_PRESETS.get(aspect_preset, (1, 1))
            
            if swap_aspect:
                source_w, source_h = source_h, source_w
        
        aspect_ratio = source_w / source_h

        # 2. RESOLUTION LOGIC (NO-SCALE)
        if method == "No Scale (Source Dims)":
            final_w = int(source_w)
            final_h = int(source_h)
            return (final_w, final_h, float(final_w), float(final_h))

        # 3. GET TARGET MP
        if method == "Use Resolution Presets":
            a = RES_PRESETS.get(res_preset, 2.07)
        else:
            a = PRECISION_TIERS.get(pixel_precision, 0.52)

        # 4. MATH (DA SI WA AREA FORMULA)
        target_total_pixels = a * 1000000
        calc_w = math.sqrt(target_total_pixels * aspect_ratio)
        calc_h = math.sqrt(target_total_pixels / aspect_ratio)
        
        # 5. MODE HANDLING (Div32 vs Standard)
        if mode == "WAN/LTX (Div32)":
            final_w = int(round(calc_w / 32.0) * 32)
            final_h = int(round(calc_h / 32.0) * 32)
        else:
            # Standard Mode: nearest integer
            final_w = int(round(calc_w))
            final_h = int(round(calc_h))

        # Final safety clamp
        return (max(final_w, 32), max(final_h, 32), float(final_w), float(final_h))