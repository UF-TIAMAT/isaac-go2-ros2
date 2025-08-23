import numpy as np
import cv2

def save_depth_as_rgb(
    depth_m: np.ndarray,
    rgb_path: str = "depth_vis.png",
    raw16_path: str = "depth_raw.png",
    near: float | None = None,
    far: float | None = None,
    invert: bool = False,
    use_percentiles: bool = True,
):
    """
    depth_m: float32/float64 array in meters, shape (H, W)
    rgb_path: output path for colorized visualization (8-bit, RGB)
    raw16_path: output path for raw 16-bit depth (millimeters, lossless)
    near/far: visualization range in meters (if None, auto-compute)
    invert: if True, nearer = brighter; if False, farther = brighter
    use_percentiles: auto-range using robust 1–99% percentiles (ignores outliers)
    """
    d = depth_m.astype(np.float32)

    # Handle invalids
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- Save raw 16-bit ----------
    # Convert meters -> millimeters and clamp to uint16 range
    d_mm_u16 = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(raw16_path, d_mm_u16)

    # ---------- Build a viewable RGB ----------
    # Choose visualization range
    if near is None or far is None:
        if use_percentiles:
            # robust to a few extreme outliers
            p1, p99 = np.percentile(d[d > 0], (1, 99)) if np.any(d > 0) else (0.0, 1.0)
            near_ = p1 if near is None else near
            far_  = p99 if far  is None else far
        else:
            # fallback to min/max of valid pixels
            valid = d[d > 0]
            near_ = float(valid.min()) if near is None and valid.size else 0.0
            far_  = float(valid.max()) if far  is None and valid.size else 1.0
    else:
        near_, far_ = float(near), float(far)

    # Prevent degenerate range
    if not np.isfinite(near_) or not np.isfinite(far_) or far_ <= near_:
        near_, far_ = 0.0, 1.0

    # Normalize to [0,1] within [near,far]
    dn = (d - near_) / (far_ - near_)
    dn = np.clip(dn, 0.0, 1.0)

    # Optional invert: many people prefer near=bright
    if invert:
        dn = 1.0 - dn

    # Map to 8-bit for coloring
    d8 = (dn * 255.0).astype(np.uint8)

    # Apply a perceptual colormap (OpenCV expects BGR)
    color_bgr = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)  # or COLORMAP_TURBO, JET, etc.

    # Save as RGB
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(rgb_path, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))  # OpenCV writes BGR
    return {
        "near": near_,
        "far": far_,
        "rgb_out": rgb_path,
        "raw16_out": raw16_path
    }