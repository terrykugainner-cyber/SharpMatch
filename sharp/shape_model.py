import cv2, numpy as np
from .gradients import gradient_mag_dir

def create(img, roi, edge_threshold=(60,140), sampling_step=2, use_polarity=False, max_points=4000):
    """Create an oriented edge-based shape model.
    Args:
        img: gray/BGR image
        roi: (x,y,w,h) rectangle in image
        edge_threshold: (low, high) for Canny
        sampling_step: pick 1 of N edge pixels (N=sampling_step) to reduce points
        use_polarity: if True, orientation uses gradient sign; else direction up to pi (ignore polarity)
        max_points: clamp points for speed
    Return:
        dict model with fields:
            image_shape, roi, points (Nx2 float32 in template coords),
            dirs (N float32 radians in (-pi, pi]), weights (N float32),
            edge_threshold, use_polarity
            template (cropped image for preview/debug)
    """
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    x,y,w,h = roi
    tpl = g[y:y+h, x:x+w].copy()

    # edges & gradients in ROI
    low, high = edge_threshold
    edges = cv2.Canny(tpl, low, high, apertureSize=3, L2gradient=True)
    mag, ang = gradient_mag_dir(tpl)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        raise ValueError("No edges found in ROI; adjust thresholds/ROI.")
    # sampling
    idx = np.arange(len(xs))[::max(1, int(sampling_step))]
    xs = xs[idx].astype(np.float32)
    ys = ys[idx].astype(np.float32)

    # directions
    dirs = ang[ys.astype(int), xs.astype(int)].astype(np.float32)
    if not use_polarity:
        # wrap direction modulo pi to ignore polarity
        dirs = ((dirs + np.pi) % np.pi) - (np.pi/2)  # still centered, equivalently reduce ambiguity

    # weights proportional to gradient magnitude (normalized)
    wts = mag[ys.astype(int), xs.astype(int)].astype(np.float32)
    if wts.max() > 0:
        wts = wts / (wts.max() + 1e-8)
    else:
        wts = np.ones_like(wts, np.float32)

    # clamp
    if len(xs) > max_points:
        sel = np.linspace(0, len(xs)-1, max_points).astype(int)
        xs, ys, dirs, wts = xs[sel], ys[sel], dirs[sel], wts[sel]

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    model = {
        "image_shape": tuple(g.shape[:2]),
        "roi": tuple(roi),
        "points": pts,
        "dirs": dirs,
        "weights": wts,
        "edge_threshold": tuple(edge_threshold),
        "use_polarity": bool(use_polarity),
        "template": tpl,
        "version": "0.1"
    }
    return model

def save(model, path_npz):
    np.savez_compressed(path_npz,
        image_shape=model["image_shape"],
        roi=model["roi"],
        points=model["points"],
        dirs=model["dirs"],
        weights=model["weights"],
        edge_low=model["edge_threshold"][0],
        edge_high=model["edge_threshold"][1],
        use_polarity=int(model["use_polarity"]),
        template=model["template"],
        version=model.get("version","0.1")
    )

def load(path_npz):
    z = np.load(path_npz, allow_pickle=True)
    model = {
        "image_shape": tuple(z["image_shape"]),
        "roi": tuple(z["roi"]),
        "points": z["points"],
        "dirs": z["dirs"],
        "weights": z["weights"],
        "edge_threshold": (int(z["edge_low"]), int(z["edge_high"])),
        "use_polarity": bool(int(z["use_polarity"])),
        "template": z["template"],
        "version": str(z["version"])
    }
    return model
