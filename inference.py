"""
Skin-cancer detection engine.

Ports the detection logic from the research notebook
    me+probal_work.ipynb
to the final trained weights
    marf_cbam_mobilenetv3_final (.pth / .onnx)

Pipeline stages (notebook cell -> method):
    cell  8 : whole-image prediction + class probabilities  -> predict_whole()
    cell 14 : Grad-CAM class                                -> GradCAM (torch backend only)
    cell 15 : Grad-CAM prediction + overlay                 -> gradcam_full_image()
    cell 22 : adaptive overlapping patch extraction        -> extract_patches()
    cell 23 : per-patch MobileNetV3 analysis + CancerScore  -> analyze_patches()
    cell 24 : smooth cancer-probability heatmap + overlay   -> smooth_heatmap()
    cell 25 : top suspicious patches                       -> top_patches()
    cell 27 : cancer detection map (R/Y/G boxes)            -> risk_map()
    cell 28 : merged suspicious regions                    -> merged_regions()
    cell 29 : AI clinical visualization + legend panel      -> clinical_panel()

Backends:
    * "onnx"  (default) — ONNX Runtime, tiny dependency footprint,
      works on Vercel / Render / Railway free tiers. No Grad-CAM
      (needs autograd), everything else identical.
    * "torch" — full PyTorch backend incl. Grad-CAM. Needs torch+timm deps.
"""

import base64
import io
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # optional — faster blur/morphology/contours when available
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    ort = None
    HAS_ORT = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from model_def import CLASS_NAMES, MALIGNANT_CLASSES, NORM_MEAN, NORM_STD, IMAGE_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(HERE, "model", "marf_cbam_mobilenetv3_final.onnx")
DEFAULT_PTH = os.path.join(HERE, "model", "marf_cbam_mobilenetv3_final.pth")

PATCH_BATCH = 32          # patch inference batch size
MAX_PATCHES = 600         # safety cap — stride is widened if exceeded
TOP_K = 12                # top suspicious patches (notebook cell 25)
GRADCAM_TOP = 6           # Grad-CAM overlays for top-N patches (torch only)
HIGH_T = 0.85             # notebook cell 27 thresholds
MED_T = 0.60
MERGE_T = 0.75            # notebook cell 28 region-merge threshold


# ----------------------------------------------------------------------------
# Small helpers (PIL / numpy fallbacks so the app also runs without OpenCV)
# ----------------------------------------------------------------------------

def encode_image_b64(img_rgb: np.ndarray, fmt="JPEG", quality=82, max_side=1024) -> str:
    """RGB numpy array -> base64 data-uri (downscaled for transfer)."""
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        pil = Image.fromarray(img_rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    else:
        pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    return "data:image/%s;base64,%s" % (fmt.lower(), base64.b64encode(buf.getvalue()).decode())


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    """JET heat colormap (0..1 float) -> RGB uint8. Matches cv2 COLORMAP_JET closely."""
    x = np.clip(gray.astype(np.float32), 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def gaussian_blur_fallback(img: np.ndarray, ksize=31, sigma=10) -> np.ndarray:
    """Separable gaussian blur with pure numpy (used when cv2 is missing)."""
    k = int(ksize) | 1
    ax = np.arange(k) - k // 2
    kernel = np.exp(-0.5 * (ax / sigma) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    pad = k // 2
    x = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, x)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, tmp)
    return out


def smooth_heatmap_array(heat: np.ndarray) -> np.ndarray:
    k = 31
    if min(heat.shape[:2]) < 64:
        k = 7
    if HAS_CV2:
        heat = cv2.GaussianBlur(heat, (k, k), sigmaX=10)
    else:
        heat = gaussian_blur_fallback(heat, ksize=k, sigma=max(2, k // 3))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat.astype(np.float32)


def draw_box_pil(img_rgb, x1, y1, x2, y2, color, thickness=2):
    d = ImageDraw.Draw(Image.fromarray(img_rgb)) if isinstance(img_rgb, np.ndarray) else ImageDraw.Draw(img_rgb)
    for t in range(thickness):
        d.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=tuple(int(c) for c in color))
    return np.array(d.im) if False else None  # placeholder (we draw on PIL images directly)


def risk_color(score: float):
    """Notebook cell-27 colors (RGB): red >= .85, yellow >= .60, else green."""
    if score >= HIGH_T:
        return (255, 0, 0)
    if score >= MED_T:
        return (255, 255, 0)
    return (0, 255, 0)


def risk_label(score: float) -> str:
    if score >= HIGH_T:
        return "HIGH"
    if score >= 0.55:
        return "MED"
    return "LOW"


def overall_risk(cancer_score: float) -> str:
    if cancer_score >= HIGH_T:
        return "High"
    if cancer_score >= MED_T:
        return "Moderate"
    return "Low"


# ----------------------------------------------------------------------------
# Pre-processing (identical normalization to every notebook transform)
# ----------------------------------------------------------------------------

def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """PIL image -> NCHW float32 tensor (Resize 224 + normalize)."""
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))  # CHW


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Grad-CAM (torch backend only) — notebook cell 14, verbatim logic
# ----------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, image_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(image_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def find_target_layer(torch_model):
    """Last Conv2d of the backbone — notebook cell-14 auto-detection."""
    for layer in reversed(list(torch_model.features.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise RuntimeError("No Conv2d layer found in model.features")


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------

class Detector:
    def __init__(self, backend="auto", onnx_path=DEFAULT_ONNX, pth_path=DEFAULT_PTH,
                 device="cpu", patch_batch=PATCH_BATCH):
        self.patch_batch = patch_batch
        self.device = device
        self.onnx_path = onnx_path
        self.pth_path = pth_path

        want = (backend or "auto").lower()
        if want == "auto":
            want = "onnx" if (HAS_ORT and os.path.exists(onnx_path)) else "torch"

        if want == "onnx":
            if not HAS_ORT:
                raise RuntimeError("onnxruntime is not installed")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
            self.session = ort.InferenceSession(onnx_path, sess_options=opts,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.backend = "onnx"
            self.torch_model = None
            self.gradcam = None
        elif want == "torch":
            if not HAS_TORCH:
                raise RuntimeError("torch is not installed")
            if not os.path.exists(pth_path):
                raise FileNotFoundError(f".pth weights not found: {pth_path}")
            from model_def import build_model
            self.torch_model = build_model(num_classes=len(CLASS_NAMES),
                                           weights_path=pth_path, device=device)
            self.session = None
            self.backend = "torch"
            self.gradcam = GradCAM(self.torch_model, find_target_layer(self.torch_model))
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # -- core forward -----------------------------------------------------
    def predict_batch(self, batch_chw: np.ndarray):
        """NCHW float32 -> (probs[N,8], alpha[N])."""
        if self.backend == "onnx":
            logits, alpha, _ = self.session.run(None, {self.input_name: batch_chw})
            return softmax_np(logits), np.asarray(alpha).reshape(-1)
        else:
            with torch.no_grad():
                x = torch.from_numpy(batch_chw).to(self.device)
                logits, alpha, _ = self.torch_model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs, np.asarray(alpha.detach().cpu()).reshape(-1)

    # -- notebook cell 8 : whole-image prediction --------------------------
    def predict_whole(self, pil_img: Image.Image):
        x = np.expand_dims(preprocess_pil(pil_img), 0)
        probs, alpha = self.predict_batch(x)
        probs = probs[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        cancer = float(probs[MALIGNANT_CLASSES].sum())
        return {
            "prediction": CLASS_NAMES[pred],
            "prediction_idx": pred,
            "confidence": conf,
            "cancer_score": cancer,
            "risk": overall_risk(cancer),
            "modality_alpha": float(alpha[0]),
            "modality_hint": "Histopathology-like" if float(alpha[0]) >= 0.5 else "Dermoscopy-like",
            "probabilities": [
                {"class": c, "prob": float(probs[i]), "malignant": bool(i in MALIGNANT_CLASSES)}
                for i, c in enumerate(CLASS_NAMES)
            ],
        }

    # -- notebook cell 22 : adaptive overlapping patch extraction ----------
    @staticmethod
    def extract_patches(image_rgb: np.ndarray):
        H, W = image_rgb.shape[:2]
        short_side = min(H, W)
        if short_side < 400:
            patch_size = 32
        elif short_side < 800:
            patch_size = 64
        elif short_side < 1600:
            patch_size = 128
        else:
            patch_size = 256
        stride = max(1, patch_size // 2)

        def _grid(st):
            ps, cs = [], []
            for y in range(0, max(H - patch_size + 1, 1), st):
                for x in range(0, max(W - patch_size + 1, 1), st):
                    y2 = min(y + patch_size, H)
                    x2 = min(x + patch_size, W)
                    patch = image_rgb[y:y2, x:x2]
                    if patch.shape[0] < patch_size // 2 or patch.shape[1] < patch_size // 2:
                        continue
                    ps.append(patch)
                    cs.append((x, y, x2, y2))
            return ps, cs

        patches, coords = _grid(stride)
        while len(patches) > MAX_PATCHES and stride < patch_size:
            stride = min(patch_size, stride * 2)  # widen stride to respect server cap
            patches, coords = _grid(stride)
        return patches, coords, patch_size, stride

    # -- notebook cell 23 : per-patch analysis -----------------------------
    def analyze_patches(self, patches, coords):
        n = len(patches)
        all_probs = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
        for s in range(0, n, self.patch_batch):
            e = min(s + self.patch_batch, n)
            batch = np.stack([
                preprocess_pil(Image.fromarray(p).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
                for p in patches[s:e]
            ]).astype(np.float32)
            probs, _ = self.predict_batch(batch)
            all_probs[s:e] = probs
        rows = []
        for i in range(n):
            probs = all_probs[i]
            pred = int(np.argmax(probs))
            rows.append({
                "patch": i,
                "x1": int(coords[i][0]), "y1": int(coords[i][1]),
                "x2": int(coords[i][2]), "y2": int(coords[i][3]),
                "prediction": CLASS_NAMES[pred],
                "confidence": float(probs[pred]),
                "cancer_score": float(probs[MALIGNANT_CLASSES].sum()),
            })
        return rows

    # -- notebook cell 15 : whole-image Grad-CAM (torch only) --------------
    def gradcam_full_image(self, pil_img: Image.Image, image_rgb: np.ndarray, pred_idx: int):
        if self.backend != "torch":
            return None, None
        x = torch.from_numpy(np.expand_dims(preprocess_pil(pil_img), 0)).to(self.device)
        cam, _ = self.gradcam.generate(x, pred_idx)
        H, W = image_rgb.shape[:2]
        if HAS_CV2:
            cam_rs = cv2.resize(cam, (W, H))
        else:
            cam_rs = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H))) / 255.0
        heat = jet_colormap(cam_rs)
        if HAS_CV2:
            overlay = cv2.addWeighted(image_rgb, 0.60, heat, 0.40, 0)
        else:
            overlay = (image_rgb.astype(np.float32) * 0.60 + heat.astype(np.float32) * 0.40).astype(np.uint8)
        return jet_colormap(cam_rs), overlay

    def gradcam_patch(self, patch_rgb: np.ndarray, pred_idx: int):
        """Grad-CAM overlay for a single patch crop (torch only, cell 26 logic)."""
        if self.backend != "torch":
            return None
        pil = Image.fromarray(patch_rgb)
        x = torch.from_numpy(np.expand_dims(preprocess_pil(pil), 0)).to(self.device)
        cam, _ = self.gradcam.generate(x, pred_idx)
        h, w = patch_rgb.shape[:2]
        if HAS_CV2:
            cam_rs = cv2.resize(cam, (w, h))
            heat = cv2.applyColorMap(np.uint8(255 * cam_rs), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(patch_rgb, 0.60, heat, 0.40, 0)
        cam_rs = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))) / 255.0
        heat = jet_colormap(cam_rs)
        return (patch_rgb.astype(np.float32) * 0.60 + heat.astype(np.float32) * 0.40).astype(np.uint8)

    # -- notebook cell 24 : smooth heatmap --------------------------------
    @staticmethod
    def smooth_heatmap(image_rgb, rows):
        H, W = image_rgb.shape[:2]
        heat = np.zeros((H, W), dtype=np.float32)
        counter = np.zeros((H, W), dtype=np.float32)
        for r in rows:
            heat[r["y1"]:r["y2"], r["x1"]:r["x2"]] += r["cancer_score"]
            counter[r["y1"]:r["y2"], r["x1"]:r["x2"]] += 1
        counter[counter == 0] = 1
        heat = smooth_heatmap_array(heat / counter)
        heat_color = jet_colormap(heat)
        if HAS_CV2:
            overlay = cv2.addWeighted(image_rgb, 0.55, heat_color, 0.45, 0)
            for r in rows:
                cv2.rectangle(overlay, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (255, 255, 255), 1)
        else:
            overlay = (image_rgb.astype(np.float32) * 0.55 + heat_color.astype(np.float32) * 0.45).astype(np.uint8)
            pil = Image.fromarray(overlay)
            d = ImageDraw.Draw(pil)
            for r in rows:
                d.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=(255, 255, 255))
            overlay = np.array(pil)
        return heat, heat_color, overlay

    # -- notebook cell 27 : risk map with R/Y/G boxes ----------------------
    @staticmethod
    def risk_map(image_rgb, rows):
        vis = image_rgb.copy()
        overlay = image_rgb.copy()
        for r in rows:
            # NOTE: our arrays are RGB and cv2 drawing calls write the tuple
            # straight into channels 0/1/2, so plain RGB tuples are correct.
            c = risk_color(r["cancer_score"])
            if HAS_CV2:
                cv2.rectangle(overlay, (r["x1"], r["y1"]), (r["x2"], r["y2"]), c, -1)
                thick = 2 if r["cancer_score"] >= MED_T else 1
                cv2.rectangle(vis, (r["x1"], r["y1"]), (r["x2"], r["y2"]), c, thick)
            else:
                overlay[r["y1"]:r["y2"], r["x1"]:r["x2"]] = (
                    0.45 * np.array(c) + 0.55 * overlay[r["y1"]:r["y2"], r["x1"]:r["x2"]]
                ).astype(np.uint8)
        if HAS_CV2:
            result = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
        else:
            result = overlay.copy()
            pil = Image.fromarray(result)
            d = ImageDraw.Draw(pil)
            for r in rows:
                c = risk_color(r["cancer_score"])
                thick = 2 if r["cancer_score"] >= MED_T else 1
                for t in range(thick):
                    d.rectangle([r["x1"] - t, r["y1"] - t, r["x2"] + t, r["y2"] + t], outline=c)
            result = np.array(pil)
        return result

    # -- notebook cell 28 : merged suspicious regions ----------------------
    @staticmethod
    def merged_regions(image_rgb, rows, heat01):
        """
        Returns list of region dicts {x,y,w,h,score,label}.
        cv2 path  = exact notebook morphology + contours.
        fallback  = union-find merge of adjacent high-risk patches on the grid.
        """
        H, W = image_rgb.shape[:2]
        if HAS_CV2:
            mask = np.zeros((H, W), dtype=np.uint8)
            for r in rows:
                if r["cancer_score"] >= MERGE_T:
                    mask[r["y1"]:r["y2"], r["x1"]:r["x2"]] = 255
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                score = float(heat01[y:y + h, x:x + w].mean()) if heat01 is not None else 1.0
                regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                                "score": score, "label": risk_label(score)})
            return regions, mask
        # ---- grid-based fallback (no cv2) ----
        hi = [r for r in rows if r["cancer_score"] >= MERGE_T]
        parent = {r["patch"]: r["patch"] for r in hi}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(hi)):
            for j in range(i + 1, len(hi)):
                a, b = hi[i], hi[j]
                # adjacent / overlapping boxes?
                if not (a["x2"] < b["x1"] or b["x2"] < a["x1"] or a["y2"] < b["y1"] or b["y2"] < a["y1"]):
                    union(a["patch"], b["patch"])
        groups = {}
        for r in hi:
            groups.setdefault(find(r["patch"]), []).append(r)
        regions = []
        for g in groups.values():
            x1 = min(r["x1"] for r in g)
            y1 = min(r["y1"] for r in g)
            x2 = max(r["x2"] for r in g)
            y2 = max(r["y2"] for r in g)
            score = float(np.mean([r["cancer_score"] for r in g]))
            regions.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "score": score, "label": risk_label(score)})
        return regions, None

    # -- notebook cell 29 : clinical panel ---------------------------------
    @staticmethod
    def clinical_panel(image_rgb, rows, regions, heat01, prediction, confidence):
        display = image_rgb.copy()
        overlay = image_rgb.copy()
        if HAS_CV2:
            for i, rg in enumerate(regions, 1):
                color = {"HIGH": (255, 0, 0), "MED": (255, 255, 0)}.get(rg["label"], (0, 255, 0))
                x, y, w, h = rg["x"], rg["y"], rg["w"], rg["h"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
                cv2.circle(display, (x + 15, y + 15), 12, (255, 255, 255), -1)
                cv2.putText(display, str(i), (x + 9, y + 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                cv2.putText(display, rg["label"], (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
            panel = np.ones((120, display.shape[1], 3), dtype=np.uint8) * 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(panel, "AI Diagnostic Summary", (20, 30), font, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, "RED    : High Risk", (20, 65), font, 0.6, (255, 0, 0), 2)
            cv2.putText(panel, "YELLOW : Medium Risk", (220, 65), font, 0.6, (255, 255, 0), 2)
            cv2.putText(panel, "GREEN  : Low Risk", (470, 65), font, 0.6, (0, 255, 0), 2)
            cv2.putText(panel, f"Prediction : {prediction}", (20, 105), font, 0.65, (255, 255, 255), 2)
            cv2.putText(panel, f"Confidence : {confidence * 100:.1f}%", (500, 105), font, 0.65, (255, 255, 255), 2)
            return np.vstack((display, panel))
        # ---- PIL fallback ----
        pil = Image.fromarray(display).convert("RGBA")
        glow = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        d = ImageDraw.Draw(pil)
        for i, rg in enumerate(regions, 1):
            color = {"HIGH": (255, 0, 0), "MED": (255, 255, 0)}.get(rg["label"], (0, 255, 0))
            x, y, w, h = rg["x"], rg["y"], rg["w"], rg["h"]
            gd.rectangle([x, y, x + w, y + h], fill=color + (90,))
            for t in range(3):
                d.rectangle([x - t, y - t, x + w + t, y + h + t], outline=color)
            d.ellipse([x + 3, y + 3, x + 27, y + 27], fill=(255, 255, 255))
            d.text((x + 9, y + 7), str(i), fill=(0, 0, 0))
            d.text((x, max(0, y - 14)), rg["label"], fill=color)
        pil = Image.alpha_composite(pil, glow).convert("RGB")
        panel = Image.new("RGB", (pil.width, 120), (30, 30, 30))
        pd = ImageDraw.Draw(panel)
        pd.text((20, 8), "AI Diagnostic Summary", fill=(255, 255, 255))
        pd.text((20, 40), "RED : High Risk", fill=(255, 60, 60))
        pd.text((220, 40), "YELLOW : Medium Risk", fill=(255, 255, 0))
        pd.text((470, 40), "GREEN : Low Risk", fill=(0, 255, 0))
        pd.text((20, 75), f"Prediction : {prediction}", fill=(255, 255, 255))
        pd.text((500, 75), f"Confidence : {confidence * 100:.1f}%", fill=(255, 255, 255))
        out = Image.new("RGB", (pil.width, pil.height + 120))
        out.paste(pil, (0, 0))
        out.paste(panel, (0, pil.height))
        return np.array(out)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def diagnose(self, image_rgb: np.ndarray, mode="full"):
        t0 = time.time()
        pil_img = Image.fromarray(image_rgb)
        timings = {}

        # 1) whole-image diagnosis (cell 8)
        t = time.time()
        whole = self.predict_whole(pil_img)
        timings["whole_image"] = round(time.time() - t, 2)

        result = {
            "backend": self.backend,
            "mode": mode,
            "prediction": whole["prediction"],
            "confidence": whole["confidence"],
            "cancer_score": whole["cancer_score"],
            "risk": whole["risk"],
            "modality_alpha": whole["modality_alpha"],
            "modality_hint": whole["modality_hint"],
            "probabilities": whole["probabilities"],
            "images": {},
            "top_patches": [],
            "patch_stats": None,
            "regions": [],
            "timings": timings,
        }

        # 2) whole-image Grad-CAM (cells 14-15, torch only)
        if self.backend == "torch":
            t = time.time()
            cam, overlay = self.gradcam_full_image(pil_img, image_rgb, whole["prediction_idx"])
            if cam is not None:
                result["images"]["gradcam"] = encode_image_b64(cam)
                result["images"]["gradcam_overlay"] = encode_image_b64(overlay)
            timings["gradcam"] = round(time.time() - t, 2)

        if mode == "quick":
            timings["total"] = round(time.time() - t0, 2)
            result["timings"] = timings
            return result

        # 3) patch extraction (cell 22)
        t = time.time()
        patches, coords, patch_size, stride = self.extract_patches(image_rgb)
        timings["patch_extract"] = round(time.time() - t, 2)

        # 4) per-patch analysis (cell 23)
        t = time.time()
        rows = self.analyze_patches(patches, coords)
        timings["patch_infer"] = round(time.time() - t, 2)
        scores = np.array([r["cancer_score"] for r in rows])
        result["patch_stats"] = {
            "total": len(rows),
            "patch_size": patch_size,
            "stride": stride,
            "high": int((scores >= HIGH_T).sum()),
            "medium": int(((scores >= MED_T) & (scores < HIGH_T)).sum()),
            "low": int((scores < MED_T).sum()),
            "avg_cancer_score": float(scores.mean()) if len(scores) else 0.0,
            "max_cancer_score": float(scores.max()) if len(scores) else 0.0,
        }

        # 5) smooth heatmap + overlay (cell 24)
        t = time.time()
        heat01, heat_color, heat_overlay = self.smooth_heatmap(image_rgb, rows)
        result["images"]["heatmap"] = encode_image_b64(heat_color)
        result["images"]["heatmap_overlay"] = encode_image_b64(heat_overlay)

        # 6) risk map (cell 27)
        result["images"]["risk_map"] = encode_image_b64(self.risk_map(image_rgb, rows))

        # 7) merged regions (cell 28) + clinical panel (cell 29)
        regions, _mask = self.merged_regions(image_rgb, rows, heat01)
        result["regions"] = regions
        result["images"]["clinical"] = encode_image_b64(
            self.clinical_panel(image_rgb, rows, regions, heat01,
                                whole["prediction"], whole["confidence"]))
        timings["visuals"] = round(time.time() - t, 2)

        # 8) top suspicious patches (cells 25-26)
        order = sorted(rows, key=lambda r: r["cancer_score"], reverse=True)[:TOP_K]
        top = []
        for rank, r in enumerate(order, 1):
            crop = patches[r["patch"]]
            item = {
                "rank": rank,
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "cancer_score": r["cancer_score"],
                "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]],
                "image": encode_image_b64(crop, max_side=256),
                "gradcam": None,
            }
            if self.backend == "torch" and rank <= GRADCAM_TOP:
                gi = self.gradcam_patch(crop, CLASS_NAMES.index(r["prediction"]))
                if gi is not None:
                    item["gradcam"] = encode_image_b64(gi, max_side=256)
            top.append(item)
        result["top_patches"] = top

        timings["total"] = round(time.time() - t0, 2)
        result["timings"] = timings
        return result
