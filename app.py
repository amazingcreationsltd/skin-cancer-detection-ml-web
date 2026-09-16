"""
Skin Cancer Detection — web server.

    Upload image -> press Diagnose -> MARF+CBAM MobileNetV3 result
    (prediction, confidence, class probabilities, cancer heatmaps,
    suspicious regions, top patches — same outputs as me+probal_work.ipynb).

Run locally:
    pip install -r requirements.txt        # light (ONNX) backend
    # or: pip install -r requirements-torch.txt   # full torch backend + Grad-CAM
    python app.py

Environment:
    MODEL_BACKEND   onnx | torch | auto   (default: auto)
    PORT            default 5000
    SESSION_TTL_MIN default 30  (temp uploads older than this are deleted)
"""

import atexit
import os
import shutil
import tempfile
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory

from inference import Detector, encode_image_b64
from model_def import CLASS_NAMES

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("MODEL_BACKEND", "auto").lower()
SESSION_TTL = int(os.environ.get("SESSION_TTL_MIN", "30")) * 60  # seconds
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))

# Temp workspace: ./uploads_temp locally, /tmp on serverless (Vercel etc.)
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    TEMP_ROOT = os.path.join(tempfile.gettempdir(), "skin_cancer_temp")
else:
    TEMP_ROOT = os.path.join(HERE, "uploads_temp")
os.makedirs(TEMP_ROOT, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print(f"[boot] backend={BACKEND} temp={TEMP_ROOT}", flush=True)
detector = Detector(backend=BACKEND)
print(f"[boot] model ready (backend={detector.backend}, classes={len(CLASS_NAMES)})", flush=True)

# ----------------------------------------------------------------------------
# Temp session helpers
# ----------------------------------------------------------------------------

def session_dir(sid: str) -> str:
    d = os.path.join(TEMP_ROOT, sid)
    os.makedirs(d, exist_ok=True)
    return d


def allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def cleanup_session(sid: str) -> bool:
    d = os.path.join(TEMP_ROOT, sid)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def sweep_old_sessions():
    """Delete temp sessions older than SESSION_TTL (runs in background)."""
    while True:
        try:
            now = time.time()
            for sid in os.listdir(TEMP_ROOT):
                d = os.path.join(TEMP_ROOT, sid)
                if not os.path.isdir(d):
                    continue
                try:
                    mtime = os.path.getmtime(d)
                except OSError:
                    continue
                if now - mtime > SESSION_TTL:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[sweep] deleted expired session {sid}", flush=True)
        except Exception as e:
            print(f"[sweep] error: {e}", flush=True)
        time.sleep(300)  # every 5 minutes


# Background sweeper (disabled on serverless — /tmp there is ephemeral anyway)
if not (os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")):
    t = threading.Thread(target=sweep_old_sessions, daemon=True)
    t.start()

    def _cleanup_all_on_exit():
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    atexit.register(_cleanup_all_on_exit)

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "backend": detector.backend,
        "classes": CLASS_NAMES,
        "gradcam": detector.backend == "torch",
        "temp_root": TEMP_ROOT,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """Save uploaded image into a temp session dir; return preview + session id."""
    if "image" not in request.files:
        return jsonify({"error": "No image file part"}), 400
    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Please upload a valid image (png/jpg/jpeg/bmp/tif/webp)"}), 400

    sid = request.form.get("session_id") or uuid.uuid4().hex
    d = session_dir(sid)
    ext = f.filename.rsplit(".", 1)[1].lower()
    if ext == "tiff":
        ext = "tif"
    path = os.path.join(d, f"original.{ext}")
    f.save(path)

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            preview = encode_image_b64(np.asarray(im), max_side=768)
    except Exception:
        os.remove(path)
        return jsonify({"error": "Could not read image file"}), 400

    # touch session dir so TTL counts from last activity
    os.utime(d, None)
    return jsonify({"session_id": sid, "width": w, "height": h, "preview": preview})


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Run the detection pipeline.
    Accepts EITHER multipart file field 'image' OR form field 'session_id'
    (image previously uploaded via /api/upload). Optional 'mode': full|quick.
    """
    mode = (request.form.get("mode") or "full").lower()
    if mode not in ("full", "quick"):
        mode = "full"

    image_rgb = None
    sid = request.form.get("session_id")

    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        if not allowed_file(f.filename):
            return jsonify({"error": "Invalid image type"}), 400
        sid = sid or uuid.uuid4().hex
        d = session_dir(sid)
        ext = f.filename.rsplit(".", 1)[1].lower()
        path = os.path.join(d, f"original.{ext}")
        f.save(path)
    elif sid:
        d = os.path.join(TEMP_ROOT, sid)
        path = None
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.startswith("original."):
                    path = os.path.join(d, name)
                    break
        if not path:
            return jsonify({"error": "Session expired or image not found — please upload again"}), 404
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        with Image.open(path) as im:
            image_rgb = np.asarray(im.convert("RGB"))
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    # Downscale very large images for server safety (keeps aspect ratio)
    h, w = image_rgb.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_rgb = np.asarray(
            Image.fromarray(image_rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        )

    try:
        result = detector.diagnose(image_rgb, mode=mode)
    except Exception as e:
        print(f"[diagnose] error: {e}", flush=True)
        return jsonify({"error": f"Diagnosis failed: {e}"}), 500

    # Persist visual outputs into the temp session dir too (previewable, auto-deleted)
    try:
        import base64 as _b64
        d = session_dir(sid)
        for name, uri in result.get("images", {}).items():
            if uri and "," in uri:
                with open(os.path.join(d, f"{name}.jpg"), "wb") as fh:
                    fh.write(_b64.b64decode(uri.split(",", 1)[1]))
        for tp in result.get("top_patches", []):
            if tp.get("image") and "," in tp["image"]:
                with open(os.path.join(d, f"patch_{tp['rank']:02d}.jpg"), "wb") as fh:
                    fh.write(_b64.b64decode(tp["image"].split(",", 1)[1]))
        os.utime(d, None)
    except Exception as e:
        print(f"[diagnose] temp-save warning: {e}", flush=True)

    result["session_id"] = sid
    result["image_size"] = {"width": int(image_rgb.shape[1]), "height": int(image_rgb.shape[0])}
    result["disclaimer"] = (
        "Research demo only — not a medical diagnosis. "
        "Consult a qualified dermatologist for clinical decisions."
    )
    return jsonify(result)


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    """Delete a temp session (called by the browser when the page is closed)."""
    sid = (request.form.get("session_id") or "").strip()
    # also accept sendBeacon blobs / JSON
    if not sid and request.data:
        try:
            import json as _json
            sid = (_json.loads(request.data.decode() or "{}").get("session_id") or "").strip()
        except Exception:
            sid = ""
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    return jsonify({"deleted": cleanup_session(sid), "session_id": sid})


@app.route("/temp/<sid>/<filename>")
def serve_temp(sid, filename):
    """Serve a file from a temp session dir (for preview)."""
    d = os.path.join(TEMP_ROOT, sid)
    if not os.path.isdir(d):
        return jsonify({"error": "session not found"}), 404
    return send_from_directory(d, filename)


@app.route("/api/temp/<sid>")
def list_temp(sid):
    d = os.path.join(TEMP_ROOT, sid)
    if not os.path.isdir(d):
        return jsonify({"error": "session not found"}), 404
    return jsonify({"session_id": sid, "files": sorted(os.listdir(d))})


# Vercel / gunicorn entrypoint compatibility
handler = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
