# Skin Cancer Detection — Web App

Upload a lesion image → press **Diagnose** → get the prediction from the
**MARF + CBAM MobileNetV3** research model (8 classes: dermoscopic /
histopathology × BCC / Melanoma / Nevus / SCC).

Detection logic is ported 1:1 from `me+probal_work.ipynb`:

| Notebook cell | Web output |
|---|---|
| 8 — whole-image prediction | predicted class, confidence, all 8 probabilities |
| 20/23 — malignant probability | **cancer score** = P(BCC)+P(Melanoma)+P(SCC) |
| 14/15 — Grad-CAM | CAM + overlay (PyTorch backend) |
| 22 — adaptive overlapping patches | patch grid sized to the image |
| 23 — per-patch analysis | high / medium / low patch counts |
| 24 — smooth heatmap | JET heatmap + overlay |
| 25/26 — top suspicious patches | top-12 crops (+ Grad-CAM on top-6, torch) |
| 27 — detection map | red / yellow / green risk boxes |
| 28/29 — merged regions + clinical panel | numbered regions + AI summary image |

Plus: MARF modality signal α (dermoscopy-like vs histopathology-like).

## Quick start (local)

```bash
pip install -r requirements.txt     # light ONNX backend (default)
python app.py                       # -> http://localhost:5000
```

For Grad-CAM overlays use the full backend instead:

```bash
pip install -r requirements-torch.txt
MODEL_BACKEND=torch python app.py
```

## Project layout

```
├── app.py                  # Flask server (pages + /api/*)
├── inference.py            # detection engine (notebook logic)
├── model_def.py            # MARF+CBAM MobileNetV3 architecture
├── model/
│   ├── marf_cbam_mobilenetv3_final.onnx (+ .onnx.data)  # light backend
│   └── marf_cbam_mobilenetv3_final.pth                  # torch backend
├── templates/index.html    # page
├── static/                 # style.css, app.js
├── uploads_temp/           # per-visit temp dir (auto-created, auto-deleted)
├── api/index.py + vercel.json     # Vercel serverless
├── render.yaml, Procfile, Dockerfile, runtime.txt
└── DEPLOY.md               # full hosting guide (GitHub, Vercel, Render…)
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | web UI |
| `/api/health` | GET | `{status, backend, classes, gradcam}` |
| `/api/upload` | POST | multipart `image` → `{session_id, preview, width, height}` |
| `/api/diagnose` | POST | `image` file *or* `session_id` + `mode=full\|quick` → full result JSON |
| `/api/cleanup` | POST | `{session_id}` → deletes that temp session |
| `/temp/<sid>/<file>` | GET | preview a temp file |
| `/api/temp/<sid>` | GET | list temp session files |

## Privacy / temp files

* Every visit gets a random `session_id`; uploads + generated heatmaps are
  stored only under `uploads_temp/<session_id>/`.
* The browser calls `/api/cleanup` via `sendBeacon` when the tab is closed
  (`pagehide` + `visibilitychange`), deleting that folder.
* A server sweeper also deletes any session idle for > `SESSION_TTL_MIN`
  (default 30 min), and everything is wiped on server shutdown.

## Disclaimer

Research demo only — **not a medical diagnosis**. Consult a qualified
dermatologist for clinical decisions.

See **[DEPLOY.md](DEPLOY.md)** for step-by-step hosting on GitHub + Render /
Vercel / Railway / Docker / Hugging Face Spaces.
