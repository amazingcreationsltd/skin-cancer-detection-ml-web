# Hosting Guide — GitHub + Render / Vercel / Railway / Docker / HF Spaces

> **Important, read first:** **GitHub Pages cannot run Python.** It only serves
> static HTML/CSS/JS, so your `.pth`/`.onnx` model can never execute there.
> The standard setup is:
>
> * **GitHub repo** = your code + model files (this project).
> * **A Python host** (Render / Railway / Vercel / VPS…) linked to that repo =
>   runs `app.py` and serves the site **including the frontend**.
>
> You do **not** need GitHub Pages at all — the Python host serves the whole
> website. GitHub Pages is only useful as an optional static mirror
> (see Option E).

---

## Step 0 — Push this project to GitHub

```bash
cd skin-cancer-web
git init
git add .
git commit -m "Skin cancer detection web app"
git branch -M main
git remote add origin https://github.com/<YOUR-USER>/<REPO>.git
git push -u origin main
```

The `model/` folder (~29 MB: `.onnx` + `.onnx.data` + `.pth`) is well under
GitHub's 100 MB per-file limit, so it uploads with a normal push — no Git LFS
needed.

---

## Option A — Render (recommended, free, ~5 min) ⭐

1. Go to https://dashboard.render.com → **New + → Web Service**.
2. **Connect your GitHub repo** (link the repo you pushed above).
3. Settings (or just use the included `render.yaml` via **New + → Blueprint**):
   * **Runtime:** Python · **Build command:** `pip install -r requirements.txt`
   * **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 300`
   * **Environment variables:** `MODEL_BACKEND=onnx`, `SESSION_TTL_MIN=30`
4. **Deploy.** You get `https://<your-app>.onrender.com` — open it, upload an
   image, press **Diagnose**. Done.
5. Every future `git push` to `main` auto-redeploys.

Notes: Render's free tier spins down after inactivity (first visit takes
~30–60 s to wake). For Grad-CAM, switch build command to
`pip install -r requirements-torch.txt` and `MODEL_BACKEND=torch`
(needs a paid instance for comfortable RAM — free 512 MB is borderline).

---

## Option B — Vercel (serverless)

The repo already contains `vercel.json` + `api/index.py`.

1. Go to https://vercel.com → **Add New → Project** → import your GitHub repo.
2. Framework preset: **Other**. No build command needed (Python is detected
   from `api/index.py`; dependencies come from `requirements.txt`).
3. Add env var `MODEL_BACKEND=onnx` (optional — it's the default).
4. **Deploy.** You get `https://<project>.vercel.app`.

Caveats (serverless limits):

* Max execution **60 s** — use **Quick mode** for big images; Full mode works
  for typical ≤1600 px images (patches are batched; ~100–300 patches ≈ 10–40 s).
* No Grad-CAM (PyTorch doesn't fit serverless) — heatmaps, risk maps,
  regions and top patches all still work.
* Temp uploads live in the function's `/tmp` (ephemeral, per-instance) and
  are still deleted via the tab-close cleanup call.

---

## Option C — Railway (free trial, no cold starts)

1. https://railway.app → **New Project → Deploy from GitHub repo**.
2. Railway auto-detects Python; set the start command to
   `gunicorn app:app --workers 1 --threads 4 --timeout 300`
   (or it will use the `Procfile` automatically).
3. Add env vars `MODEL_BACKEND=onnx`, `SESSION_TTL_MIN=30`.
4. **Generate Domain** under Settings → Networking. Done.

---

## Option D — Docker (any VPS / local server)

```bash
docker build -t skin-cancer .
docker run -d -p 5000:5000 -e MODEL_BACKEND=onnx --restart unless-stopped skin-cancer
```

For the full PyTorch backend with Grad-CAM, edit the `pip install` line in
`Dockerfile` to `requirements-torch.txt` and run with `-e MODEL_BACKEND=torch`.

---

## Option E — GitHub Pages as a static mirror (optional)

Use this only if you specifically want the page served from
`https://<user>.github.io/<repo>/`. The backend must still run on one of the
options above.

```bash
BACKEND_URL=https://<your-backend-host> python build_ghpages.py
git add docs && git commit -m "GitHub Pages build" && git push
```

Then: repo **Settings → Pages → Deploy from a branch → main → `/docs`**.
The generated `docs/index.html` calls your backend's `/api/*` endpoints.

---

## Option F — Hugging Face Spaces (free GPU optional)

1. Create a Space (Docker SDK), push this repo content.
2. Spaces uses the `Dockerfile` automatically; the app listens on
   `$PORT`… set container port 7860 by adding `-e PORT=7860` — or simply
   change `PORT` handling: the Dockerfile already respects `PORT`, and
   Spaces sets `PORT=7860` by default. ✔️

---

## Choosing a backend

| Backend | Deps size | Hosts | Grad-CAM | Set via |
|---|---|---|---|---|
| `onnx` (default) | ~120 MB | Render, Railway, Vercel, Docker, HF | ❌ | `MODEL_BACKEND=onnx` + `requirements.txt` |
| `torch` (full) | ~800 MB | Render (paid), Railway, Docker, VPS, local | ✅ | `MODEL_BACKEND=torch` + `requirements-torch.txt` |

Everything else (prediction, probabilities, cancer score, patch heatmaps,
risk map, merged regions, clinical panel, top-12 patches) is **identical**
in both backends — the `.onnx` export was numerically verified against the
`.pth` (max diff ~5e-06).

## Temp-file lifecycle (how "delete on close" works)

1. Upload/diagnose creates `uploads_temp/<random-session-id>/` holding the
   original + generated images (also previewable at `/temp/<sid>/<file>`).
2. When the tab closes or hides, the page fires `navigator.sendBeacon` to
   `POST /api/cleanup`, which deletes that folder immediately.
3. Safety nets: a background sweeper deletes sessions idle for more than
   `SESSION_TTL_MIN` (default 30), and the whole temp dir is wiped whenever
   the server restarts. On Vercel, `/tmp` is ephemeral per function instance.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model file not found` on boot | Ensure `model/*.onnx*` (or `.pth` for torch) was pushed; check filename case |
| Vercel `FUNCTION_INVOCATION_TIMEOUT` | Use Quick mode, or smaller images; consider Render instead |
| Render free wakes slowly | Normal (~1 min cold start); upgrade or use Railway for always-on |
| `413 Request Entity Too Large` | Image > 15 MB; raise `MAX_UPLOAD_MB` or compress the image |
| Grad-CAM cards missing | Expected on `onnx` backend — switch to `torch` backend |
