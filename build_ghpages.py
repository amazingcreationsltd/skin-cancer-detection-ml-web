"""
Build a standalone single-file frontend for GitHub Pages hosting.

GitHub Pages can only serve STATIC files (no Python), so this script packs
index.html + style.css + app.js into one file (docs/index.html) that talks to
your Python backend running elsewhere (Render / Railway / Vercel / VPS).

Usage:
    BACKEND_URL=https://skin-cancer-api.onrender.com python build_ghpages.py

Then: push, and in the repo go to Settings -> Pages -> Deploy from branch ->
select branch + /docs folder. Your site appears at
https://<user>.github.io/<repo>/
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_URL = os.environ.get("BACKEND_URL", "https://YOUR-BACKEND-URL").rstrip("/")

with open(os.path.join(HERE, "templates", "index.html")) as f:
    html = f.read()
with open(os.path.join(HERE, "static", "style.css")) as f:
    css = f.read()
with open(os.path.join(HERE, "static", "app.js")) as f:
    js = f.read()

js = js.replace('const API_BASE = "";', f'const API_BASE = "{BACKEND_URL}";')

html = html.replace(
    '<link rel="stylesheet" href="/static/style.css" />',
    "<style>\n" + css + "\n</style>",
)
html = html.replace(
    '<script src="/static/app.js"></script>',
    "<script>\n" + js + "\n</script>",
)
html = html.replace(
    "<title>",
    f'<!-- Standalone GitHub-Pages build. Backend: {BACKEND_URL} -->\n<title>',
    1,
)

docs = os.path.join(HERE, "docs")
os.makedirs(docs, exist_ok=True)
with open(os.path.join(docs, "index.html"), "w") as f:
    f.write(html)
with open(os.path.join(docs, ".nojekyll"), "w") as f:
    f.write("")

print(f"docs/index.html written ({len(html) / 1024:.0f} KB), backend = {BACKEND_URL}")
