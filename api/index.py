# Vercel serverless entrypoint — serves the whole Flask app (pages + API).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MODEL_BACKEND", "onnx")

from app import app  # noqa: F401  (Vercel serves this WSGI `app`)
