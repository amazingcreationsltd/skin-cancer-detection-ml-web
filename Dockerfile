# Docker deploy (light ONNX backend by default).
#   docker build -t skin-cancer .
#   docker run -p 5000:5000 -e MODEL_BACKEND=onnx skin-cancer
# For the full PyTorch backend (Grad-CAM), change the pip line to
# requirements-torch.txt and set MODEL_BACKEND=torch.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=onnx \
    SESSION_TTL_MIN=30 \
    PORT=5000

WORKDIR /app
COPY requirements.txt requirements-torch.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:5000"]
