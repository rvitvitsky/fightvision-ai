FROM python:3.11-slim

WORKDIR /app

# system deps для opencv и ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download YOLO веса в кеш (опционально — можно убрать чтобы сэкономить время сборки)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY config.py worker.py tasks.py ./
COPY pipeline ./pipeline
COPY visualization ./visualization

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
