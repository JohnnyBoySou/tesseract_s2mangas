FROM python:3.11-slim

# Instalar Tesseract e idiomas japoneses
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
