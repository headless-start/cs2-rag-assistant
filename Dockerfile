FROM python:3.12-slim

WORKDIR /app

# CPU torch wheels keep the image lean; the GPU path is for running on the host.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
