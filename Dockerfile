FROM python:3.13-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    google-cloud-storage \
    tensorflow \
    fastapi \
    "uvicorn[standard]"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8080"]