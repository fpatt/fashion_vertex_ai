FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    google-cloud-storage \
    tensorflow \
    fastapi \
    "uvicorn[standard]"
COPY predictor.py /app/predictor.py
EXPOSE 8080
CMD ["python", "predictor.py"]