FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# One image serves both the API and the app; docker-compose.yml picks which
# one to run per-container via the `command:` override.
COPY api/main.py ./api_main.py
COPY app/app.py ./app_main.py

EXPOSE 8000 8501
