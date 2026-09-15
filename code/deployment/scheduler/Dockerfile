FROM python:3.11-slim

WORKDIR /app

ENV GIT_PYTHON_REFRESH=quiet

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_loop.sh /app/run_loop.sh
RUN chmod +x /app/run_loop.sh

# code/, data/, and models/ are mounted as volumes at runtime (see docker-compose.yml)
# so retraining picks up code and data changes without rebuilding the image.
ENTRYPOINT ["/app/run_loop.sh"]
