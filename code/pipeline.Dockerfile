FROM python:3.11-slim
WORKDIR /project

ENV GIT_PYTHON_REFRESH=quiet

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY code/datasets/data_pipeline.py /project/code/datasets/data_pipeline.py
COPY code/models/train.py /project/code/models/train.py
COPY docker_scheduler.py /project/docker_scheduler.py
CMD ["python", "docker_scheduler.py"]
