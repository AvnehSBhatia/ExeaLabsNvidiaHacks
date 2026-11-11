# Headless CPU image: MuJoCo renders through EGL/OSMesa, no GPU needed.
FROM python:3.15-slim

ENV MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY arrival ./arrival
RUN pip install -e ".[sim]"
COPY configs ./configs
COPY scans ./scans
COPY scripts ./scripts

EXPOSE 8000
CMD ["python", "-m", "arrival", "serve", "--host", "0.0.0.0"]
