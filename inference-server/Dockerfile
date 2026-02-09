ARG DEVICE=gpu

# -- builder stage --
FROM python:3.12-slim AS builder
ARG DEVICE

RUN pip install uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
RUN if [ "$DEVICE" = "cpu" ]; then \
      uv pip install --python .venv/bin/python \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu --reinstall; \
    fi

# -- production stage --
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY main.py .

ENV INDEX_DIR=/data/index
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 6000

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6000"]
