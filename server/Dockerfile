FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Workspace metadata first, so dependency layers are cached between code changes.
# Every workspace member's pyproject.toml must be copied here.
COPY pyproject.toml uv.lock ./
COPY core/pyproject.toml core/pyproject.toml
COPY client-windows/pyproject.toml client-windows/pyproject.toml
COPY server/pyproject.toml server/pyproject.toml
RUN uv sync --locked --no-dev --package svd-server --no-install-workspace

COPY server/src server/src
RUN uv sync --locked --no-dev --package svd-server --no-editable

RUN useradd --create-home --uid 10001 svd \
    && mkdir -p /home/svd/.cache/huggingface \
    && chown -R svd:svd /home/svd
USER svd

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/home/svd/.cache/huggingface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["svd-server", "serve"]
