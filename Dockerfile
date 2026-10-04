# Status: draft / pending owner review.
FROM python:3.12-slim

ARG GGUF_BACKEND=none

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

# PyPI may build the CPU llama.cpp binding from source.
RUN if [ "$GGUF_BACKEND" = "cpu" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake && rm -rf /var/lib/apt/lists/*; \
    fi

# One CUDA-enabled Torch build supports both CPU and GPU execution.
RUN python -m pip install \
        "torch==2.14.0" \
        --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install ".[service]" \
    && if [ "$GGUF_BACKEND" = "cpu" ]; then \
         python -m pip install ".[gguf]"; \
       elif [ "$GGUF_BACKEND" = "cuda" ]; then \
         python -m pip install "diskcache>=5.6.1" "jinja2>=2.11.3" \
           "typing-extensions>=4.5.0" \
         && python -m pip install --no-deps --only-binary=:all: \
           "llama-cpp-python>=0.3.16,<1" \
           --index-url https://abetlen.github.io/llama-cpp-python/whl/cu130; \
       elif [ "$GGUF_BACKEND" != "none" ]; then \
         echo "GGUF_BACKEND must be none, cpu, or cuda" >&2; exit 1; \
       fi

# Torch's CUDA 13 libraries live outside the system linker path. The CUDA
# llama.cpp wheel needs these at import time; libcuda comes from the GPU host.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/site-packages/llama_cpp/lib

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "decision_runtime.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
