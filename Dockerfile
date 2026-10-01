# Status: draft / pending owner review.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF"

WORKDIR /app

# llama-cpp-python is currently a core package dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# One CUDA-enabled Torch build supports both CPU and GPU execution.
RUN python -m pip install \
        "torch==2.14.0" \
        --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install ".[service]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "decision_runtime.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
