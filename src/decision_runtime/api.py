"""Versioned JSON API. Start with one Uvicorn worker; model loading is in lifespan."""

import gc
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
)

from .device_selection import select_device, select_gguf_device
from .runtime import DecisionRuntime
from .service import InferenceService, InvalidState, UnknownHandle


class StateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefix_text: Annotated[StrictStr, Field(min_length=1)]
    suffix_text: StrictStr


class DecisionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    diagnostics: bool | None = None


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[StrictStr, Field(min_length=1)]
    type: Literal["choice"]
    instructions: Annotated[StrictStr, Field(min_length=1)]
    criteria: dict[StrictStr, StrictStr | None]
    decision_options: DecisionOptions | None = None

    @field_validator("criteria")
    @classmethod
    def valid_criteria(cls, criteria):
        if not 2 <= len(criteria) <= 64:
            raise ValueError("Criteria must contain 2–64 choices.")
        if any(not key.strip() for key in criteria):
            raise ValueError("Criterion keys must be non-empty strings.")
        return criteria


def load_service() -> InferenceService:
    data_dir = Path(os.environ.get("DATA_DIR", "data")).expanduser().resolve()
    threads = int(os.environ.get("TORCH_THREADS", "4"))
    if threads < 1:
        raise ValueError("TORCH_THREADS must be positive.")
    torch.set_num_threads(threads)
    requested_device = os.environ.get("DEVICE", "auto")
    runtime = DecisionRuntime(
        configuration_id=os.environ.get("MODEL_CONFIGURATION_ID", "decider_2b_bfloat16_safetensors"),
        device=requested_device,
        models_dir=os.environ.get("MODELS_DIR", str(data_dir / "models")),
        cache_dir=os.environ.get("PREFIX_CACHE_DIR", str(data_dir / "prefix_caches")),
    )
    budget_text = os.environ.get("GPU_MEMORY_BUDGET_GB")
    gpu_budget = float(budget_text) if budget_text else None
    if runtime.approved_configuration["artifact_format"] == "gguf":
        runtime.device, reason = select_gguf_device(requested_device, gpu_budget)
    else:
        dtype = getattr(torch, runtime.approved_configuration["dtype"])
        runtime.device, reason = select_device(requested_device, dtype, gpu_budget)
    print(f"Selected device: {runtime.device}. {reason}")

    started = perf_counter()
    retry_on_cpu = False
    try:
        runtime.load_model()
    except torch.cuda.OutOfMemoryError:
        if requested_device != "auto" or runtime.device != "cuda":
            raise
        retry_on_cpu = True

    # Leave the exception handler first so its traceback cannot retain GPU tensors.
    if retry_on_cpu:
        runtime.model = None
        gc.collect()
        torch.cuda.empty_cache()
        runtime.device = "cpu"
        print("GPU ran out of memory while loading; retrying a fresh model on CPU.")
        runtime.load_model()
    return InferenceService(
        runtime,
        logs_dir=os.environ.get("LOGS_DIR", str(data_dir / "logs")),
        model_load_seconds=perf_counter() - started,
    )


def create_app(service: InferenceService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service if service is not None else load_service()
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="Decision Runtime", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(UnknownHandle)
    async def unknown_handle(request: Request, error: UnknownHandle):
        return JSONResponse(status_code=404, content={"detail": error.args[0]})

    @app.exception_handler(InvalidState)
    async def invalid_state(request: Request, error: InvalidState):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/v1/health")
    def health(request: Request):
        return {"ready": True, "service_instance_id": request.app.state.service.instance_id}

    @app.get("/v1/model")
    def model(request: Request):
        return request.app.state.service.model_details()

    @app.post("/v1/runs", status_code=201)
    def create_run(request: Request):
        return request.app.state.service.create_run()

    @app.post("/v1/runs/{run_id}/states", status_code=201)
    def prepare_state(run_id: str, body: StateRequest, request: Request):
        return request.app.state.service.prepare_state_cache(run_id, body.prefix_text, body.suffix_text)

    @app.post("/v1/runs/{run_id}/states/{state_cache_key}/questions")
    def ask_question(run_id: str, state_cache_key: str, body: QuestionRequest, request: Request):
        return request.app.state.service.ask_question(
            run_id,
            state_cache_key,
            body.model_dump(exclude_none=True, exclude_unset=True),
        )

    @app.delete("/v1/runs/{run_id}/states/{state_cache_key}")
    def release_state(run_id: str, state_cache_key: str, request: Request):
        return request.app.state.service.release_state_cache(run_id, state_cache_key)

    @app.delete("/v1/runs/{run_id}")
    def close_run(run_id: str, request: Request):
        return request.app.state.service.close_run(run_id)

    return app


app = create_app()
