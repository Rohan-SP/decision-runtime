"""Service startup device policy. Status: draft / pending owner review."""

import math

import torch


def select_device(
    mode: str,
    dtype: torch.dtype,
    gpu_memory_budget_gb: float | None,
) -> tuple[str, str]:
    """Select CPU/CUDA using a configured total inference budget in decimal GB."""
    if mode not in {"cpu", "cuda", "auto"}:
        raise ValueError("DEVICE must be cpu, cuda, or auto.")

    if mode == "cpu":
        return "cpu", "CPU explicitly selected."

    if gpu_memory_budget_gb is None:
        problem = "GPU memory budget has not been configured."
    elif not math.isfinite(gpu_memory_budget_gb) or gpu_memory_budget_gb <= 0:
        raise ValueError("GPU memory budget must be finite and positive.")
    elif not torch.cuda.is_available():
        problem = "CUDA is unavailable."
    else:
        probe = result = None
        try:
            with torch.inference_mode():
                probe = torch.ones((16, 16), device="cuda", dtype=dtype)
                result = probe @ probe
                torch.cuda.synchronize()
        except RuntimeError as error:
            problem = f"GPU check failed: {error}"
        else:
            problem = None
        finally:
            # Drop probe references even when the dtype/kernel check fails.
            del probe, result

        if problem is None:
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except RuntimeError as error:
                problem = f"GPU memory check failed: {error}"
            else:
                free_gb = free_bytes / 1_000_000_000
                if free_gb >= gpu_memory_budget_gb:
                    return (
                        "cuda",
                        f"{free_gb:.2f} GB free; budget is {gpu_memory_budget_gb:.2f} GB.",
                    )
                problem = (
                    f"GPU has {free_gb:.2f} GB free; "
                    f"budget requires {gpu_memory_budget_gb:.2f} GB."
                )

    if mode == "cuda":
        raise RuntimeError(problem)
    return "cpu", problem


def select_gguf_device(mode: str, gpu_memory_budget_gb: float | None) -> tuple[str, str]:
    """Select a GGUF backend only when CUDA offload and the budget are available."""
    if mode not in {"cpu", "cuda", "auto"}:
        raise ValueError("DEVICE must be cpu, cuda, or auto.")
    if mode == "cpu":
        return "cpu", "CPU explicitly selected."
    if gpu_memory_budget_gb is None:
        problem = "GPU memory budget has not been configured."
    elif not math.isfinite(gpu_memory_budget_gb) or gpu_memory_budget_gb <= 0:
        raise ValueError("GPU memory budget must be finite and positive.")
    elif not torch.cuda.is_available():
        problem = "CUDA is unavailable."
    else:
        try:
            from llama_cpp import llama_cpp
        except ImportError as error:
            raise RuntimeError("GGUF loading requires llama-cpp-python; install the gguf extra.") from error
        if not llama_cpp.llama_supports_gpu_offload():
            problem = "Installed llama-cpp-python does not support GPU offload."
        else:
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except RuntimeError as error:
                problem = f"GPU memory check failed: {error}"
            else:
                free_gb = free_bytes / 1_000_000_000
                if free_gb >= gpu_memory_budget_gb:
                    return "cuda", f"{free_gb:.2f} GB free; budget is {gpu_memory_budget_gb:.2f} GB."
                problem = (
                    f"GPU has {free_gb:.2f} GB free; "
                    f"budget requires {gpu_memory_budget_gb:.2f} GB."
                )
    if mode == "cuda":
        raise RuntimeError(problem)
    return "cpu", problem
