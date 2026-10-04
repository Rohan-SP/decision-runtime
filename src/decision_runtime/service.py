"""Coordinate one loaded model, isolated runs, state handles, and decision logs."""

import copy
import hashlib
import platform
import random
import sys
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

import torch

from .logger import DecisionLogger
from .runtime import DecisionRuntime


class UnknownHandle(KeyError):
    """A run or state does not belong to this live service session."""


class InvalidState(ValueError):
    """The supplied prefix cannot create a reusable model cache."""


@dataclass
class RunSession:
    logger: DecisionLogger
    states: dict[str, dict[str, str]] = field(default_factory=dict)


class InferenceService:
    """Serialize model operations and retain exact text for server-side logging."""

    def __init__(self, runtime, logs_dir: str | Path, model_load_seconds=0.0):
        self.runtime = runtime
        self.logs_dir = Path(logs_dir).expanduser().resolve()
        self.model_info = runtime.get_model_info()
        self.instance_id = uuid4().hex
        self.model_load_seconds = model_load_seconds
        self.runs: dict[str, RunSession] = {}
        self.inference_lock = RLock()
        folder = Path(__file__).resolve().parent
        hashes = {
            f"decision_runtime/{name}": hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("__init__.py", "runtime.py", "logger.py", "service.py", "api.py", "device_selection.py", "compatible_models.yaml")
        }
        self.implementation_hashes = hashes

    def model_details(self) -> dict:
        gpus = []
        if str(self.model_info.get("device", "cpu")).startswith("cuda"):
            for index in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(index)
                gpus.append({"index": index, "name": gpu.name, "total_memory_bytes": gpu.total_memory})
        return {
            "model_info": self.model_info,
            "service_info": {
                "api_version": 1,
                "instance_id": self.instance_id,
                "package_version": version("decision-runtime"),
                "model_load_seconds": self.model_load_seconds,
                "implementation_hashes": self.implementation_hashes,
                "host": {"platform": platform.platform(), "python": sys.version, "torch_threads": torch.get_num_threads(), "gpus": gpus},
            },
        }

    def _run(self, run_id: str) -> RunSession:
        if run_id not in self.runs:
            raise UnknownHandle("Unknown run; create a run on this service instance.")
        return self.runs[run_id]

    def _synchronize(self) -> None:
        device = str(self.model_info.get("device", "cpu"))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)

    def create_run(self) -> dict:
        with self.inference_lock:
            logger = DecisionLogger(self.logs_dir)
            created = logger.create_run()
            self.runs[created["run_id"]] = RunSession(logger)
            return {"run_id": created["run_id"], "service_instance_id": self.instance_id}

    def prepare_state_cache(self, run_id, prefix_text, suffix_text) -> dict:
        with self.inference_lock:
            run = self._run(run_id)
            self._synchronize()
            started = perf_counter()
            prefix_ids, _ = self.runtime.split_state_tokens(prefix_text, suffix_text)
            if not prefix_ids:
                raise InvalidState("The prefix must contain reusable tokens before the suffix.")
            prefix_key = self.runtime.build_prefix_cache_key(prefix_ids)
            reused = self.runtime.prefix_cache_path(prefix_key).is_file()
            prepared = self.runtime.prepare_state_cache(prefix_text, suffix_text)
            self._synchronize()
            elapsed = perf_counter() - started
            run.states[prepared["state_cache_key"]] = {
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
            }
            return {
                **prepared,
                "metadata": {"prefix_cache_reused": reused, "prepare_seconds": elapsed},
            }

    def ask_question(self, run_id, state_cache_key, question) -> dict:
        with self.inference_lock:
            run = self._run(run_id)
            if state_cache_key not in run.states:
                raise UnknownHandle("Unknown state for this run; prepare the state again.")
            state = run.states[state_cache_key]
            question = copy.deepcopy(question)
            # Randomize the keyed alternatives on every call. The exact order
            # used for scoring is retained in the question log and prompt.
            criteria = list(question["criteria"].items())
            random.SystemRandom().shuffle(criteria)
            question["criteria"] = dict(criteria)
            self._synchronize()
            started = perf_counter()
            answer = self.runtime.ask_question(state_cache_key, question)
            self._synchronize()
            elapsed = perf_counter() - started
            decision_id = run.logger.log_decision(
                prefix_text=state["prefix_text"],
                suffix_text=state["suffix_text"],
                question=question,
                question_text=self.runtime.format_question(question),
                answer=answer,
                model_info=self.model_info,
            )
            return {
                "answer": answer,
                "decision_id": decision_id,
                "metadata": {"inference_seconds": elapsed},
            }

    def release_state_cache(self, run_id, state_cache_key) -> dict:
        with self.inference_lock:
            run = self._run(run_id)
            if state_cache_key not in run.states:
                return {"released": False}
            self.runtime.release_state_cache(state_cache_key)
            del run.states[state_cache_key]
            return {"released": True}

    def close_run(self, run_id) -> dict:
        with self.inference_lock:
            run = self.runs.get(run_id)
            if run is None:
                return {"closed": False, "released_states": 0}
            count = len(run.states)
            for key in list(run.states):
                self.runtime.release_state_cache(key)
                del run.states[key]
            del self.runs[run_id]
            return {"closed": True, "released_states": count}

    def close(self) -> None:
        with self.inference_lock:
            for run_id in list(self.runs):
                self.close_run(run_id)
