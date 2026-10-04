"""Orchestrator-facing entry point for decision-model workflows."""

import copy
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
import yaml
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class DecisionRuntime:
    """Coordinate model loading, reusable state caches, and decision questions."""

    def __init__(
        self,
        configuration_id: str,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        registry_path = Path(__file__).with_name("compatible_models.yaml")
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

        approved_configuration = next(
            (
                configuration
                for configuration in registry["approved_configurations"]
                if configuration["id"] == configuration_id
            ),
            None,
        )
        if approved_configuration is None:
            raise ValueError(
                "Configuration id does not exist in the compatible models registry: "
                f"{configuration_id!r}."
            )
        if "development" not in approved_configuration["approved_for"]:
            raise ValueError(
                f"Configuration {configuration_id!r} is not approved for development."
            )

        model_config = next(
            (
                model
                for model in registry["models"]
                if model["id"] == approved_configuration["model_id"]
            ),
            None,
        )
        if model_config is None:
            raise ValueError(
                "The approved configuration references a model missing from the "
                f"compatible models registry: {approved_configuration['model_id']!r}."
            )

        self.configuration_id = configuration_id
        self.model_id = approved_configuration["model_id"]
        self.device = device
        self.model_config = model_config
        self.approved_configuration = approved_configuration

        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "prefix_caches"

        self.cache_dir = Path(cache_dir).expanduser().resolve()

        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"

        self.models_dir = Path(models_dir).expanduser().resolve()
        self._state_caches: dict[str, object] = {}
        self._verified_answer_labels: list[tuple[str, int]] | None = None
        self.model = None
        self.tokenizer = None
        self.gguf_context_tokens = 4096
        self.gguf_batch_tokens = 64
        if self.is_gguf:
            self.gguf_context_tokens = int(os.environ.get("GGUF_CONTEXT_TOKENS", "4096"))
            self.gguf_batch_tokens = int(os.environ.get("GGUF_BATCH_TOKENS", "64"))
        if self.is_gguf and (self.gguf_context_tokens < 2 or self.gguf_batch_tokens < 1):
            raise ValueError("GGUF_CONTEXT_TOKENS must be at least 2 and GGUF_BATCH_TOKENS positive.")

    @property
    def is_gguf(self) -> bool:
        return self.approved_configuration["artifact_format"] == "gguf"

    def load_model(self) -> None:
        """Download the approved artifact if needed, then load it locally."""
        configuration = self.approved_configuration
        self.model_path = (
            self.models_dir
            / configuration["id"]
            / configuration["revision"]
        )
        download_files = configuration["download_files"]

        model_files_exist = (
            self.model_path.is_dir()
            and all(
                any(
                    file_path.is_file()
                    for file_path in self.model_path.glob(file_pattern)
                )
                for file_pattern in download_files
            )
        )

        if not model_files_exist:
            self.model_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=self.model_config["repo_id"],
                revision=configuration["revision"],
                local_dir=self.model_path,
                allow_patterns=download_files,
            )

        self._load_from_local_files()

    def _load_from_local_files(self) -> None:
        """Load the selected local artifact with its matching backend."""
        artifact_format = self.approved_configuration["artifact_format"]

        if artifact_format == "safetensors":
            dtype = getattr(torch, self.approved_configuration["dtype"])
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.model.to(device)
            self.model.eval()

        elif artifact_format == "gguf":
            try:
                from llama_cpp import Llama, llama_cpp
            except ImportError as error:
                raise RuntimeError(
                    "GGUF loading requires llama-cpp-python; install the gguf extra."
                ) from error

            if self.device == "auto":
                # Direct library callers have no service GPU budget to inspect.
                self.device = "cpu"
            if self.device not in {"cpu", "cuda"}:
                raise ValueError("GGUF device must be cpu, cuda, or auto.")
            if self.device == "cuda" and (
                not torch.cuda.is_available()
                or not llama_cpp.llama_supports_gpu_offload()
            ):
                raise RuntimeError("CUDA GGUF loading requires CUDA and GPU-enabled llama-cpp-python.")

            weight_variant = self.approved_configuration["weight_variant"]
            gguf_files = sorted(self.model_path.glob(f"*{weight_variant}.gguf"))
            if len(gguf_files) != 1:
                raise FileNotFoundError(
                    f"Expected one {weight_variant} GGUF file in {self.model_path}; found {len(gguf_files)}."
                )

            self.model = Llama(
                model_path=str(gguf_files[0]),
                n_ctx=self.gguf_context_tokens,
                n_batch=self.gguf_batch_tokens,
                n_threads=torch.get_num_threads(),
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                logits_all=False,
                verbose=False,
            )
            self.tokenizer = None

        else:
            raise ValueError(f"Unsupported artifact format: {artifact_format!r}")

    def get_model_info(self) -> dict:
        """Return model identity and settings needed to record a decision."""

        configuration = self.approved_configuration
        temperature = self._default_choice_temperature()

        if self.is_gguf:
            dtype = configuration["weight_variant"]
            device = self.device
            backend = {
                "name": "llama.cpp",
                "llama_cpp_python_version": version("llama-cpp-python"),
                "context_tokens": self.model.n_ctx(),
                "batch_tokens": self.gguf_batch_tokens,
                "gpu_layers_requested": -1 if device == "cuda" else 0,
            }
        else:
            parameter = next(self.model.parameters())
            dtype = str(parameter.dtype)
            device = str(parameter.device)
            backend = {
                "name": "transformers",
                "torch_version": str(torch.__version__),
                "transformers_version": version("transformers"),
                "tokenizers_version": version("tokenizers"),
            }

        return {
            "model_id": self.model_id,
            "configuration_id": self.configuration_id,
            "repo_id": self.model_config["repo_id"],
            "revision": configuration["revision"],
            "artifact_format": configuration["artifact_format"],
            "weight_variant": configuration.get("weight_variant"),
            "dtype": dtype,
            "device": device,
            "choice_temperature": temperature,
            "backend": backend,
        }

    def split_state_tokens(
        self,
        prefix_text: str,
        suffix_text: str,
    ) -> tuple[list[int], list[int]]:
        """Split a complete state's tokens into reusable prefix and suffix IDs."""
        if self.is_gguf:
            prefix_ids = self.model.tokenize(prefix_text.encode("utf-8"), add_bos=True)
            full_state_ids = self.model.tokenize(
                (prefix_text + suffix_text).encode("utf-8"), add_bos=True
            )
        else:
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            full_state_ids = self.tokenizer.encode(
                prefix_text + suffix_text, add_special_tokens=False
            )

        reusable_count = 0
        for prefix_id, full_state_id in zip(prefix_ids, full_state_ids):
            if prefix_id != full_state_id:
                break
            reusable_count += 1

        if (
            self.is_gguf
            and reusable_count == 1
            and full_state_ids[0] == self.model.token_bos()
        ):
            reusable_count = 0

        cache_prefix_ids = full_state_ids[:reusable_count]
        suffix_ids = full_state_ids[reusable_count:]
        return cache_prefix_ids, suffix_ids

    def build_prefix_cache_key(self, cache_prefix_ids: list[int]) -> str:
        """Hash reusable prefix tokens and the configuration that produced them."""
        configuration = self.approved_configuration
        if self.is_gguf:
            libraries = {"llama-cpp-python": version("llama-cpp-python")}
            device_type = self.device
            cache_version = 3
        else:
            libraries = {
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "tokenizers": version("tokenizers"),
            }
            device_type = next(self.model.parameters()).device.type
            cache_version = 2

        identity = {
            "cache_format_version": cache_version,
            "libraries": libraries,
            "model_id": self.model_id,
            "repo_id": self.model_config["repo_id"],
            "configuration_id": self.configuration_id,
            "revision": configuration["revision"],
            "artifact_format": configuration["artifact_format"],
            "dtype": configuration.get("dtype"),
            "weight_variant": configuration.get("weight_variant"),
            "device_type": device_type,
            "prefix_token_ids": cache_prefix_ids,
        }
        if self.is_gguf:
            identity["context_tokens"] = self.model.n_ctx()
            identity["batch_tokens"] = self.gguf_batch_tokens

        canonical_identity = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical_identity).hexdigest()

    def create_prefix_cache(self, cache_prefix_ids: list[int]):
        """Run the reusable prefix through the model and return its cache."""
        if self.is_gguf:
            self._check_gguf_capacity(len(cache_prefix_ids))
            self.model.reset()
            self.model.eval(cache_prefix_ids)
            return self._save_gguf_state()
        model_device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [cache_prefix_ids],
            dtype=torch.long,
            device=model_device,
        )

        attention_mask = torch.ones_like(input_ids)

        position_ids = torch.arange(
            len(cache_prefix_ids),
            device=model_device,
        ).unsqueeze(0)

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )

        prefix_cache = output.past_key_values
        return prefix_cache

    def save_prefix_cache(self, cache_key: str, prefix_cache) -> Path:
        """Save a computed prefix cache and return its file path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.prefix_cache_path(cache_key)
        temporary_path = cache_path.with_suffix(".tmp")

        if self.is_gguf:
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    input_ids=prefix_cache.input_ids,
                    scores=prefix_cache.scores,
                    n_tokens=np.array(prefix_cache.n_tokens, dtype=np.int64),
                    llama_state=np.frombuffer(prefix_cache.llama_state, dtype=np.uint8),
                    seed=np.array(prefix_cache.seed, dtype=np.int64),
                )
        else:
            torch.save(prefix_cache, temporary_path)
        temporary_path.replace(cache_path)

        return cache_path

    def load_prefix_cache(self, cache_key: str):
        """Load a saved prefix cache, or return None if it is missing."""
        cache_path = self.prefix_cache_path(cache_key)

        if not cache_path.is_file():
            return None

        if self.is_gguf:
            from llama_cpp import LlamaState

            with np.load(cache_path, allow_pickle=False) as saved:
                raw_state = saved["llama_state"].tobytes()
                return LlamaState(
                    input_ids=saved["input_ids"].copy(),
                    scores=saved["scores"].copy(),
                    n_tokens=int(saved["n_tokens"]),
                    llama_state=raw_state,
                    llama_state_size=len(raw_state),
                    seed=int(saved["seed"]),
                )

        model_device = next(self.model.parameters()).device

        prefix_cache = torch.load(
            cache_path,
            map_location=model_device,
            weights_only=False,
        )

        return prefix_cache

    def prefix_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}{'.npz' if self.is_gguf else '.pt'}"

    def _check_gguf_capacity(self, token_count: int) -> None:
        if token_count >= self.model.n_ctx():
            raise ValueError(
                f"Prompt needs {token_count} tokens plus an answer, but GGUF context is "
                f"{self.model.n_ctx()} tokens. Increase GGUF_CONTEXT_TOKENS."
            )

    def _save_gguf_state(self):
        state = self.model.save_state()
        # llama-cpp-python retains a scores buffer even with logits_all=False.
        # Those scores are unused here and may contain uninitialized values.
        state.scores.fill(0)
        state.input_ids[state.n_tokens:] = 0
        return state

    def store_state_cache(self, state_cache) -> str:
        """Keep a prepared state cache in memory and return its lookup key."""
        state_cache_key = uuid4().hex
        self._state_caches[state_cache_key] = state_cache
        return state_cache_key

    def get_state_cache(self, state_cache_key: str):
        """Retrieve a prepared state cache from memory."""
        return self._state_caches[state_cache_key]

    def release_state_cache(self, state_cache_key: str) -> None:
        """Remove the runtime's reference to a prepared state cache."""
        self._state_caches.pop(state_cache_key, None)

    def extend_prefix_cache(self, prefix_cache, suffix_ids: list[int]):
        """Extend a copy of the prefix cache with the suffix tokens."""
        if self.is_gguf:
            self._check_gguf_capacity(prefix_cache.n_tokens + len(suffix_ids))
            self.model.load_state(prefix_cache)
            if suffix_ids:
                self.model.eval(suffix_ids)
            return self._save_gguf_state()
        state_cache = copy.deepcopy(prefix_cache)

        if not suffix_ids:
            return state_cache

        model_device = next(self.model.parameters()).device
        prefix_length = state_cache.get_seq_length()

        input_ids = torch.tensor(
            [suffix_ids],
            dtype=torch.long,
            device=model_device,
        )

        attention_mask = torch.ones(
            (1, prefix_length + len(suffix_ids)),
            dtype=torch.long,
            device=model_device,
        )

        position_ids = torch.arange(
            prefix_length,
            prefix_length + len(suffix_ids),
            device=model_device,
        ).unsqueeze(0)

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=state_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        state_cache = output.past_key_values
        return state_cache

    def prepare_state_cache(self, prefix_text: str, suffix_text: str):
        """Retrieve or create the prefix cache and extend it with the suffix."""
        prefix_ids, suffix_ids = self.split_state_tokens(
            prefix_text,
            suffix_text,
        )
        if not prefix_ids:
            raise ValueError("The prefix contains no reusable tokens.")

        cache_key = self.build_prefix_cache_key(prefix_ids)
        prefix_cache = self.load_prefix_cache(cache_key)

        if prefix_cache is None:
            prefix_cache = self.create_prefix_cache(prefix_ids)
            self.save_prefix_cache(cache_key, prefix_cache)

        state_cache = self.extend_prefix_cache(prefix_cache, suffix_ids)
        if self.is_gguf:
            state_cache = {
                "state": state_cache,
                "token_ids": prefix_ids + suffix_ids,
                "text": prefix_text + suffix_text,
            }
        state_cache_key = self.store_state_cache(state_cache)

        return {
            "prefix_cache_key": cache_key,
            "state_cache_key": state_cache_key,
        }

    def format_question(self, question: dict) -> str:
        """Render generic instructions and keyed choice criteria for scoring."""
        criteria = list(question["criteria"].items())
        labels = self._get_verified_answer_labels(len(criteria))
        options = []
        for (label, _), (key, description) in zip(labels, criteria):
            if description is None:
                option = key
            else:
                option = f"{key}: {description}"
            options.append(f"{label}: {option}\n")

        return (
            f"\nInstructions:\n{question['instructions']}\nOptions:\n"
            + "".join(options)
            + "Return the correct letter label.\nAnswer:\n"
        )

    def _default_choice_temperature(self) -> float:
        """Read the selected model's declared decision settings, if present."""
        settings_name = self.model_config.get("decision_settings_file")
        if not settings_name:
            return 1.0

        settings_path = self.model_path / settings_name
        if not settings_path.is_file():
            return 1.0
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        return settings.get("temperature_by_type", {}).get(
            "choice",
            settings.get("choice_temperature", settings.get("temperature", 1.0)),
        )

    def _get_verified_answer_labels(self, count: int) -> list[tuple[str, int]]:
        """Return unique labels proven to be exact single-token strings."""
        if self.tokenizer is None and not self.is_gguf:
            raise RuntimeError("Typed choice scoring requires a tokenizer.")
        if self._verified_answer_labels is None:
            candidates = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
            candidates.extend(
                first + second
                for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            verified: list[tuple[str, int]] = []
            seen_token_ids: set[int] = set()
            for label in candidates:
                if self.is_gguf:
                    token_ids = self.model.tokenize(label.encode("utf-8"), add_bos=False)
                else:
                    token_ids = self.tokenizer.encode(label, add_special_tokens=False)
                if len(token_ids) != 1 or token_ids[0] in seen_token_ids:
                    continue
                if self.is_gguf:
                    decoded = self.model.detokenize(token_ids).decode("utf-8")
                else:
                    decoded = self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                if decoded != label:
                    continue
                verified.append((label, token_ids[0]))
                seen_token_ids.add(token_ids[0])
                if len(verified) == 64:
                    break
            self._verified_answer_labels = verified

        if len(self._verified_answer_labels) < count:
            raise ValueError(
                f"This model has only {len(self._verified_answer_labels)} verified "
                f"single-token answer labels; the question needs {count}."
            )
        return self._verified_answer_labels[:count]
    def ask_question(self, state_cache_key: str, question: dict) -> dict:
        """Return choice probabilities and statistics without changing the state."""
        state_cache = self.get_state_cache(state_cache_key)
        question_cache = None if self.is_gguf else copy.deepcopy(state_cache)

        ordered_criteria = list(question["criteria"].items())
        choices = [key for key, _ in ordered_criteria]
        answer_labels = self._get_verified_answer_labels(len(choices))

        question_text = self.format_question(question)

        label_ids = [token_id for _, token_id in answer_labels]
        if self.is_gguf:
            from llama_cpp import llama_cpp

            full_ids = self.model.tokenize(
                (state_cache["text"] + question_text).encode("utf-8"), add_bos=True
            )
            self._check_gguf_capacity(len(full_ids))
            state_ids = state_cache["token_ids"]
            if full_ids[:len(state_ids)] == state_ids:
                self.model.load_state(state_cache["state"])
                self.model.eval(full_ids[len(state_ids):])
            else:
                # A token spanning the state/question boundary invalidates that
                # cache; evaluate the exact concatenated prompt instead.
                self.model.reset()
                self.model.eval(full_ids)
            logits_pointer = llama_cpp.llama_get_logits(self.model.ctx)
            if not logits_pointer:
                raise RuntimeError("llama.cpp did not return next-token logits.")
            label_logits = [float(logits_pointer[token_id]) for token_id in label_ids]
        else:
            question_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
            state_length = question_cache.get_seq_length()
            model_device = next(self.model.parameters()).device
            input_ids = torch.tensor([question_ids], dtype=torch.long, device=model_device)
            attention_mask = torch.ones(
                (1, state_length + len(question_ids)), dtype=torch.long, device=model_device
            )
            position_ids = torch.arange(
                state_length, state_length + len(question_ids), device=model_device
            ).unsqueeze(0)
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=question_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                label_logits_tensor = output.logits[0, -1, label_ids].float()
                label_logits = label_logits_tensor.cpu().tolist()

        default_temperature = self._default_choice_temperature()
        decision_options = question.get("decision_options") or {}
        temperature = decision_options.get("temperature", default_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Choice temperature must be positive and finite.")
        diagnostics = bool(decision_options.get("diagnostics", False))

        if self.is_gguf:
            largest_logit = max(label_logits)
            weights = [
                math.exp((logit - largest_logit) / temperature)
                for logit in label_logits
            ]
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
        else:
            probabilities = torch.softmax(
                label_logits_tensor / temperature, dim=0
            ).cpu().tolist()

        distribution = dict(zip(choices, probabilities))

        ranked = sorted(
            distribution.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_choice, top_probability = ranked[0]
        runner_up_choice, runner_up_probability = ranked[1]

        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        normalized_entropy = entropy / math.log(len(choices))

        answer = {
            "question_id": question["id"],
            "state_cache_key": state_cache_key,
            "type": "choice",
            "choice": top_choice,
            "probabilities": distribution,
            "confidence": max(0.0, min(1.0, 1.0 - normalized_entropy)),
            "statistics": {
                "top_probability": top_probability,
                "runner_up_choice": runner_up_choice,
                "runner_up_probability": runner_up_probability,
                "top_two_margin": top_probability - runner_up_probability,
                "normalized_entropy": normalized_entropy,
            },
        }
        if diagnostics:
            answer["diagnostics"] = {
                "choice_logits": dict(
                    zip(choices, label_logits)
                ),
                "temperature": temperature,
                "probability_semantics": "conditional_on_answers",
                "confidence_method": "normalized_inverse_entropy",
            }
        return answer
