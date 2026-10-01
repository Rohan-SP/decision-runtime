"""Orchestrator-facing entry point for decision-model workflows."""

import copy
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
import yaml
from huggingface_hub import snapshot_download
from llama_cpp import Llama
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
        self.model = None
        self.tokenizer = None

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
            weight_variant = self.approved_configuration["weight_variant"]
            gguf_file = next(
                self.model_path.glob(f"*{weight_variant}.gguf"),
                None,
            )
            if gguf_file is None:
                raise FileNotFoundError(
                    f"No {weight_variant} GGUF file found in {self.model_path}."
                )

            self.model = Llama(
                model_path=str(gguf_file),
                n_ctx=2048,
                verbose=False,
            )
            self.tokenizer = None

        else:
            raise ValueError(f"Unsupported artifact format: {artifact_format!r}")

    def get_model_info(self) -> dict:
        """Return model identity and settings needed to record a decision."""

        configuration = self.approved_configuration
        parameter = next(self.model.parameters())

        config_path = self.model_path / "decider_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        temperature = config.get("temperature_by_type", {}).get(
            "choice",
            config.get("temperature", 1.0),
        )

        return {
            "model_id": self.model_id,
            "configuration_id": self.configuration_id,
            "repo_id": self.model_config["repo_id"],
            "revision": configuration["revision"],
            "artifact_format": configuration["artifact_format"],
            "weight_variant": configuration.get("weight_variant"),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "choice_temperature": temperature,
            "backend": {
                "name": "transformers",
                "torch_version": str(torch.__version__),
                "transformers_version": version("transformers"),
                "tokenizers_version": version("tokenizers"),
            },
        }

    def split_state_tokens(
        self,
        prefix_text: str,
        suffix_text: str,
    ) -> tuple[list[int], list[int]]:
        """Split a complete state's tokens into reusable prefix and suffix IDs."""
        prefix_ids = self.tokenizer.encode(
            prefix_text,
            add_special_tokens=False,
        )
        full_state_ids = self.tokenizer.encode(
            prefix_text + suffix_text,
            add_special_tokens=False,
        )

        reusable_count = 0
        for prefix_id, full_state_id in zip(prefix_ids, full_state_ids):
            if prefix_id != full_state_id:
                break
            reusable_count += 1

        cache_prefix_ids = full_state_ids[:reusable_count]
        suffix_ids = full_state_ids[reusable_count:]
        return cache_prefix_ids, suffix_ids

    def build_prefix_cache_key(self, cache_prefix_ids: list[int]) -> str:
        """Hash reusable prefix tokens and the configuration that produced them."""
        configuration = self.approved_configuration
        device_type = next(self.model.parameters()).device.type

        identity = {
            "cache_format_version": 2,
            "libraries": {
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "tokenizers": version("tokenizers"),
            },
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

        canonical_identity = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical_identity).hexdigest()

    def create_prefix_cache(self, cache_prefix_ids: list[int]):
        """Run the reusable prefix through the model and return its cache."""
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

        cache_path = self.cache_dir / f"{cache_key}.pt"
        temporary_path = cache_path.with_suffix(".tmp")

        torch.save(prefix_cache, temporary_path)
        temporary_path.replace(cache_path)

        return cache_path

    def load_prefix_cache(self, cache_key: str):
        """Load a saved prefix cache, or return None if it is missing."""
        cache_path = self.cache_dir / f"{cache_key}.pt"

        if not cache_path.is_file():
            return None

        model_device = next(self.model.parameters()).device

        prefix_cache = torch.load(
            cache_path,
            map_location=model_device,
            weights_only=False,
        )

        return prefix_cache

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

        cache_key = self.build_prefix_cache_key(prefix_ids)
        prefix_cache = self.load_prefix_cache(cache_key)

        if prefix_cache is None:
            prefix_cache = self.create_prefix_cache(prefix_ids)
            self.save_prefix_cache(cache_key, prefix_cache)

        state_cache = self.extend_prefix_cache(prefix_cache, suffix_ids)
        state_cache_key = self.store_state_cache(state_cache)

        return {
            "prefix_cache_key": cache_key,
            "state_cache_key": state_cache_key,
        }

    def format_question(self, question: dict) -> str:
        """Render a question in the format used for model inference."""

        labels = "ABCDEFGHIJ"[:len(question["choices"])]

        options = "".join(
            f"\n({label}) {choice}"
            for label, choice in zip(labels, question["choices"])
        )

        return (
            f"\n\nQuestion: {question['stem']}"
            f"\nRubric: {question['rubric']}"
            f"\nOptions:{options}"
            "\nAnswer: ("
        )

    # kjl2: Add optional ask_question diagnostics exposing raw choice-label
    # logits, mapped to the supplied choices, for scoring and calibration work.
    # kjl2: Include the applied choice temperature in optional diagnostics so
    # callers can track how the returned probabilities were produced.
    # kjl2: Consider optional diagnostics for allowed-answer-token probability
    # mass across the full vocabulary before normalizing among choices; this
    # is a formatting/scoring signal, not a probability that the answer is right.

    # kjl3: Consider returning only the full probability distribution and base
    # model data from ask_question, leaving selected-choice, top-probability,
    # runner-up, margin, and entropy calculations to the orchestrator. This is
    # a tentative simplification idea, not a decided change to the output.
    def ask_question(self, state_cache_key: str, question: dict) -> dict:
        """Return choice probabilities and statistics without changing the state."""
        state_cache = self.get_state_cache(state_cache_key)
        question_cache = copy.deepcopy(state_cache)

        choices = question["choices"]
        labels = "ABCDEFGHIJ"[:len(choices)]

        question_text = self.format_question(question)

        question_ids = self.tokenizer.encode(
            question_text,
            add_special_tokens=False,
        )

        label_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in labels
        ]

        state_length = question_cache.get_seq_length()
        model_device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [question_ids],
            dtype=torch.long,
            device=model_device,
        )

        attention_mask = torch.ones(
            (1, state_length + len(question_ids)),
            dtype=torch.long,
            device=model_device,
        )

        position_ids = torch.arange(
            state_length,
            state_length + len(question_ids),
            device=model_device,
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

            label_logits = output.logits[0, -1, label_ids].float()

        config_path = self.model_path / "decider_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        temperature = config.get("temperature_by_type", {}).get(
            "choice",
            config.get("temperature", 1.0),
        )

        probabilities = torch.softmax(
            label_logits / temperature,
            dim=0,
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

        return {
            "question_id": question["id"],
            "state_cache_key": state_cache_key,
            "type": "choice",
            "choice": top_choice,
            "probabilities": distribution,
            "statistics": {
                "top_probability": top_probability,
                "runner_up_choice": runner_up_choice,
                "runner_up_probability": runner_up_probability,
                "top_two_margin": top_probability - runner_up_probability,
                "normalized_entropy": normalized_entropy,
            },
        }
