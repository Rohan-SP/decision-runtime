# Decision Runtime

Status: **draft / pending owner review**.

A standalone inference service for decision questions against reusable context.
It owns model loading, prefix caches, combined state caches and completed-decision
logs. Callers provide exact context text and questions; workflow definitions and
graph/contract assembly belong to the caller.

## Install and start

Use Python 3.11 or newer and an isolated environment. From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[service,test]"
$env:DATA_DIR = ".\data"
$env:DEVICE = "cpu"
.\.venv\Scripts\python.exe -m uvicorn decision_runtime.api:app --host 127.0.0.1 --port 8000 --workers 1
```

The installed Torch build determines CPU/CUDA availability. For GPU deployment,
install a matching CUDA Torch build and set DEVICE to auto or cuda with a
GPU_MEMORY_BUDGET_GB value.
The model loads once during startup, before requests are accepted.
Use **one worker**, because model and state-handle ownership live in one process.
The interactive API reference is available at http://127.0.0.1:8000/docs.

If the selected weights are missing, the runtime downloads its approved pinned
snapshot. To use existing weights, set MODELS_DIR to their configuration/revision
directory parent. HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 disable downloads.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| MODEL_CONFIGURATION_ID | decider_2b_bfloat16_safetensors | Approved configuration in the packaged registry |
| DEVICE | auto | cpu, cuda, or auto; service startup policy below |
| GPU_MEMORY_BUDGET_GB | unset | Required for GPU selection; total inference budget in decimal GB |
| TORCH_THREADS | 4 | CPU Torch threads |
| DATA_DIR | data under the working directory | Default persistent storage root |
| MODELS_DIR | DATA_DIR/models | Downloaded model storage |
| PREFIX_CACHE_DIR | DATA_DIR/prefix_caches | Saved prefix caches |
| LOGS_DIR | DATA_DIR/logs | Completed-decision log runs |

Absolute mount paths can override all three storage directories.
The API accepts context and questions, rather than caller-selected filesystem
paths. Serving currently supports the Safetensors state/scoring implementation.
GGUF model loading exists in the prototype runtime; GGUF service scoring is pending.


## Startup device selection

The service uses [device_selection.py](src/decision_runtime/device_selection.py)
before loading the model:

- DEVICE=cpu always uses CPU without checking CUDA.
- DEVICE=auto uses GPU only when a positive, finite budget is configured, CUDA
  is available, a small operation works with the approved model dtype, and free
  VRAM meets the budget. Otherwise it selects CPU and prints the reason.
- DEVICE=cuda requires those checks to pass; it reports an error instead of
  silently switching to CPU.

In auto mode only, a CUDA out-of-memory error during model loading discards the
partial model, clears unused GPU allocations and reloads a fresh model on CPU.
Invalid configuration and other loading errors propagate. The selected device
stays fixed while serving
questions; no automatic switch happens during inference.

The budget includes weights, expected live state caches, temporary computation
and reserve. It is not the downloaded file size. For the 2B BF16 model on the
8 GB GPU, 6 decimal GB is an initial value to measure and adjust, not a validated
capacity guarantee. Larger contexts or concurrent retained states may need more
memory. The startup check does not reserve VRAM or predict every later workload.
The actual device remains part of model metadata and prefix-cache identity.
Startup policy source is included in service implementation hashes.

## Container (CPU or GPU)

Status: **draft / pending owner review**. Docker Desktop must be running with
Linux containers. The Dockerfile now installs CUDA-enabled Torch 2.14.0 from
the official [cu130 wheel index](https://download.pytorch.org/whl/cu130/torch/).
One image supports CPU and GPU execution; DEVICE selects the startup behavior.
The shared image was built and verified in CPU mode on 2026-10-01 with
Torch 2.14.0+cu130 and existing 2B BF16 weights. Real GPU execution remains
unverified; it needs a check on the inference PC.
The build installs the service dependencies and compiles the currently mandatory
llama-cpp-python dependency. Model weights are provided through a mount.
The initial draft caches the system-dependency layer; Python dependencies are
installed after source copying, so source changes rerun that install/compilation.

From this repository, create local settings and persistent folders:

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force data/prefix_caches, data/logs
docker compose config
docker compose up -d --build
docker compose logs -f inference
```

Keep an existing .env when updating. Edit MODELS_MOUNT, PREFIX_CACHES_MOUNT and
LOGGING_MOUNT for your machine; the example uses Neurorium's already downloaded
weights and this repository's ignored data/ folders for caches and logs.
The example and existing laptop .env select DEVICE=cpu.
Downloads are disabled in Compose, so MODELS_MOUNT must contain the approved
configuration/revision snapshot. To provision missing weights deliberately,
change the two offline settings before startup.

The container maps its three directories to /data/models,
/data/prefix_caches and /data/logs. The API is published at
http://127.0.0.1:8000. Neurorium uses the same DecisionClient URL as the native
service. Stop a native server on that port before starting the container.

After editing source, use docker compose up -d --build again. This rebuilds and
recreates the service while retaining the same mounted folders. Changing
IMAGE_TAG in .env names the local image; publishing/pulling remote images and
SSH deployment and verification on the GPU PC are separate later stages.

### GPU host configuration

After building the updated image, edit the GPU PC's .env:

```dotenv
DEVICE=auto
GPU_MEMORY_BUDGET_GB=6
```

Start the same image with GPU access requested by the override:

```powershell
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

The override changes GPU access only; image tag, storage mounts and DEVICE still
come from the main settings. The host needs working NVIDIA drivers and Docker GPU
support: see [Docker's GPU setup](https://docs.docker.com/compose/how-tos/gpu-support/)
and [Docker Desktop WSL2 GPU support](https://docs.docker.com/desktop/features/gpu/).
On Linux, configure the NVIDIA Container Toolkit. Host/driver compatibility and
real GPU inference must be verified on that PC. Auto CPU fallback cannot recover
from Docker failing to start a container because GPU access is misconfigured.

To force CPU on the GPU PC, set DEVICE=cpu and use the base Compose file without
the GPU override. Both modes use the same image and three persisted mounts.

Saved prefix keys include model configuration/revision, token IDs, dtype,
device type, cache-format version and Torch/Transformers/tokenizers versions.
Library upgrades create different keys, leaving old files intact. Changes to
cache construction/serialization must increment cache_format_version.
This conservatively invalidates version changes; it does not promise that every
hardware/kernel combination is numerically identical. Live run/state handles
expire at restart, while logs and compatible disk prefixes remain available.

Run the repeatable real-model container check with any host Python:

```powershell
python -B tests/check_container.py
```

It answers a small question, verifies exact saved context and decision data,
recreates only this service, then verifies prefix reuse, matching probabilities,
unchanged weight files and expired old run handles. It requires an already built,
running container and can take several minutes on CPU. The report is written to
data/container_check_results.json; its caches/logs remain available for review.
The 2026-10-01 CPU container check passed with existing 2B BF16 weights:
prefix reuse, exact snapshots, unchanged weights and matching probabilities
survived container recreation. Eight API/cache-identity tests and the Linux
image dependency check also passed. This is a deployment check, not workflow
golden accuracy or a GPU benchmark.

The 2026-10-01 shared-image CPU check also passed. Inside the new Linux image,
all 19 unit tests, pip check and an actual CPU BF16 calculation passed. Torch
reports CUDA 13.0 compiled support and no GPU availability on this laptop.
The real 2B model answered Red with probability 0.998647392 before and after
container recreation, reusing the new saved prefix after restart. The changed
Torch build generated a new cache key; the earlier cache remains intact.
All 32 original model files, one original prefix cache and eight original log
files retained their sizes/timestamps; cache/log bytes were also checked.
Running service source hashes match the local package. The service was left
healthy in CPU mode at localhost port 8000. Local evidence is saved under ignored
data/shared_image_verification.json and data/container_check_results.json;
the previous CPU-image report remains in data/container_check_results.cpu_image.json.
These are deployment checks, not a real GPU or workflow accuracy benchmark.

Stop inference after testing to release model memory:

```powershell
docker compose stop inference
```

## API v1

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /v1/health | Ready status and service instance ID |
| GET | /v1/model | Model settings, server identity and source hashes |
| POST | /v1/runs | Start a run with its own logger |
| POST | /v1/runs/{run_id}/states | Prepare prefix plus suffix |
| POST | /v1/runs/{run_id}/states/{state_cache_key}/questions | Answer and log one question |
| DELETE | /v1/runs/{run_id}/states/{state_cache_key} | Release one combined cache |
| DELETE | /v1/runs/{run_id} | Close the run and release its remaining states |

State requests contain prefix_text and suffix_text strings. Question requests
contain id, stem, rubric and 2–10 unique, nonempty string choices.
Unknown runs/states return 404; invalid request shapes return 422.
A state can only be queried by the run that prepared it.
All inference/cache/log mutations are serialized.

Preparation returns prefix_cache_key, state_cache_key and metadata containing
prefix_cache_reused and prepare_seconds. Question responses contain the existing
runtime answer, decision_id and metadata.inference_seconds. CUDA timing synchronizes
the configured device. HTTP elapsed time includes transfer, queueing and logging;
inference_seconds measures the model/scoring call after acquiring the service lock.

The server retains exact text for each live state. It uses the runtime's formatter
and model settings to save a completed decision before returning the response.
Logs contain exact UTF-8 prefix/suffix snapshots, state references, questions,
rendered prompt text, full probabilities, statistics and model metadata.

Closing a run preserves its logs and disk prefix caches. State handles and live
runs expire on service restart. Disk records do not resume a live run automatically.
Clients should close runs in a finally block or context manager. Automatic retries
are not provided; a lost response can leave a completed decision on disk.

## Verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

These focused tests use a test double, covering validation, run ownership,
prefix reuse metadata, exact decision snapshots, shutdown cleanup and serialized
inference, service device selection and startup OOM cleanup/retry. CUDA branches
are simulated without GPU allocation or model downloads; they do not measure
model accuracy or validate real GPU execution. Neurorium owns its workflow-specific
golden suites and client integration harness.

## Remaining deployment work

Real GPU/driver verification, remote image publication, SSH access configuration
and dependency-layer optimization remain. The shared CUDA-enabled image is built
and verified on CPU; startup device selection and the GPU Compose override are
implemented. Prefix-cache identity includes library versions.
Automatic expiry for abandoned runs, durable session resume, and GGUF scoring
are future work. The runtime and registry remain development scaffolding.
