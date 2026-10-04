"""Manual real-container persistence check. Run after docker compose up -d --build.

Uses the existing offline 2B weights. Recreates only this Compose service.
Writes a report under ignored data/. Logs and prefix caches remain for inspection.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
PREFIX = "Context:\nPorco Rosso flies a red aircraft.\n"
SUFFIX = "Today he leaves Genoa.\n"
QUESTION = {
    "id": "aircraft-color",
    "type": "choice",
    "instructions": "Use the stated facts. What color is Porco Rosso's aircraft?",
    "criteria": {"RED": "The aircraft is red.", "BLUE": "The aircraft is blue."},
}


def compose(*args):
    result = subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return result.stdout


def request(method, path, body=None):
    content = json.dumps(body).encode() if body is not None else None
    req = Request(
        BASE_URL + path, data=content, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=900) as response:
        return json.load(response)


def wait_ready():
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            if request("GET", "/v1/health")["ready"]:
                return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("Service was not ready after 15 minutes; inspect docker compose logs.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ask_once(close=True):
    run = request("POST", "/v1/runs")
    path = f"/v1/runs/{run['run_id']}"
    completed = False
    try:
        prepared = request("POST", path + "/states", {
            "prefix_text": PREFIX, "suffix_text": SUFFIX,
        })
        packet = request(
            "POST", path + f"/states/{prepared['state_cache_key']}/questions",
            QUESTION,
        )
        assert packet["answer"]["choice"] == "RED", packet
        completed = True
        return {
            **run, **prepared,
            "decision_id": packet["decision_id"],
            "answer": packet["answer"],
            "question_metadata": packet["metadata"],
        }
    finally:
        if close or not completed:
            request("DELETE", path)


def main():
    config = json.loads(compose("config", "--format", "json"))
    mounts = {
        item["target"]: Path(item["source"])
        for item in config["services"]["inference"]["volumes"]
    }
    models = mounts["/data/models"]
    caches = mounts["/data/prefix_caches"]
    logs = mounts["/data/logs"]
    weights_before = {
        str(path.relative_to(models)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in models.rglob("*") if path.is_file()
    }
    assert weights_before, "Models mount is empty."
    print("Waiting for first container startup...", flush=True)
    wait_ready()
    model_before = request("GET", "/v1/model")
    print("Preparing state and asking the first question...", flush=True)
    first = ask_once(close=False)
    cache_extension = ".npz" if model_before["model_info"]["artifact_format"] == "gguf" else ".pt"
    cache_file = caches / f"{first['prefix_cache_key']}{cache_extension}"
    cache_before = digest(cache_file)
    decision_path = logs / first["run_id"] / "decisions" / f"{first['decision_id']}.json"
    record_before = digest(decision_path)
    record = json.loads(decision_path.read_text(encoding="utf-8"))
    assert record["answer"] == first["answer"]
    assert record["model_info"] == model_before["model_info"]
    assert {key: record["question"][key] for key in QUESTION} == QUESTION
    state_record = json.loads(
        (logs / first["run_id"] / "states" / f"{record['state_id']}.json").read_text()
    )
    assert (logs / first["run_id"] / "prefixes" / f"{state_record['prefix_id']}.txt").read_bytes() == PREFIX.encode()
    assert (logs / first["run_id"] / "suffixes" / f"{state_record['suffix_id']}.txt").read_bytes() == SUFFIX.encode()

    print("Recreating the container with the same mounts...", flush=True)
    compose("up", "-d", "--force-recreate", "--no-build", "inference")
    wait_ready()
    model_after = request("GET", "/v1/model")
    assert model_before["service_info"]["instance_id"] != model_after["service_info"]["instance_id"]
    assert model_before["model_info"] == model_after["model_info"]
    assert cache_before == digest(cache_file)
    assert record_before == digest(decision_path)
    print("Preparing the same state from the saved prefix cache...", flush=True)
    second = ask_once()
    assert second["metadata"]["prefix_cache_reused"] is True
    assert first["prefix_cache_key"] == second["prefix_cache_key"]
    assert set(first["answer"]["probabilities"]) == set(QUESTION["criteria"])
    assert set(second["answer"]["probabilities"]) == set(QUESTION["criteria"])
    weights_after = {
        str(path.relative_to(models)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in models.rglob("*") if path.is_file()
    }
    assert weights_before == weights_after
    assert cache_before == digest(cache_file)
    try:
        request("POST", f"/v1/runs/{first['run_id']}/states", {
            "prefix_text": PREFIX, "suffix_text": SUFFIX,
        })
    except HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("An old live run survived the service restart.")
    report = {
        "container_recreated": True, "weights_unchanged": True,
        "saved_prefix_unchanged": True, "saved_decision_unchanged": True,
        "prefix_reused_after_restart": True, "keyed_answers_returned": True,
        "old_run_expired": True, "first": first, "second": second,
        "model_info": model_after["model_info"],
    }
    target = ROOT / "data/container_check_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
