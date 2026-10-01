"""API lifecycle, ownership, validation, logging and serialization; no model weights."""

import hashlib
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from decision_runtime.api import create_app
from decision_runtime.runtime import DecisionRuntime
from decision_runtime.service import InferenceService


class FakeRuntime:
    def __init__(self, folder):
        self.cache_dir = folder / "caches"
        self.cache_dir.mkdir()
        self._state_caches = {}
        self.calls = []
        self.active = self.max_active = 0
        self.delay = False

    def get_model_info(self):
        return {"device": "cpu", "configuration_id": "test-only", "test_double": True}

    def split_state_tokens(self, prefix, suffix):
        return list(prefix.encode()), list(suffix.encode())

    def build_prefix_cache_key(self, ids):
        return hashlib.sha256(bytes(ids)).hexdigest()

    def prepare_state_cache(self, prefix, suffix):
        prefix_key = hashlib.sha256(prefix.encode()).hexdigest()
        (self.cache_dir / f"{prefix_key}.pt").write_text("test double")
        key = uuid4().hex
        self._state_caches[key] = prefix + suffix
        return {"prefix_cache_key": prefix_key, "state_cache_key": key}

    def ask_question(self, key, question):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(0.01)
            self.calls.append((key, question["id"]))
            choices = question["choices"]
            return {
                "question_id": question["id"], "state_cache_key": key,
                "type": "choice", "choice": choices[0],
                "probabilities": {c: 1 / len(choices) for c in choices},
                "statistics": {},
            }
        finally:
            self.active -= 1

    def format_question(self, question):
        return DecisionRuntime.format_question(self, question)

    def release_state_cache(self, key):
        self._state_caches.pop(key, None)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="decision-service-tests-")
        self.root = Path(self.temporary.name)
        self.runtime = FakeRuntime(self.root)
        self.service = InferenceService(self.runtime, self.root / "logs")
        self.http = TestClient(create_app(self.service))
        self.http.__enter__()
        self.question = {"id": "color", "stem": "Which color?", "rubric": "Use the fact.", "choices": ["red", "blue"]}

    def tearDown(self):
        self.http.__exit__(None, None, None)
        self.temporary.cleanup()

    def run_id(self):
        response = self.http.post("/v1/runs")
        self.assertEqual(response.status_code, 201)
        return response.json()["run_id"]

    def state(self, run, suffix="suffix"):
        response = self.http.post(f"/v1/runs/{run}/states", json={"prefix_text": "prefix\n", "suffix_text": suffix})
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_full_record_reuse_and_cleanup(self):
        run = self.run_id()
        first = self.state(run)
        second = self.state(run, "different")
        self.assertFalse(first["metadata"]["prefix_cache_reused"])
        self.assertTrue(second["metadata"]["prefix_cache_reused"])
        self.assertNotEqual(first["state_cache_key"], second["state_cache_key"])
        key = first["state_cache_key"]
        url = f"/v1/runs/{run}/states/{key}/questions"
        result = self.http.post(url, json=self.question)
        self.assertEqual(result.status_code, 200)
        packet = result.json()
        repeated = self.http.post(url, json=self.question).json()
        self.assertNotEqual(packet["decision_id"], repeated["decision_id"])
        path = self.root / "logs" / run / "decisions" / (packet["decision_id"] + ".json")
        record = json.loads(path.read_text())
        self.assertEqual(record["answer"], packet["answer"])
        self.assertEqual(record["model_info"], self.runtime.get_model_info())
        self.assertEqual(record["question"]["rendered_text"], self.runtime.format_question(self.question))
        self.assertEqual(next((self.root / "logs" / run / "prefixes").glob("*.txt")).read_bytes(), b"prefix\n")
        self.assertEqual(next((self.root / "logs" / run / "suffixes").glob("*.txt")).read_bytes(), b"suffix")
        self.assertEqual(self.http.delete(f"/v1/runs/{run}/states/{key}").json(), {"released": True})
        self.assertIn(second["state_cache_key"], self.runtime._state_caches)
        self.assertEqual(self.http.post(url, json=self.question).status_code, 404)
        self.assertEqual(self.http.delete(f"/v1/runs/{run}").json()["released_states"], 1)
        self.assertFalse(self.runtime._state_caches)
        self.assertTrue(path.is_file())
        self.assertTrue((self.runtime.cache_dir / (first["prefix_cache_key"] + ".pt")).is_file())
        self.assertEqual(self.http.delete(f"/v1/runs/{run}").status_code, 200)

    def test_run_ownership(self):
        owner, other = self.run_id(), self.run_id()
        key = self.state(owner)["state_cache_key"]
        self.assertEqual(self.http.post(f"/v1/runs/{other}/states/{key}/questions", json=self.question).status_code, 404)
        self.assertEqual(self.http.delete(f"/v1/runs/{other}/states/{key}").json(), {"released": False})
        self.assertIn(key, self.runtime._state_caches)
        self.assertFalse(self.runtime.calls)
        self.assertEqual(self.http.post("/v1/runs/missing/states", json={"prefix_text": "x", "suffix_text": "y"}).status_code, 404)

    def test_request_validation(self):
        run = self.run_id()
        key = self.state(run)["state_cache_key"]
        url = f"/v1/runs/{run}/states/{key}/questions"
        for choices in (["x"], ["x", "x"], list(map(str, range(11))), ["x", 3]):
            self.assertEqual(self.http.post(url, json={**self.question, "choices": choices}).status_code, 422)
        self.assertEqual(self.http.post(url, json={**self.question, "extra": True}).status_code, 422)
        self.assertFalse(self.runtime.calls)

    def test_lifespan_and_identity(self):
        self.assertTrue(self.http.get("/v1/health").json()["ready"])
        details = self.http.get("/v1/model").json()
        self.assertEqual(details["model_info"], self.runtime.get_model_info())
        self.assertIn("decision_runtime/runtime.py", details["service_info"]["implementation_hashes"])
        run = self.run_id()
        self.state(run)
        with patch("decision_runtime.api.load_service", return_value=self.service) as loader:
            with TestClient(create_app()) as http:
                http.get("/v1/health")
                http.get("/v1/health")
                self.assertEqual(loader.call_count, 1)
        self.assertFalse(self.runtime._state_caches)
        self.assertFalse(self.service.runs)

    def test_inference_serialization(self):
        run = self.run_id()
        key = self.state(run)["state_cache_key"]
        self.runtime.delay = True
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda _: self.service.ask_question(run, key, self.question), range(5)))
        self.assertEqual(self.runtime.max_active, 1)
        self.assertEqual(len(self.runtime.calls), 5)


if __name__ == "__main__":
    unittest.main()
