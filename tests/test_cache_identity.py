"""Cache reuse and upgrade invalidation checks without loading model weights."""

import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch

from decision_runtime.runtime import DecisionRuntime


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runtime = DecisionRuntime(
            "decider_2b_bfloat16_safetensors",
            device="cpu",
            cache_dir=Path(self.folder.name) / "caches",
            models_dir=Path(self.folder.name) / "models",
        )
        self.runtime.model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        self.ids = [1, 20, 300]

    def test_same_identity_survives_storage_relocation(self):
        original = self.runtime.build_prefix_cache_key(self.ids)
        self.runtime.cache_dir = Path(self.folder.name) / "other"
        self.runtime.models_dir = Path(self.folder.name) / "other-models"
        self.assertEqual(original, self.runtime.build_prefix_cache_key(list(self.ids)))

    def test_library_upgrades_invalidate_key(self):
        original = self.runtime.build_prefix_cache_key(self.ids)
        with patch("decision_runtime.runtime.torch.__version__", "test-new-torch"):
            self.assertNotEqual(original, self.runtime.build_prefix_cache_key(self.ids))
        for library in ("transformers", "tokenizers"):
            with self.subTest(library=library), patch(
                "decision_runtime.runtime.version",
                side_effect=lambda name: "test-new-version" if name == library else version(name),
            ):
                self.assertNotEqual(original, self.runtime.build_prefix_cache_key(self.ids))
        self.assertEqual(original, self.runtime.build_prefix_cache_key(self.ids))

    def test_revision_and_token_changes_invalidate_key(self):
        original = self.runtime.build_prefix_cache_key(self.ids)
        self.assertNotEqual(original, self.runtime.build_prefix_cache_key(self.ids + [4]))
        self.runtime.approved_configuration = {
            **self.runtime.approved_configuration, "revision": "different-revision",
        }
        self.assertNotEqual(original, self.runtime.build_prefix_cache_key(self.ids))


if __name__ == "__main__":
    unittest.main()
