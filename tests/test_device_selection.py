"""Startup selection/fallback checks with simulated CUDA; no model downloads."""

import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from decision_runtime import api
from decision_runtime.device_selection import select_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_does_not_probe_cuda(self):
        with patch("decision_runtime.device_selection.torch.cuda.is_available") as available:
            device, reason = select_device("cpu", torch.bfloat16, None)
        self.assertEqual(device, "cpu")
        self.assertIn("explicitly", reason)
        available.assert_not_called()

    def test_missing_budget_falls_back_or_fails_before_probing(self):
        with patch("decision_runtime.device_selection.torch.cuda.is_available") as available:
            self.assertEqual(select_device("auto", torch.bfloat16, None)[0], "cpu")
            with self.assertRaisesRegex(RuntimeError, "not been configured"):
                select_device("cuda", torch.bfloat16, None)
        available.assert_not_called()

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            select_device("gpu", torch.bfloat16, 6)
        for budget in (0, -1, float("nan"), float("inf")):
            for mode in ("auto", "cuda"):
                with self.subTest(budget=budget, mode=mode), self.assertRaises(ValueError):
                    select_device(mode, torch.bfloat16, budget)

    def test_cuda_unavailable_falls_back_or_fails(self):
        with patch("decision_runtime.device_selection.torch.cuda.is_available", return_value=False):
            device, reason = select_device("auto", torch.bfloat16, 6)
            self.assertEqual(device, "cpu")
            self.assertIn("unavailable", reason)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                select_device("cuda", torch.bfloat16, 6)

    def test_free_memory_budget_and_dtype_probe(self):
        for free_gb in (5, 6, 7):
            for mode in ("auto", "cuda"):
                with self.subTest(free_gb=free_gb, mode=mode), ExitStack() as stack:
                    stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.is_available", return_value=True))
                    ones = stack.enter_context(patch("decision_runtime.device_selection.torch.ones"))
                    sync = stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.synchronize"))
                    memory = stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.mem_get_info", return_value=(free_gb * 1_000_000_000, 8_000_000_000)))
                    if free_gb < 6 and mode == "cuda":
                        with self.assertRaisesRegex(RuntimeError, "budget requires"):
                            select_device(mode, torch.bfloat16, 6)
                    else:
                        device, _ = select_device(mode, torch.bfloat16, 6)
                        self.assertEqual(device, "cuda" if free_gb >= 6 else "cpu")
                    ones.assert_called_once_with((16, 16), device="cuda", dtype=torch.bfloat16)
                    ones.return_value.__matmul__.assert_called_once_with(ones.return_value)
                    sync.assert_called_once()
                    memory.assert_called_once()

    def test_probe_and_memory_errors_fall_back_only_in_auto(self):
        for failing_call in ("torch.ones", "torch.cuda.synchronize", "torch.cuda.mem_get_info"):
            for mode in ("auto", "cuda"):
                with self.subTest(call=failing_call, mode=mode), ExitStack() as stack:
                    stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.is_available", return_value=True))
                    stack.enter_context(patch("decision_runtime.device_selection.torch.ones"))
                    stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.synchronize"))
                    stack.enter_context(patch("decision_runtime.device_selection.torch.cuda.mem_get_info", return_value=(7_000_000_000, 8_000_000_000)))
                    stack.enter_context(patch("decision_runtime.device_selection." + failing_call, side_effect=RuntimeError("simulated unsupported operation")))
                    if mode == "cuda":
                        with self.assertRaisesRegex(RuntimeError, "simulated"):
                            select_device(mode, torch.bfloat16, 6)
                    else:
                        device, reason = select_device(mode, torch.bfloat16, 6)
                        self.assertEqual(device, "cpu")
                        self.assertIn("simulated", reason)


class ServiceStartupTests(unittest.TestCase):
    def startup_patches(self, requested="auto", selected="cuda", budget="6"):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {
            "DEVICE": requested, "GPU_MEMORY_BUDGET_GB": budget,
            "TORCH_THREADS": "4", "DATA_DIR": "test-data",
        }, clear=True))
        stack.enter_context(patch("decision_runtime.api.torch.set_num_threads"))
        runtime = stack.enter_context(patch("decision_runtime.api.DecisionRuntime")).return_value
        runtime.approved_configuration = {"artifact_format": "safetensors", "dtype": "bfloat16"}
        runtime.model = object()
        selector = stack.enter_context(patch("decision_runtime.api.select_device", return_value=(selected, "simulated selection")))
        service = stack.enter_context(patch("decision_runtime.api.InferenceService"))
        collect = stack.enter_context(patch("decision_runtime.api.gc.collect"))
        empty = stack.enter_context(patch("decision_runtime.api.torch.cuda.empty_cache"))
        stack.enter_context(patch("builtins.print"))
        return runtime, selector, service, collect, empty

    def test_selected_device_is_used_for_loading(self):
        runtime, selector, service, collect, empty = self.startup_patches()
        seen = []
        runtime.load_model.side_effect = lambda: seen.append(runtime.device)
        self.assertIs(api.load_service(), service.return_value)
        self.assertEqual(seen, ["cuda"])
        selector.assert_called_once_with("auto", torch.bfloat16, 6.0)
        self.assertIs(service.call_args.args[0], runtime)
        collect.assert_not_called()
        empty.assert_not_called()

    def test_auto_gpu_oom_cleans_up_and_reloads_cpu_outside_handler(self):
        runtime, _, service, collect, empty = self.startup_patches()
        seen = []

        def load():
            seen.append(runtime.device)
            if len(seen) == 1:
                raise torch.cuda.OutOfMemoryError("simulated model allocation")
            self.assertIsNone(sys.exception())
            self.assertIsNone(runtime.model)
            runtime.model = "fresh CPU model"

        def cleanup():
            self.assertIsNone(sys.exception())
            self.assertIsNone(runtime.model)

        runtime.load_model.side_effect = load
        collect.side_effect = cleanup
        empty.side_effect = cleanup
        self.assertIs(api.load_service(), service.return_value)
        self.assertEqual(seen, ["cuda", "cpu"])
        self.assertEqual(runtime.model, "fresh CPU model")
        collect.assert_called_once()
        empty.assert_called_once()

    def test_failures_outside_auto_gpu_oom_propagate_without_retry(self):
        cases = (
            ("cuda", "cuda", torch.cuda.OutOfMemoryError("forced GPU OOM")),
            ("cpu", "cpu", torch.cuda.OutOfMemoryError("CPU load failure")),
            ("auto", "cpu", torch.cuda.OutOfMemoryError("already CPU")),
            ("auto", "cuda", RuntimeError("corrupt model")),
        )
        for mode, selected, error in cases:
            with self.subTest(mode=mode, selected=selected):
                runtime, _, service, collect, empty = self.startup_patches(mode, selected)
                runtime.load_model.side_effect = error
                with self.assertRaises(type(error)):
                    api.load_service()
                runtime.load_model.assert_called_once()
                service.assert_not_called()
                collect.assert_not_called()
                empty.assert_not_called()

    def test_cpu_retry_failure_propagates(self):
        runtime, _, service, _, _ = self.startup_patches()
        runtime.load_model.side_effect = [torch.cuda.OutOfMemoryError("GPU OOM"), RuntimeError("CPU failed too")]
        with self.assertRaisesRegex(RuntimeError, "CPU failed too"):
            api.load_service()
        self.assertEqual(runtime.load_model.call_count, 2)
        self.assertEqual(runtime.device, "cpu")
        service.assert_not_called()

    def test_service_without_budget_selects_cpu_using_real_policy(self):
        runtime, selector, _, collect, empty = self.startup_patches(budget="")
        selector.side_effect = select_device
        seen = []
        runtime.load_model.side_effect = lambda: seen.append(runtime.device)
        api.load_service()
        self.assertEqual(seen, ["cpu"])
        selector.assert_called_once_with("auto", torch.bfloat16, None)
        collect.assert_not_called()
        empty.assert_not_called()


if __name__ == "__main__":
    unittest.main()
