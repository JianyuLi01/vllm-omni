# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Operator-call tracking for vllm-omni.

This module provides a lightweight tracker that records whether (and how
many times) a configurable set of low-level vLLM operators were invoked
while the model is generating a response to a prompt.

Two complementary mechanisms are used:

1. ``_TrackedTorchDispatchMode`` — a ``TorchDispatchMode`` that intercepts
   *every* ``torch.ops.*`` call.  When the unqualified op name (the part
   after ``::``) matches one of the names in :data:`TRACKED_OPS` the call
   is counted.  This catches the bulk of the operators in the user list
   because they are registered as torch custom ops (``_C::rms_norm``,
   ``_C::silu_and_mul``, ``_C::reshape_and_cache``, ``_moe_C::topk_softmax``
   …).  It runs only for the duration of one model forward pass and adds
   one Python call per dispatched op, so the overhead is negligible.

2. :func:`install_python_patches` — best-effort monkey-patches for
   functions that are *not* dispatched through ``torch.ops`` (pure-Python
   helpers such as ``weak_ref_tensor``, ``is_bmg`` / ``is_pvc`` platform
   flags, ``varlen_fwd`` from flash-attn, the ``bgmv_*`` LoRA wrappers,
   ``deepseek_scaling_rope`` …).  Each candidate location is tried and
   silently skipped if the import fails – different deployments expose
   different subsets of these helpers.

The tracker is **disabled by default**.  Set the environment variable
``VLLM_OMNI_TRACK_OPS=1`` (or pass ``enabled=True`` explicitly) to turn
it on.  When disabled, all hooks degrade to no-ops and incur no runtime
cost.

Usage from a model runner::

    from vllm_omni.profiler.op_tracker import get_op_tracker

    tracker = get_op_tracker()
    tracker.maybe_install()
    with tracker.maybe_active():
        output = self.model(...)
    if scheduler_output.finished_req_ids:
        tracker.report_and_reset(scheduler_output.finished_req_ids)
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from typing import Any

from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# The list of operators the user asked to track.  Order is preserved so the
# report keeps the same shape as the request.
# ---------------------------------------------------------------------------
TRACKED_OPS: tuple[str, ...] = (
    "weak_ref_tensor",
    "get_xpu_view_from_cpu_tensor",
    "is_bmg",
    "is_pvc",
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "mul_and_silu",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gelu_fast",
    "gelu_new",
    "gelu_quick",
    "swigluoai_and_mul",
    "rotary_embedding",
    "deepseek_scaling_rope",
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
    "per_token_group_fp8_quant",
    "convert_fp8",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "concat_and_cache_mla",
    "gather_cache",
    "swap_blocks",
    "fp8_gemm",
    "fp8_gemm_w8a16",
    "int4_gemm_w4a16",
    "int4_gemm_w4a8",
    "cutlass_grouped_gemm_interface",
    "bgmv_shrink",
    "bgmv_expand",
    "bgmv_expand_slice",
    "gdn_attention",
    "varlen_fwd",
    "moe_sum",
    "moe_align_block_size",
    "batched_moe_align_block_size",
    "moe_lora_align_block_size",
    "grouped_topk",
    "fused_grouped_topk",
    "topk_softmax",
    "topk_sigmoid",
    "moe_gather",
    "fused_moe_prologue",
)

_TRACKED_SET: frozenset[str] = frozenset(TRACKED_OPS)

# ---------------------------------------------------------------------------
# Best-effort import targets for pure-Python wrappers.  Each entry maps an
# operator name to a list of ``(module_path, attr_name)`` candidates – the
# first one that imports successfully is patched.  ``None`` means "no
# python-level wrapper expected — relies on TorchDispatchMode".
# ---------------------------------------------------------------------------
_PYTHON_PATCH_TARGETS: dict[str, list[tuple[str, str]]] = {
    "weak_ref_tensor": [
        ("vllm.utils.torch_utils", "weak_ref_tensor"),
        ("vllm.utils", "weak_ref_tensor"),
    ],
    "get_xpu_view_from_cpu_tensor": [
        ("vllm.utils.torch_utils", "get_xpu_view_from_cpu_tensor"),
        ("vllm.utils", "get_xpu_view_from_cpu_tensor"),
        ("vllm.platforms.xpu", "get_xpu_view_from_cpu_tensor"),
    ],
    "is_bmg": [
        ("vllm.platforms.xpu", "is_bmg"),
    ],
    "is_pvc": [
        ("vllm.platforms.xpu", "is_pvc"),
    ],
    "deepseek_scaling_rope": [
        ("vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope",
         "deepseek_scaling_rope"),
    ],
    "static_scaled_fp8_quant": [
        ("vllm._custom_ops", "static_scaled_fp8_quant"),
    ],
    "dynamic_scaled_fp8_quant": [
        ("vllm._custom_ops", "dynamic_scaled_fp8_quant"),
    ],
    "dynamic_per_token_scaled_fp8_quant": [
        ("vllm._custom_ops", "dynamic_per_token_scaled_fp8_quant"),
    ],
    "per_token_group_fp8_quant": [
        ("vllm.model_executor.layers.quantization.utils.fp8_utils",
         "per_token_group_quant_fp8"),
        ("vllm._custom_ops", "per_token_group_fp8_quant"),
    ],
    "gather_cache": [
        ("vllm._custom_ops", "gather_cache"),
        ("vllm._custom_ops", "cp_gather_cache"),
    ],
    "fp8_gemm": [
        ("vllm.model_executor.layers.quantization.utils.fp8_utils", "fp8_gemm"),
    ],
    "fp8_gemm_w8a16": [
        ("vllm.model_executor.layers.quantization.utils.fp8_utils",
         "fp8_gemm_w8a16"),
    ],
    "int4_gemm_w4a16": [
        ("vllm.model_executor.layers.quantization.utils.int4_utils",
         "int4_gemm_w4a16"),
    ],
    "int4_gemm_w4a8": [
        ("vllm.model_executor.layers.quantization.utils.int4_utils",
         "int4_gemm_w4a8"),
    ],
    "cutlass_grouped_gemm_interface": [
        ("vllm._custom_ops", "cutlass_grouped_gemm_interface"),
    ],
    "bgmv_shrink": [
        ("vllm.lora.ops.triton_ops.bgmv_shrink", "bgmv_shrink"),
        ("vllm.lora.ops.triton_ops", "bgmv_shrink"),
    ],
    "bgmv_expand": [
        ("vllm.lora.ops.triton_ops.bgmv_expand", "bgmv_expand"),
        ("vllm.lora.ops.triton_ops", "bgmv_expand"),
    ],
    "bgmv_expand_slice": [
        ("vllm.lora.ops.triton_ops.bgmv_expand_slice", "bgmv_expand_slice"),
        ("vllm.lora.ops.triton_ops", "bgmv_expand_slice"),
    ],
    "gdn_attention": [
        ("vllm.attention.ops.gdn_attention", "gdn_attention"),
    ],
    "varlen_fwd": [
        ("flash_attn.flash_attn_interface", "_flash_attn_varlen_forward"),
        ("flash_attn", "flash_attn_varlen_func"),
        ("vllm_flash_attn.flash_attn_interface", "_flash_attn_varlen_forward"),
        ("vllm_flash_attn", "flash_attn_varlen_func"),
    ],
    "fused_grouped_topk": [
        ("vllm.model_executor.layers.fused_moe.fused_moe", "fused_grouped_topk"),
        ("vllm.model_executor.layers.fused_moe", "fused_grouped_topk"),
    ],
    "moe_gather": [
        ("vllm.model_executor.layers.fused_moe.fused_moe", "moe_gather"),
        ("vllm.model_executor.layers.fused_moe", "moe_gather"),
    ],
    "fused_moe_prologue": [
        ("vllm.model_executor.layers.fused_moe.fused_moe",
         "fused_moe_prologue"),
        ("vllm.model_executor.layers.fused_moe", "fused_moe_prologue"),
    ],
}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class _TrackedTorchDispatchMode(TorchDispatchMode):
    """TorchDispatchMode that increments the tracker for matching ops."""

    def __init__(self, tracker: OpCallTracker) -> None:
        super().__init__()
        self._tracker = tracker

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):  # type: ignore[override]
        # ``func`` is an OpOverload; ``func._opname`` is the unqualified
        # op name, e.g. "rms_norm" for ``_C::rms_norm``.  Older torch
        # versions may not expose ``_opname``; fall back to splitting the
        # full string representation.
        name = getattr(func, "_opname", None)
        if name is None:
            full = str(getattr(func, "name", lambda: "")())
            if "::" in full:
                # Strip the overload (".default") if present.
                name = full.split("::", 1)[1].split(".", 1)[0]
            else:
                name = full
        if name in _TRACKED_SET:
            self._tracker._record(name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class OpCallTracker:
    """Process-local tracker for the operators in :data:`TRACKED_OPS`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {name: 0 for name in TRACKED_OPS}
        self._dispatch_mode: _TrackedTorchDispatchMode | None = None
        self._installed = False
        self._enabled = _env_truthy("VLLM_OMNI_TRACK_OPS")
        self._patched: list[tuple[Any, str, Callable]] = []

    # ------------------------------------------------------------------ API

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, value: bool = True) -> None:
        self._enabled = value

    def _record(self, name: str) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1

    def maybe_install(self) -> None:
        """Install python-level patches once.  Safe to call repeatedly."""
        if not self._enabled or self._installed:
            return
        self._installed = True
        self._dispatch_mode = _TrackedTorchDispatchMode(self)
        installed = self._install_python_patches()
        logger.info(
            "[OpTracker] enabled — tracking %d operators "
            "(python wrappers patched: %d).",
            len(TRACKED_OPS),
            installed,
        )

    def maybe_active(self):
        """Context manager that activates the dispatch mode for the
        duration of a forward pass.  Returns a no-op context if the
        tracker is disabled."""
        if not self._enabled or self._dispatch_mode is None:
            return nullcontext()
        return self._dispatch_mode

    @contextmanager
    def force_active(self):
        """Always-active context manager (used in tests)."""
        if self._dispatch_mode is None:
            self._dispatch_mode = _TrackedTorchDispatchMode(self)
        with self._dispatch_mode:
            yield

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            for name in self._counts:
                self._counts[name] = 0

    def report_and_reset(
        self,
        finished_req_ids: Iterable[str] | None = None,
        label: str | None = None,
    ) -> dict[str, int]:
        """Log a single-line summary and reset all counters.

        Returns the snapshot that was logged (useful in tests).
        """
        snap = self.snapshot()
        if not self._enabled:
            return snap

        called = [(n, c) for n, c in snap.items() if c > 0]
        not_called = [n for n, c in snap.items() if c == 0]
        called.sort(key=lambda kv: TRACKED_OPS.index(kv[0]))

        header = "[OpTracker]"
        if label:
            header += f" {label}"
        if finished_req_ids:
            ids = list(finished_req_ids)
            header += f" finished_req_ids={ids}"

        if called:
            called_str = ", ".join(f"{n}={c}" for n, c in called)
        else:
            called_str = "<none>"
        logger.info("%s called: %s", header, called_str)
        logger.info("%s not_called (%d): %s",
                    header, len(not_called), not_called)

        self.reset()
        return snap

    # ------------------------------------------------- python-side patches

    def _install_python_patches(self) -> int:
        """Best-effort monkey-patch of pure-Python wrappers.

        Returns the number of wrappers that were successfully patched.
        """
        installed = 0
        for op_name, candidates in _PYTHON_PATCH_TARGETS.items():
            for module_path, attr in candidates:
                try:
                    module = importlib.import_module(module_path)
                except Exception:
                    continue
                fn = getattr(module, attr, None)
                if fn is None or not callable(fn):
                    continue
                # Already wrapped?  Skip.
                if getattr(fn, "_vllm_omni_op_tracked", False):
                    installed += 1
                    break
                wrapped = self._make_wrapper(op_name, fn)
                wrapped._vllm_omni_op_tracked = True  # type: ignore[attr-defined]
                try:
                    setattr(module, attr, wrapped)
                    self._patched.append((module, attr, fn))
                    installed += 1
                except Exception:
                    continue
                break  # first successful candidate wins
        return installed

    def _make_wrapper(self, op_name: str, fn: Callable) -> Callable:
        tracker = self

        def _wrapped(*args, **kwargs):
            tracker._record(op_name)
            return fn(*args, **kwargs)

        _wrapped.__name__ = getattr(fn, "__name__", op_name)
        _wrapped.__qualname__ = getattr(fn, "__qualname__", op_name)
        _wrapped.__doc__ = getattr(fn, "__doc__", None)
        _wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
        return _wrapped


# Singleton accessor -------------------------------------------------------

_TRACKER: OpCallTracker | None = None
_TRACKER_LOCK = threading.Lock()


def get_op_tracker() -> OpCallTracker:
    global _TRACKER
    if _TRACKER is None:
        with _TRACKER_LOCK:
            if _TRACKER is None:
                _TRACKER = OpCallTracker()
    return _TRACKER
