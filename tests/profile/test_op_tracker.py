# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm_omni.profiler.op_tracker``."""

from __future__ import annotations

import logging

import pytest
import torch

from vllm_omni.profiler.op_tracker import (
    _PYTHON_PATCH_TARGETS,
    TRACKED_OPS,
    OpCallTracker,
)


@pytest.fixture()
def tracker(monkeypatch):
    """Return a fresh tracker with the env flag honored as enabled."""
    # Force-enable for the duration of the test, regardless of env.
    t = OpCallTracker()
    t.enable(True)
    return t


def test_tracked_ops_have_no_duplicates():
    assert len(TRACKED_OPS) == len(set(TRACKED_OPS))


def test_python_patch_targets_only_reference_known_ops():
    extra = set(_PYTHON_PATCH_TARGETS) - set(TRACKED_OPS)
    assert not extra, f"Patch table references unknown ops: {extra}"


def test_disabled_tracker_is_no_op():
    t = OpCallTracker()  # honors env var; default unset → disabled
    t.enable(False)
    t.maybe_install()
    # When disabled, dispatch mode is not created and counters never move.
    with t.maybe_active():
        x = torch.zeros(2, 2)
        _ = x + 1
    assert all(v == 0 for v in t.snapshot().values())


def test_dispatch_mode_counts_matching_torch_ops(tracker):
    """A torch op whose unqualified name is in the tracked set should be
    counted exactly once per call.  We register a synthetic op named
    ``rms_norm`` in a private namespace so we don't depend on vLLM's CUDA
    kernels being available."""
    lib = torch.library.Library("vllm_omni_test", "DEF")
    try:
        lib.define("rms_norm(Tensor x) -> Tensor")
        lib.impl("rms_norm", lambda x: x + 0.0, "CompositeExplicitAutograd")
        op = torch.ops.vllm_omni_test.rms_norm

        with tracker.force_active():
            op(torch.zeros(3))
            op(torch.zeros(3))
            # An op whose name is *not* in TRACKED_OPS must not be counted.
            torch.add(torch.zeros(2), torch.zeros(2))

        snap = tracker.snapshot()
        assert snap["rms_norm"] == 2
        # Sanity: a different tracked op was never invoked.
        assert snap["silu_and_mul"] == 0
    finally:
        del lib


def test_python_wrapper_patching_counts_calls(tracker, monkeypatch):
    """``install_python_patches`` should wrap an existing python wrapper
    so that calling it increments the counter."""
    # Build a fake module exposing one of the tracked wrappers.
    import sys
    import types

    fake_mod = types.ModuleType("vllm_omni_fake_quant")

    def per_token_group_quant_fp8(x):
        return x

    fake_mod.per_token_group_quant_fp8 = per_token_group_quant_fp8
    monkeypatch.setitem(sys.modules, "vllm_omni_fake_quant", fake_mod)
    # Redirect the patch table to point at our fake module instead of
    # the real (possibly missing) vllm location.
    monkeypatch.setitem(
        _PYTHON_PATCH_TARGETS,
        "per_token_group_fp8_quant",
        [("vllm_omni_fake_quant", "per_token_group_quant_fp8")],
    )

    tracker.maybe_install()

    fake_mod.per_token_group_quant_fp8(torch.zeros(1))
    fake_mod.per_token_group_quant_fp8(torch.zeros(1))

    assert tracker.snapshot()["per_token_group_fp8_quant"] == 2


def test_report_and_reset_logs_and_clears(tracker):
    tracker._record("rms_norm")
    tracker._record("rms_norm")
    tracker._record("silu_and_mul")

    # Attach a list handler directly to the module logger so we are
    # independent of vllm/vllm_omni logger propagation settings.
    from vllm_omni.profiler import op_tracker as ot_mod

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    ot_mod.logger.addHandler(handler)
    prev_level = ot_mod.logger.level
    ot_mod.logger.setLevel(logging.INFO)
    try:
        snap = tracker.report_and_reset(
            finished_req_ids=["req-1"], label="unit-test"
        )
    finally:
        ot_mod.logger.removeHandler(handler)
        ot_mod.logger.setLevel(prev_level)

    assert snap["rms_norm"] == 2
    assert snap["silu_and_mul"] == 1
    # All counters reset to zero.
    assert all(v == 0 for v in tracker.snapshot().values())

    messages = "\n".join(r.getMessage() for r in records)
    assert "rms_norm=2" in messages
    assert "silu_and_mul=1" in messages
    assert "req-1" in messages
    assert "not_called" in messages


def test_report_and_reset_no_op_when_disabled():
    t = OpCallTracker()
    t.enable(False)
    t._record("rms_norm")  # would normally bump counter
    snap = t.report_and_reset(["r"])
    # Returned snapshot reflects current state but counters are not reset.
    assert snap["rms_norm"] == 1
    assert t.snapshot()["rms_norm"] == 1
