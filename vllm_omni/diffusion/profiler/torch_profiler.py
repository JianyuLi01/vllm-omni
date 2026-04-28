# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from contextlib import nullcontext

from torch.profiler import ProfilerActivity, profile
from vllm.logger import init_logger

from .base import ProfilerBase

logger = init_logger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Trace export is performed explicitly in stop() (not via on_trace_ready),
    so no per-step prof.step() calls are required from the diffusion loop.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.
    """

    _profiler: profile | None = None
    _trace_template: str = ""

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        """
        Start the profiler with the given trace path template.
        """
        # 1. Cleanup any existing profiler
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            try:
                cls._profiler.stop()
            except Exception as e:
                logger.warning("[Rank %s] Failed to stop existing profiler: %s", cls._get_rank(), e)
            cls._profiler = None

        rank = cls._get_rank()

        # 2. Make path absolute
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template

        # Expected paths
        json_file = f"{trace_path_template}_rank{rank}.json"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        logger.info(f"[Rank {rank}] Starting End-to-End Torch profiler")

        # 3. Initialize profiler in continuous (no-schedule) mode.
        #
        # Note: We intentionally do NOT use torch.profiler.schedule() with
        # on_trace_ready here. That callback only fires after the schedule's
        # active period completes, which requires explicit prof.step() calls
        # from the diffusion loop. Those step() calls do not currently exist
        # in this codebase, so relying on on_trace_ready would silently drop
        # the trace (no .json / .json.gz produced) even though stop() reported
        # success. Doing the export explicitly inside stop() guarantees the
        # trace is written.
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        # 4. Start profiling
        cls._profiler.start()

        # Return the expected final path
        return f"{trace_path_template}_rank{rank}.json.gz"

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()

        # Determine expected paths
        base_path = f"{cls._trace_template}_rank{rank}"
        json_path = f"{base_path}.json"
        gz_path = f"{base_path}.json.gz"

        prof = cls._profiler
        cls._profiler = None

        try:
            prof.stop()
        except Exception as e:
            logger.warning(f"[Rank {rank}] Profiler stop failed: {e}")
            return {"trace": None, "table": None}

        # Export the chrome trace explicitly. With no schedule/on_trace_ready,
        # nothing else will write the trace file for us. The output directory
        # was already created in start().
        try:
            prof.export_chrome_trace(json_path)
            logger.info(f"[Rank {rank}] Trace exported to {json_path}")
        except Exception as e:
            logger.warning(f"[Rank {rank}] Failed to export trace: {e}")
            return {"trace": None, "table": None}

        # Compress to .json.gz in the background. If gzip is unavailable,
        # fall back to returning the uncompressed .json path so the caller
        # still gets a real file on disk.
        try:
            subprocess.Popen(["gzip", "-f", json_path])
            logger.info(f"[Rank {rank}] Triggered background compression for {json_path}")
            return {"trace": gz_path, "table": None}
        except Exception as compress_err:
            logger.warning(
                f"[Rank {rank}] Background gzip failed to start: {compress_err}; "
                f"returning uncompressed trace path"
            )
            return {"trace": json_path, "table": None}

    @classmethod
    def step(cls):
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()
