#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare cuDNN Frontend NWH causal-conv1d with attention-gym PR 266.

The default timer is CUDAGym's cold-L2 CUDA-event methodology.  A small
standalone event-timer fallback keeps the PoC runnable when CUDAGym is not
installed, while clearly labelling the changed methodology in the JSON output.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from attn_gym.linear.kda.short_conv.cute import (
    _backward_custom_op as attention_gym_backward,
)
from attn_gym.linear.kda.short_conv.cute import cute_causal_conv1d_silu
from cudnn.ops.causal_conv1d import causal_conv1d_nwh

ATTENTION_GYM_PR266_SHA = "f61ee082ac34b433c386b0980604cd03972f3be3"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        batch, tokens = value.lower().split("x", 1)
        parsed = (int(batch), int(tokens))
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("shape must be BxT, for example 1x16384") from error
    if parsed[0] < 1 or parsed[1] < 1:
        raise argparse.ArgumentTypeError("B and T must both be positive")
    return parsed


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.mean(times_ms) * 1000.0,
        "median_us": statistics.median(times_ms) * 1000.0,
        "min_us": min(times_ms) * 1000.0,
        "p20_us": _percentile(times_ms, 0.20) * 1000.0,
        "max_us": max(times_ms) * 1000.0,
    }


def _event_times(
    fn: Callable[..., Any],
    args: list[Any],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Standalone cold-L2 approximation for users without CUDAGym.

    The reported NVIDIA measurements use CUDAGym, not this fallback.  This
    path keeps the public PoC easy to run while retaining CUDA-event timing
    and a 256 MiB cache flush between invocations.
    """

    cache = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    torch.cuda.synchronize()
    for _ in range(warmup):
        cache.zero_()
        fn(*args)
    for index in range(iterations):
        cache.zero_()
        starts[index].record()
        fn(*args)
        ends[index].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _measure(
    fn: Callable[..., Any],
    args: list[Any],
    *,
    timer: str,
    warmup: int,
    iterations: int,
) -> tuple[str, list[float], dict[str, Any] | None]:
    if timer != "events":
        try:
            from cudagym.bench.profiling.timing import time_runnable
        except ImportError:
            if timer == "cudagym":
                raise
        else:
            result = time_runnable(
                fn=fn,
                inputs=args,
                outputs=[],
                device="cuda:0",
                warmup=warmup,
                rep=iterations,
                summarization_statistic="all",
                methodology="cuda_events",
            )
            return (
                "cudagym_cuda_events_cold_l2_shifting_inputs",
                result.measured_times,
                result.telemetry.model_dump(mode="json"),
            )
    return (
        "standalone_cuda_events_cold_l2_static_inputs",
        _event_times(fn, args, warmup=warmup, iterations=iterations),
        None,
    )


def _nvidia_smi_state() -> dict[str, str] | None:
    fields = [
        "index",
        "uuid",
        "pci.bus_id",
        "clocks.current.graphics",
        "clocks.applications.graphics",
        "clocks.max.graphics",
        "clocks.current.memory",
        "clocks.applications.memory",
        "clocks.max.memory",
        "power.limit",
        "persistence_mode",
        "driver_version",
    ]
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [value.strip() for value in line.split(",")]
    return dict(zip(fields, values, strict=True))


def _cudnn_forward(x: torch.Tensor, weight_kd: torch.Tensor, bias: torch.Tensor):
    return causal_conv1d_nwh(x, weight_kd, bias, activation="silu")


def _cudnn_backward(
    x: torch.Tensor,
    weight_kd: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
):
    return torch.ops.cudnn.causal_conv1d_nwh_bwd_primitive(grad_output, x, weight_kd, bias, "silu")


def _attention_forward(x: torch.Tensor, weight_dk: torch.Tensor):
    return cute_causal_conv1d_silu(x, weight_dk)


def _attention_backward(
    x: torch.Tensor,
    weight_dk: torch.Tensor,
    grad_output: torch.Tensor,
):
    return attention_gym_backward(x, weight_dk, grad_output)


def _kernel_names(fn: Callable[..., Any], args: list[Any]) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        fn(*args)
        torch.cuda.synchronize()
    names = {event.name for event in profile.events() if "cuda" in str(getattr(event, "device_type", "")).lower()}
    return sorted(names)


def _correctness(
    x: torch.Tensor,
    weight_kd: torch.Tensor,
    weight_dk: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, float]:
    cudnn_y = _cudnn_forward(x, weight_kd, bias)
    attention_y = _attention_forward(x, weight_dk)
    cudnn_dx, cudnn_dw, _ = _cudnn_backward(x, weight_kd, bias, grad_output)
    attention_dx, attention_dw = _attention_backward(x, weight_dk, grad_output)
    torch.cuda.synchronize()
    return {
        "forward_max_abs": float((cudnn_y - attention_y).abs().max()),
        "dx_max_abs": float((cudnn_dx - attention_dx).abs().max()),
        "dw_max_abs": float((cudnn_dw.T - attention_dw).abs().max()),
    }


def _benchmark_shape(
    batch: int,
    tokens: int,
    *,
    channels: int,
    width: int,
    timer: str,
    warmup: int,
    iterations: int,
    check_correctness: bool,
    print_kernels: bool,
) -> dict[str, Any]:
    torch.manual_seed(0)
    x = (torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight_dk = (torch.randn(channels, width, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight_kd = weight_dk.T.contiguous()
    bias = torch.zeros(channels, device="cuda", dtype=torch.bfloat16)
    grad_output = (torch.randn_like(x) * 0.2).contiguous()

    calls = {
        "cudnn_fwd": (_cudnn_forward, [x, weight_kd, bias]),
        "attention_gym_fwd": (_attention_forward, [x, weight_dk]),
        "cudnn_bwd": (_cudnn_backward, [x, weight_kd, bias, grad_output]),
        "attention_gym_bwd": (_attention_backward, [x, weight_dk, grad_output]),
    }

    # Compile both JIT paths before correctness checks and timed runs.
    for fn, args in calls.values():
        fn(*args)
    torch.cuda.synchronize()

    record: dict[str, Any] = {
        "shape": {"batch": batch, "tokens": tokens, "channels": channels, "width": width},
        "dtype": "bfloat16",
        "activation": "silu",
        "layout": "NWH/channel-last",
        "timings": {},
    }
    if check_correctness:
        record["cross_implementation_max_abs"] = _correctness(x, weight_kd, weight_dk, bias, grad_output)
    if print_kernels:
        record["kernels"] = {name: _kernel_names(fn, args) for name, (fn, args) in calls.items()}

    active_timer = None
    for name, (fn, args) in calls.items():
        active_timer, times, telemetry = _measure(
            fn,
            args,
            timer=timer,
            warmup=warmup,
            iterations=iterations,
        )
        record["timings"][name] = {
            **_summary(times),
            "gpu_telemetry": telemetry,
        }
        torch.cuda.empty_cache()
    record["timer"] = active_timer
    record["speedup_attention_over_cudnn"] = {
        "fwd_mean": (record["timings"]["attention_gym_fwd"]["mean_us"] / record["timings"]["cudnn_fwd"]["mean_us"]),
        "bwd_mean": (record["timings"]["attention_gym_bwd"]["mean_us"] / record["timings"]["cudnn_bwd"]["mean_us"]),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=None,
        help="B x T pair; repeat for a batch matrix (default: 1x16384)",
    )
    parser.add_argument("--channels", type=int, default=12288)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timer", choices=("auto", "cudagym", "events"), default="auto")
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend-sha", default="unknown")
    parser.add_argument("--backend-artifact-sha", default="unknown")
    parser.add_argument("--backend-pipeline", type=int)
    parser.add_argument("--frontend-sha", default="unknown")
    parser.add_argument("--attention-gym-sha", default=ATTENTION_GYM_PR266_SHA)
    parser.add_argument(
        "--locked-graphics-clock-mhz",
        type=int,
        help="requested nvidia-smi graphics clock lock; recorded for auditability",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    shapes = args.shape or [(1, 16384)]
    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn_backend_version": str(__import__("cudnn").backend_version()),
            "backend_sha": args.backend_sha,
            "backend_artifact_sha": args.backend_artifact_sha,
            "backend_pipeline": args.backend_pipeline,
            "frontend_sha": args.frontend_sha,
            "attention_gym_sha": args.attention_gym_sha,
            "cudagym": _package_version("cudagym"),
            "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
            "requested_graphics_clock_lock_mhz": args.locked_graphics_clock_mhz,
            "nvidia_smi_before_benchmark": _nvidia_smi_state(),
        },
        "methodology": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "requested_timer": args.timer,
            "bias": "preallocated zeros so both implementations compute the same operation",
            "backward": (
                "direct registered backward custom ops, including all kernels and reductions; "
                "cuDNN additionally computes dBias while attention-gym returns dX and dWeight"
            ),
        },
        "results": [
            _benchmark_shape(
                batch,
                tokens,
                channels=args.channels,
                width=args.width,
                timer=args.timer,
                warmup=args.warmup,
                iterations=args.iterations,
                check_correctness=not args.no_correctness,
                print_kernels=args.print_kernels,
            )
            for batch, tokens in shapes
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
