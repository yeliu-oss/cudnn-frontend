# cuDNN NWH causal-conv1d vs attention-gym PR 266

This PoC compares the cuDNN Frontend channel-last (`NWH`) causal-conv1d call
with the CuTeDSL implementation merged in
[attention-gym PR 266](https://github.com/meta-pytorch/attention-gym/pull/266).
Both paths run BF16, SiLU, width 4, use the same input/filter values, and include
the complete registered backward operation.  cuDNN's timed backward produces
`dX`, `dWeight`, and `dBias`; attention-gym's produces `dX` and `dWeight`, so the
comparison does not omit cuDNN's extra bias-gradient work.

The NVIDIA measurements use CUDAGym 1.8.3's cold-L2 CUDA-event timer with
shifting input pointers: 10 warmups and 50 measured iterations.  The script has
a clearly labelled standalone CUDA-event fallback for users without CUDAGym.

## GB300 result

Measured on NVIDIA GB300 (SM103), BF16, SiLU, channel-last, `D=12288`, `W=4`.
GPU 0 was explicitly locked to 1972 MHz graphics; memory stayed at its only
reported point, 3996 MHz.  CUDAGym telemetry observed 1972 MHz with 0 MHz
standard deviation in every timed region.  Driver 580.159.04, PyTorch
2.12.0+cu132, CuTeDSL 4.5.2, and CUDAGym 1.8.3 were used.

Primary shape (`B=1`, `T=16384`):

| Direction | cuDNN NWH mean (median, min) | attention-gym mean (median, min) | attention / cuDNN median |
|---|---:|---:|---:|
| Forward | 135.0 us (130.8, 129.0) | 132.8 us (123.9, 123.4) | 0.95x |
| Backward | 433.9 us (430.1, 426.3) | 528.5 us (522.8, 518.5) | **1.22x** |
| Forward + backward | 568.9 us (561.0, 555.3) | 661.3 us (646.7, 641.9) | **1.15x** |

The current PR 266 implementation is slightly faster for forward alone.  cuDNN
is 1.22x faster in backward and about 1.15x faster for the combined training
path, even though the cuDNN backward also computes `dBias`.

Constant-total-token batch matrix (median microseconds):

| B | T | cuDNN fwd | attention fwd | cuDNN bwd | attention bwd | training-path speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16384 | 130.8 | 123.9 | 430.1 | 522.8 | **1.15x** |
| 2 | 8192 | 131.1 | 124.9 | 429.1 | 528.4 | **1.17x** |
| 4 | 4096 | 131.0 | 125.1 | 428.9 | 535.3 | **1.18x** |
| 8 | 2048 | 131.1 | 124.9 | 430.8 | 534.9 | **1.17x** |

Across these shapes, the maximum absolute difference was at most 0.0009765625
for forward and `dX`, and 0.03125 for `dWeight` (BF16 outputs).

The profiled cuDNN sequence-parallel kernels for the primary shape were:

```text
cudnn_causal_conv1d_nwh_fwd_sp_k4_silu_cutlass__bfloat16_t_e32x2y64v8
cudnn_causal_conv1d_nwh_bwd_sp_k4_silu_cutlass__bfloat16_t_e32x2y64
```

The exact inputs are recorded in the JSON results: backend feature head
`b068acb86de2967fa75886e8f64c0f29051419c4`, backend pipeline 65123544
(`dea9bcdeb26048112aa16811d39e4038400d244b`), Frontend `develop`
`b2855e7e645041d7d25df3df18f80174fee8df79`, and attention-gym PR head
`f61ee082ac34b433c386b0980604cd03972f3be3`.
See the [primary raw result](gb300_b1_pipeline65123544.json) and
[batch-matrix raw result](gb300_batch_matrix_pipeline65123544.json).

## Reproduce

Use attention-gym at the exact PR head:

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]==4.5.2" apache-tvm-ffi==0.1.10
python -m pip install "git+https://github.com/meta-pytorch/attention-gym.git@f61ee082ac34b433c386b0980604cd03972f3be3"
```

Build/install cuDNN Frontend `develop` against a cuDNN backend containing the
NWH sequence-parallel kernels, then run the primary shape:

```bash
python benchmark_nwh_vs_attention_gym.py \
  --shape 1x16384 --channels 12288 --width 4 \
  --warmup 10 --iterations 50 --timer auto \
  --print-kernels --output gb300_b1.json
```

The same-total-token batch matrix demonstrates that the NWH API is not limited
to batch 1:

```bash
python benchmark_nwh_vs_attention_gym.py \
  --shape 1x16384 --shape 2x8192 --shape 4x4096 --shape 8x2048 \
  --channels 12288 --width 4 --output gb300_batch_matrix.json
```

cuDNN Frontend additionally exposes a channel-first causal-conv1d API and its
NWH API supports BF16, FP16, and FP32, optional bias, identity or SiLU, and
filter widths 2 through 128.  PR 266's optimized comparison path is contiguous
channel-last BF16 + SiLU with no bias; width 4 is its specialized fast path.

For strict reproduction of the numbers above, use CUDAGym and lock the GPU to
a supported stable point before running.  Without CUDAGym, `--timer auto`
falls back to the script's cold-L2 CUDA-event timer and labels that different
methodology in the output.
