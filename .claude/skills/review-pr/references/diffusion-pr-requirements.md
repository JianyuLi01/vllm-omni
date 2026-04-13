# PR Body Requirements for Diffusion Model Contributions

This document defines the exact structure and content expected in the PR description when adding a new diffusion model to vLLM-Omni. Reviewers use this as the source of truth when evaluating PR body completeness (Dimension 1).

---

## Complete PR Body Template

Below is the canonical template. PRs that follow this template pass Dimension 1 automatically.

````markdown
## Summary

[One paragraph: what model is being added, architecture type (DiT / AR+DiT / etc.),
and which capabilities are introduced (text-to-image / video / editing / etc.)]

**Model:** `org/model-id`
**Architecture:** DiT / AR+DiT / other
**Modality:** text-to-image / text-to-video / image-editing

---

## vLLM-Omni Generation

### Offline Inference

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="org/model-id")
outputs = omni.generate("a cup of coffee on a wooden table, photorealistic")
outputs[0].request_output[0].images[0].save("output.png")
```

### Online Serving

```bash
# Start server
vllm serve org/model-id --omni --port 8091

# Generate
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "a cup of coffee on a wooden table"}],
    "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 50, "seed": 42}
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
     | cut -d',' -f2 | base64 -d > output.png
```

### Sample Outputs

| Prompt | Output |
|--------|--------|
| "a cup of coffee on a wooden table, photorealistic" | ![sample1](./assets/sample1.png) |
| "a futuristic city at night, neon lights" | ![sample2](./assets/sample2.png) |

### Performance Measurements

**Hardware:** 1× NVIDIA A100 80GB  
**Resolution:** 1024×1024  
**Steps:** 50  

| Metric | Value |
|--------|-------|
| e2e latency | X.X s |
| Peak VRAM | XX.X GB |

---

## diffusers Baseline (Recommended)

### Script

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("org/model-id", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
image = pipe("a cup of coffee on a wooden table, photorealistic",
             num_inference_steps=50, height=1024, width=1024).images[0]
image.save("output_diffusers.png")
```

### Comparison

| | vLLM-Omni | diffusers |
|-|-----------|-----------|
| e2e latency | X.X s | X.X s |
| Peak VRAM | XX.X GB | XX.X GB |
| Sample | ![vllm_omni](./assets/sample_vllm.png) | ![diffusers](./assets/sample_diffusers.png) |

---

## Acceleration Features

### Supported Features

| Feature | Supported | CLI Flag / Config Key |
|---------|----------|----------------------|
| Sequence Parallel | ✅ | `--sequence-parallel-size N` |
| CFG Parallel | ❌ | — |
| VAE Patch Parallel | ✅ | `vae_patch_parallel: true` |
| Tensor Parallel | ✅ | `--tensor-parallel-size N` |
| TeaCache | ✅ | `--teacache-thresh 0.15` |

### Acceleration Demo

```bash
# Tensor Parallel across 2 GPUs
vllm serve org/model-id --omni --tensor-parallel-size 2

# TeaCache for 1.5× speedup
vllm serve org/model-id --omni --teacache-thresh 0.15
```

| Config | Latency | VRAM/GPU |
|--------|---------|---------|
| Baseline (1 GPU) | X.X s | XX GB |
| TP=2 | X.X s | XX GB |
| TeaCache | X.X s | XX GB |
| TP=2 + TeaCache | X.X s | XX GB |

---

## Memory Optimization Features

### Supported Options

| Feature | Supported | CLI Flag / Config Key |
|---------|----------|----------------------|
| CPU Offload | ✅ | `--enable-cpu-offload` |
| FP8 Quantization | ✅ | `--quantization fp8` |
| VAE Tiling | ✅ | `vae_tiling: true` |

### Memory Reduction Demo

```bash
# CPU offload (reduces VRAM from XX GB to YY GB)
vllm serve org/model-id --omni --enable-cpu-offload

# FP8 quantization
vllm serve org/model-id --omni --quantization fp8
```

| Config | Peak VRAM | Latency |
|--------|----------|---------|
| Baseline (BF16) | XX.X GB | X.X s |
| CPU Offload 10 GB | XX.X GB | X.X s |
| FP8 Quantization | XX.X GB | X.X s |

---

## Combined Feature Validation

```bash
# CPU offload + TeaCache + Tensor Parallel
vllm serve org/model-id --omni \
  --tensor-parallel-size 2 \
  --teacache-thresh 0.15 \
  --cpu-offload-gb 5
```

| Config | Latency | VRAM/GPU | Quality Notes |
|--------|---------|---------|--------------|
| All features combined | X.X s | XX GB | No visible quality degradation |

Sample output with combined config: ![combined](./assets/sample_combined.png)

---

## Documentation Updates

- [ ] `docs/models/supported_models.md` — model row added
- [ ] `docs/features/feature_support.md` — feature row added
- [ ] `docs/features/compatibility.md` — compatibility table updated (if applicable)
- [ ] `docs/models/<model_name>.md` — usage example doc added

---

## Test Coverage

- [ ] `tests/e2e/online_serving/test_<model_name>_expansion.py` — online serving e2e test
- [ ] `tests/models/test_<model_name>.py` — offline inference test
````

---

## Measurement Guidelines

### How to Measure e2e Latency

**Offline:**
```python
import time
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="org/model-id")

# Warm-up (1 run, not measured)
omni.generate("warm-up prompt")

# Timed run
start = time.perf_counter()
outputs = omni.generate("a cup of coffee on a wooden table")
end = time.perf_counter()
print(f"e2e latency: {end - start:.2f}s")
```

**Online (using `curl` with timing):**
```bash
time curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "a cup of coffee"}], ...}'
```

### How to Measure Peak VRAM

```python
import torch
torch.cuda.reset_peak_memory_stats()

# Run generation here

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM: {peak_mb / 1024:.1f} GB")
```

Or via `nvidia-smi`:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1
```

### What "Same Conditions" Means for diffusers Comparison

When comparing vLLM-Omni vs diffusers:
- Same GPU model and count
- Same prompt
- Same resolution (H × W)
- Same `num_inference_steps`
- Same `seed`
- Same data type (both BF16 or both FP16)
- Both measured after one warm-up run

---

## Common PR Body Mistakes

| Mistake | Correct Approach |
|---------|-----------------|
| Using a tiny resolution (256×256) for latency measurement | Use the model's recommended resolution (typically 512×512 minimum, 1024×1024 for flagship models) |
| Reporting latency without warm-up run | Always warm up the model first; measure on the second+ run |
| Showing only one sample image | Show 2–3 samples with diverse prompts to demonstrate quality |
| Only demonstrating offline mode | Include both offline (`Omni`) and online (`vllm serve`) |
| Reporting VRAM from `nvidia-smi` at idle | Measure peak VRAM during active generation |
| Listing features as "planned" in the support table | Only mark ✅ for features that are implemented and tested in this PR |
