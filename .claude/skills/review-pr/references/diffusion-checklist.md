# Diffusion Model Contribution Review Checklist

Use this checklist as a structured pass after reading the PR diff and body. Each item maps to a dimension in the main SKILL.md workflow.

---

## Dimension 1: PR Body Completeness

### 1.1 Required Evidence (Table A)

| Item | What to Check | Missing → Action |
|------|---------------|------------------|
| **vLLM-Omni generation script** | Runnable Python or bash snippet using `Omni` / `vllm serve` that produces output | Request script |
| **Generated sample outputs** | At least 1 image / video / audio sample attached or linked | Request sample |
| **vLLM-Omni e2e latency** | Wall-clock time from request to output, with GPU model, count, resolution, steps | Request measurement |
| **vLLM-Omni VRAM usage** | Peak VRAM in GB during generation, with resolution / steps | Request measurement |

- [ ] **vLLM-Omni generation script present**
  - Must use `Omni(model=...)` for offline, or `vllm serve ... --omni` for online
  - Script must be runnable (no pseudocode, no placeholder `<model_id>`)
  - If only offline or only online is shown, note which is missing

- [ ] **Sample output attached or linked**
  - Image PR: at least 1 image (PNG/JPEG), ideally multiple prompts
  - Video PR: at least 1 video clip or animated GIF
  - Audio PR: audio file or spectrogram + waveform visualization
  - Resolution and generation parameters must be visible (caption or table)

- [ ] **vLLM-Omni e2e latency recorded**
  - Wall-clock time from request submission to final output received
  - Must include: GPU model, GPU count, resolution (H×W), `num_inference_steps`
  - Single measurement is acceptable; batch=1 is the baseline

- [ ] **vLLM-Omni peak VRAM recorded**
  - Peak GPU memory in GB
  - Measurement method noted (e.g., `torch.cuda.max_memory_allocated()`, `nvidia-smi`)
  - Must match the hardware spec reported with latency

### 1.2 Recommended Evidence (Table B)

| Item | Why It Matters |
|------|----------------|
| **diffusers generation script** | Reproducible baseline for quality and speed comparison |
| **diffusers sample outputs** | Side-by-side quality comparison demonstrates parity |
| **diffusers e2e latency** | Quantifies vLLM-Omni speedup relative to reference |
| **diffusers VRAM usage** | Quantifies memory reduction or overhead |

- [ ] **diffusers generation script**
  - Same prompt, same resolution, same number of steps as vLLM-Omni script
  - Uses `diffusers` library directly (not wrapped in vllm-omni)

- [ ] **diffusers sample output**
  - Side-by-side or clearly labeled comparison with vLLM-Omni output
  - Same prompt used for both

- [ ] **diffusers e2e latency**
  - Recorded under same hardware conditions as vLLM-Omni measurement
  - Speedup ratio (vLLM-Omni latency / diffusers latency) stated or calculable

- [ ] **diffusers VRAM usage**
  - Peak VRAM under same conditions
  - Delta (VRAM_vllm_omni − VRAM_diffusers) highlighted if significant

---

## Dimension 2: Code Checks

### 2.1–2.2 Inference Mode Coverage (both required)

| Check | Pass Condition | Flag |
|-------|----------------|------|
| **Offline inference** | `Omni` / `OmniLLM` integration exists; model can be instantiated and called | Missing offline path |
| **Online serving** | `vllm serve` or `AsyncOmni` handles the model; API routes return correct responses | Missing online path |

### 2.1 Offline Inference

- [ ] `Omni` or `OmniLLM` can instantiate the model (`model="<hf_id>"`)
- [ ] `.generate()` returns a valid output object with images/video/audio fields
- [ ] Default stage config YAML exists under `vllm_omni/configs/` or equivalent
- [ ] Model is registered in the model registry (import path resolvable)

### 2.2 Online Serving

- [ ] `AsyncOmni` or API server route handles the model
- [ ] `vllm serve <model> --omni` startup does not error
- [ ] `/v1/chat/completions` or appropriate endpoint returns correct response schema
- [ ] Input validation rejects malformed requests with 400, not engine crash

### 2.3 Diffusers Mixin Cleanup

- [ ] No `DiffusionPipelineMixin` in added/modified code
- [ ] No `SchedulerMixin` in added/modified code
- [ ] No `ConfigMixin` (diffusers) in added/modified code
- [ ] No direct `diffusers.pipelines.*` import in production model code
  - Exception: `diffusers` may appear in test files or benchmark scripts

**Detection command:**
```bash
gh pr diff <pr_number> --repo vllm-project/vllm-omni \
  | grep -E '^\+' \
  | grep -E 'DiffusionPipelineMixin|SchedulerMixin|ConfigMixin|from diffusers\.pipelines'
```

### 2.4 Acceleration Features

At least one of the following must be implemented and demonstrably working:

| Feature | Detection Pattern | Notes |
|---------|------------------|-------|
| **Sequence Parallel** | `sequence_parallel`, `sp_group`, `ring_attn` | For transformer attention |
| **CFG Parallel** | `cfg_parallel`, `cfg_group`, dual-batch guidance | Splits positive/negative across devices |
| **VAE Patch Parallel** | `vae_patch_parallel`, `patch_parallel` | For high-res VAE decoding |
| **Tensor Parallel** | `tp_group`, `ColumnParallelLinear`, `RowParallelLinear` | Linear layer sharding |
| **TeaCache / Step Cache** | `teacache`, `cache_dit`, `cache_interval` | Skip redundant denoising steps |

- [ ] At least one acceleration feature implemented
- [ ] Feature can be enabled via config or CLI flag (not hardcoded)
- [ ] PR body or docs show the speedup from enabling the feature

### 2.5 Memory Optimization Features

At least one of the following must be implemented:

| Feature | Detection Pattern | Notes |
|---------|------------------|-------|
| **CPU Offload** | `cpu_offload`, `offload_to_cpu`, `--enable-cpu-offload` | Move weights to CPU when idle |
| **Quantization** | `quantization`, `int8`, `fp8`, `bnb`, `bitsandbytes` | Reduced precision weights |
| **VAE Tiling** | `vae_tiling`, `tile_size`, `tiled_decode` | Decode VAE in spatial tiles |

- [ ] At least one memory optimization feature implemented
- [ ] Feature can be enabled via config or CLI flag
- [ ] PR body or docs show VRAM reduction from enabling the feature

---

## Dimension 3: Documentation

**Find doc files:**
```bash
gh pr view <pr_number> --repo vllm-project/vllm-omni \
  --json files --jq '.files[].path' | grep -E '\.md$|docs/'
```

| Doc Artifact | Required | What to Check |
|--------------|----------|---------------|
| **Model support table** | Yes | Row added: model name, architecture, HF model ID, modality, min VRAM |
| **Feature support table** | Yes | Row showing which acceleration and memory features are supported |
| **Usage example doc** | Yes | Runnable offline + online example for the new model |
| **Feature compatibility table** | Optional | If multiple features: matrix showing validated combinations |

### 3.1 Model Support Table

- [ ] Table updated in `docs/models/supported_models.md` (or equivalent)
- [ ] New row contains: Model Name, Architecture, HF Model ID, Modality, Min VRAM
- [ ] Modality correctly labeled (text-to-image / text-to-video / image-editing / etc.)

### 3.2 Feature Support Table

- [ ] Table updated showing which features apply to the new model
- [ ] Each column maps to a specific feature (Seq Parallel, CFG Parallel, VAE Patch, TP, CPU Offload, Quantization)
- [ ] Cells use consistent notation (✅ supported / ❌ not supported / 🔄 planned)

### 3.3 Feature Compatibility Table (Optional)

Only checked when model supports multiple features:

- [ ] Compatibility matrix exists showing which feature combinations are validated
- [ ] Untested combinations marked clearly (e.g., `❓`)

### 3.4 Usage Example Doc

- [ ] Example doc added or updated for the new model
- [ ] Offline inference example present (with `Omni`)
- [ ] Online serving example present (with `vllm serve` + `curl`)
- [ ] At least one acceleration feature example shown
- [ ] At least one memory optimization example shown
- [ ] Generation parameters documented (resolution, steps, guidance scale)

---

## Dimension 4: Test Coverage

**Find test files:**
```bash
gh pr view <pr_number> --repo vllm-project/vllm-omni \
  --json files --jq '.files[].path' | grep '^tests/'
```

| Test Type | Location | Required | What to Verify |
|-----------|----------|----------|----------------|
| **e2e online serving test** | `tests/e2e/online_serving/` | Yes | Start server, send request, assert output shape / non-null |
| **Offline inference test** | `tests/` or `tests/models/` | No (if e2e test exists) | Instantiate `Omni`, call `.generate()`, assert output |
| **Acceleration / memory feature test** | Alongside above | Recommended | Enable each supported feature, verify output and speed |
| **Combined feature test** | Alongside above | Required if 2+ features | Enable multiple features together, assert output + VRAM + latency |

### 4.1 e2e Online Serving Test (Required)

Location: `tests/e2e/online_serving/test_<model_name>.py`

- [ ] Test file exists in `tests/e2e/online_serving/`
- [ ] Test starts the server (or uses a fixture that does)
- [ ] Test sends a generation request to the API endpoint
- [ ] Test asserts response is non-null and output has correct shape/type
- [ ] Test includes at least one functional scenario (basic generation)
- [ ] Test includes at least one negative case (invalid input → 400 error)

### 4.2 Offline Inference Test (Recommended; required only when no e2e test is present)

- [ ] Test instantiates `Omni(model="<hf_id>")`
- [ ] Test calls `.generate()` with at least one prompt
- [ ] Test asserts output is a valid image/video/audio
- [ ] Test covers at least one edge case (empty prompt, special characters, max resolution)

### 4.3 Acceleration and Memory Feature Tests (Recommended)

- [ ] Test enables each acceleration feature and asserts output validity
- [ ] Test enables each memory optimization and asserts output validity + VRAM within budget

### 4.4 Combined Feature Test (Required when 2+ features implemented)

Only checked when the PR implements **two or more** acceleration or memory features:

- [ ] Test (or an extended e2e test case) enables multiple features simultaneously
- [ ] Output under combined config is asserted as valid (quality preserved)
- [ ] Latency under combined config is reported
- [ ] VRAM under combined config is reported
- [ ] Any quality degradation under combination is explicitly documented

---

## Quick Red Flags Summary

Flag immediately and post a blocking comment if any of these are true:

| Red Flag | Severity | Action |
|----------|---------|--------|
| No generation script in PR body | 🔴 Blocking | Request before review proceeds |
| No sample output in PR body | 🔴 Blocking | Request before review proceeds |
| No latency or VRAM measurements | 🔴 Blocking | Request before review proceeds |
| Online serving not implemented | 🔴 Blocking | Must implement both modes |
| Offline inference not implemented | 🔴 Blocking | Must implement both modes |
| Diffusers mixin still in code | 🔴 Blocking | Must remove before merge |
| No acceleration feature | 🔴 Blocking | Must implement at least one |
| No memory optimization | 🔴 Blocking | Must implement at least one |
| No e2e online serving test | 🔴 Blocking | Must add test |
| No offline inference test (and no e2e test) | 🔴 Blocking | Must add at least one test |
| No offline inference test (e2e test exists) | 💡 Recommended | Suggest, do not block |
| Model support table not updated | 🔴 Blocking | Must update docs |
| Feature support table not updated | 🔴 Blocking | Must update docs |
| No diffusers comparison | 💡 Recommended | Suggest, do not block |
| No combined feature test (2+ features) | ⚠️ Conditional | Required when 2+ features implemented |
| No usage example doc | 🔴 Blocking | Must add example |
