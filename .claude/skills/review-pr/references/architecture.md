# vLLM-Omni Architecture

## Overview

vLLM-Omni extends vLLM for omni-modal (multi-modal) model serving, supporting text, image, video, and audio generation. It provides a unified inference engine with OpenAI-compatible APIs for both LLM and diffusion models.

**Key Differentiators from vLLM:**
- Multi-stage pipeline architecture (e.g., Thinker → Talker → Code2Wav)
- Diffusion Transformer (DiT) model support
- Inter-stage communication via shared memory connectors
- Stage-level concurrency control

---

## Five-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  1. User Interface Layer                            │
│     OpenAI-compatible API, CLI, Python SDK          │
├─────────────────────────────────────────────────────┤
│  2. Orchestration Layer                             │
│     Omni, AsyncOmni - coordinates stages            │
├─────────────────────────────────────────────────────┤
│  3. Engine Layer                                    │
│     ModelExecutor, Scheduler, KV cache management   │
├─────────────────────────────────────────────────────┤
│  4. Execution Layer                                 │
│     Workers (GPUARWorker, GPUGenerationWorker)      │
├─────────────────────────────────────────────────────┤
│  5. Model Layer                                     │
│     OmniLLM, OmniDiffusion, model implementations   │
└─────────────────────────────────────────────────────┘
```

---

## Key Directories

| Directory | Purpose | Critical? | Review Focus |
|-----------|---------|-----------|--------------|
| `vllm_omni/entrypoints/` | API server, CLI | High | Input validation, error handling |
| `vllm_omni/engine/` | Engine, scheduler | **Critical** | Concurrency, state management |
| `vllm_omni/model_executor/` | Model execution | **Critical** | Weight loading, memory |
| `vllm_omni/diffusion/` | Diffusion support | High | Latent cache, generation |
| `vllm_omni/connectors/` | Inter-stage IPC | High | Shared memory, cleanup |
| `vllm_omni/stages/` | Stage definitions | High | Lifecycle, state |
| `vllm_omni/config/` | Configuration | Medium | Validation, defaults |
| `vllm_omni/utils/` | Utilities | Low | Test coverage |

---

## Core Components

### Entry Points

**Synchronous:** `Omni` class
```python
from vllm_omni import Omni
llm = Omni(model="Qwen/Qwen2.5-Omni-7B")
outputs = llm.generate("Hello")
```

**Asynchronous:** `AsyncOmni` class
```python
from vllm_omni import AsyncOmni
llm = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
outputs = await llm.generate("Hello")
```

### Stage Types

| Type | Worker Classes | Use Case |
|------|---------------|----------|
| `llm` | `GPUARWorker`, `GPUGenerationWorker` | Text generation, multimodal understanding |
| `diffusion` | Diffusion-specific workers | Image/video generation |
| `audio` | Audio workers | TTS, audio synthesis |

### Connectors

| Connector | Use Case | Performance |
|-----------|----------|-------------|
| `OmniShmConnector` | Same-machine inter-process | Fastest |
| `OmniZmqConnector` | Distributed/multi-node | Network-capable |

---

## Multi-Stage Pipeline

### Example: Qwen-Omni Audio Generation

```
┌─────────┐     ┌─────────┐     ┌───────────┐
│ Thinker │ ──▶ │ Talker  │ ──▶ │ Code2Wav  │
│ (LLM)   │     │ (LLM)   │     │ (Audio)   │
└─────────┘     └─────────┘     └───────────┘
     │               │                │
   Text/Audio     Text Code      Audio Output
   Understanding  Generation     Synthesis
```

### Configuration (YAML)

```yaml
stages:
  - name: thinker
    type: llm
  - name: talker
    type: llm
  - name: code2wav
    type: audio
```

---

## Data Flow

```
Request → API Layer → AsyncOmni/Omni
                         │
                         ▼
                    Stage Coordinator
                    (YAML config)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Stage 1        Stage 2        Stage N
      (Worker)       (Worker)       (Worker)
          │              │              │
          └──────────────┴──────────────┘
                         │
                    OmniConnector
                    (Shm/Zmq)
                         │
                         ▼
                    Response
```

---

## Memory Management

### KV Cache (LLM Stages)
- Paged attention inherited from vLLM
- Block-based memory allocation
- Prefix caching for repeated contexts

### Diffusion Latent Cache
- Intermediate latent storage
- Timestep-based scheduling
- Memory-pressure aware

---

## Supported Models

### Omni-Modal Models
- Qwen3-Omni
- Qwen2.5-Omni
- Qwen3-TTS (CustomVoice, VoiceDesign, Base)
- MiMo-Audio
- Bagel

### Diffusion Models
- Z-Image
- Qwen-Image
- Wan2.2 (video)
- FLUX

---

## Review Considerations

### Critical Paths (High Impact)
- `vllm_omni/engine/` — scheduler changes affect all workloads
- `vllm_omni/model_executor/` — model loading bugs break inference
- `vllm_omni/connectors/` — communication bugs cause hangs/crashes

### High-Risk Patterns
1. **Stage coordination changes** — can break multi-stage pipelines
2. **Memory management in connectors** — shared memory leaks
3. **Worker lifecycle changes** — affect tensor parallelism
4. **Input validation gaps** — engine crashes instead of 400 errors

### Testing Requirements
| Component | Test Requirement |
|-----------|------------------|
| LLM stages | Actual model inference |
| Diffusion stages | Generation quality |
| Connectors | Load testing, memory leak check |
| Multi-stage | End-to-end pipeline |
| API endpoints | Input validation, error responses |
