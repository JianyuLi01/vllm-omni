# Parallelism Patterns Reference

## Tensor Parallelism (TP)

Replace standard `nn.Linear` with vLLM's parallel linear layers:

| Pattern | vLLM Layer | When to Use |
|---------|-----------|-------------|
| Fan-out (first in FFN) | `ColumnParallelLinear` | Projection that splits output across ranks |
| Fan-in (second in FFN) | `RowParallelLinear` | Projection that gathers across ranks |
| QKV projection | `QKVParallelLinear` | Fused Q/K/V for self-attention |
| Single Q or K or V | `ColumnParallelLinear` | Separate projections (cross-attention) |
| Attention output | `RowParallelLinear` | Output projection after attention |

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)

class TPFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = ColumnParallelLinear(dim, ffn_dim)
        self.fc2 = RowParallelLinear(ffn_dim, dim)

    def forward(self, x):
        x, _ = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x, _ = self.fc2(x)
        return x
```

**TP constraints**: `hidden_dim`, `num_heads`, and `num_kv_heads` must be divisible by `tp_size`.

### RMSNorm with TP

When RMSNorm sits between TP-sharded dimensions, use `DistributedRMSNorm` from the Wan2.2 implementation pattern — it computes global RMS via all-reduce across TP ranks.

## CFG Parallelism

Inherit `CFGParallelMixin` in your pipeline and implement `predict_noise()`:

```python
from vllm_omni.diffusion.distributed.cfg_parallel.cfg_parallel import CFGParallelMixin

class MyPipeline(nn.Module, CFGParallelMixin):
    def predict_noise(self, model, latent_model_input, t, prompt_embeds, **kwargs):
        return model(latent_model_input, t, prompt_embeds, **kwargs)

    def forward(self, req):
        # In the denoising loop:
        noise_pred = self.predict_noise_maybe_with_cfg(
            model=self.transformer,
            sample=latents,
            timestep=t,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=guidance_scale > 1.0,
        )
        latents = self.scheduler_step_maybe_with_cfg(
            self.scheduler, noise_pred, t, latents
        )
```

## Sequence Parallelism (SP)

SP is applied non-intrusively via the `_sp_plan` dict on the transformer class. The framework applies hooks at module boundaries to shard/gather sequences.

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

class MyTransformer(nn.Module):
    _sp_plan = {
        # Split hidden_states input on dim=1 before first block
        "blocks.0": SequenceParallelInput(split_dim=1),
        # Gather output on dim=1 after final projection
        "proj_out": SequenceParallelOutput(gather_dim=1),
    }
```

For RoPE that needs splitting, add an entry for the RoPE module:

```python
_sp_plan = {
    "rope": SequenceParallelInput(split_dim=1, split_output=True, auto_pad=True),
    "blocks.0": SequenceParallelInput(split_dim=1),
    "proj_out": SequenceParallelOutput(gather_dim=1),
}
```

The `auto_pad=True` flag handles variable sequence lengths by padding to be divisible by SP degree and creating attention masks accordingly.

## VAE Patch Parallelism

If using `DistributedAutoencoderKLWan` or similar distributed VAE, the framework handles spatial sharding automatically. Set `vae_patch_parallel_size` in the parallel config.

## HSDP (Hybrid Sharded Data Parallel)

HSDP uses PyTorch FSDP2 to shard transformer weights. No code changes needed in the model — the loader handles it. Set `use_hsdp=True` in `DiffusionParallelConfig`.

## Adding Parallelism Incrementally

Recommended order:
1. **Basic single-GPU**: Get generation working first
2. **Tensor Parallelism**: Replace Linear layers, update `load_weights` for QKV fusion
3. **CFG Parallel**: Add `CFGParallelMixin`, implement `predict_noise`
4. **Sequence Parallelism**: Add `_sp_plan` to transformer
5. **HSDP**: Usually works out-of-box after TP is done
6. **VAE Patch Parallel**: Switch to distributed VAE class
