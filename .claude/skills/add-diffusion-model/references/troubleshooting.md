# Troubleshooting Reference

## Common Errors When Adding a Diffusion Model

### ImportError / ModuleNotFoundError

**Cause**: Missing or incorrect registration.

**Fix checklist**:
1. Model registered in `vllm_omni/diffusion/registry.py` `_DIFFUSION_MODELS` dict
2. `__init__.py` exports the pipeline class
3. Pipeline file exists at the correct path: `vllm_omni/diffusion/models/{folder}/{file}.py`
4. Class name in registry matches the actual class name in the file

### Shape Mismatch in Attention

**Symptom**: `RuntimeError: shape mismatch` or `expected 4D tensor`

**Cause**: QKV tensors not reshaped to `[batch, seq_len, num_heads, head_dim]`.

**Fix**: Before calling `self.attn(q, k, v, ...)`, ensure:
```python
q = q.view(batch, seq_len, self.num_heads, self.head_dim)
k = k.view(batch, kv_seq_len, self.num_kv_heads, self.head_dim)
v = v.view(batch, kv_seq_len, self.num_kv_heads, self.head_dim)
```

After attention, reshape back:
```python
out = out.reshape(batch, seq_len, -1)
```

### Weight Loading Failures

**Symptom**: `RuntimeError: size mismatch for parameter ...` or missing keys

**Debugging**:
1. Print diffusers weight names: `safetensors.safe_open(path, "pt").keys()`
2. Print model parameter names: `dict(model.named_parameters()).keys()`
3. Compare and add name remappings in `load_weights()`

**Common remappings needed**:
- `ff.net.0.proj` → `ff.net_0.proj` (PyTorch Sequential indexing)
- `.to_out.0.` → `.to_out.` (Sequential unwrapping)
- `scale_shift_table` → moved to a wrapper module

### Black/Blank/Noisy Output

**Possible causes**:
1. **Wrong latent normalization**: Check VAE expects latents scaled by `vae.config.scaling_factor`
2. **Wrong scheduler**: Using the wrong scheduler class or wrong `flow_shift`
3. **Missing CFG**: Some models require `guidance_scale > 1.0` with negative prompt
4. **Wrong timestep format**: Some schedulers expect float, others expect int/long
5. **Missing post-processing**: Raw VAE output may need denormalization

**Quick test**: Run with diffusers directly using the same seed and compare latents at each step.

### OOM (Out of Memory)

**Solutions** (in order of preference):
1. `--enforce-eager` to disable torch.compile (saves compile memory)
2. `--enable-cpu-offload` for model-level offload
3. `--enable-layerwise-offload` for block-level offload (better for large models)
4. `--vae-use-slicing --vae-use-tiling` for VAE memory reduction
5. Reduce resolution: `--height 480 --width 832`
6. Use TP: `--tensor-parallel-size 2`

### Different Output vs Diffusers Reference

**Common causes**:
1. **Attention backend difference**: FlashAttention vs SDPA may produce slightly different results. Set `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` to match diffusers
2. **Float precision**: vLLM-Omni may use bfloat16 where diffusers uses float32 for some operations
3. **Missing normalization**: Check all LayerNorm/RMSNorm are preserved
4. **Scheduler rounding**: Some schedulers have numerical sensitivity

### Tensor Parallel Errors

**Symptom**: `AssertionError: not divisible` or incorrect output with TP>1

**Fix**:
1. Verify `hidden_dim % tp_size == 0` and `num_heads % tp_size == 0`
2. Ensure `ColumnParallelLinear` / `RowParallelLinear` are used correctly
3. Check that norms between parallel layers use distributed norm if needed
4. Verify `load_weights` handles TP sharding for norm weights

### Model Not Detected / Wrong Pipeline Class

**Symptom**: `ValueError: Model class ... not found in diffusion model registry`

**Cause**: The model's `model_index.json` has a `_class_name` for the pipeline that doesn't match registry keys.

**Fix**: The registry key must match the diffusers pipeline class name from `model_index.json`. If using a different name, map it in the registry:
```python
"DiffusersPipelineClassName": ("your_folder", "your_file", "YourVllmClassName"),
```

## Debugging Workflow

1. **Add verbose logging**: Use `logger.info()` to print tensor shapes at each stage
2. **Compare step-by-step**: Run diffusers and vllm-omni side by side, comparing tensors after each major operation
3. **Use small configs**: Reduce `num_inference_steps=2`, small resolution for fast iteration
4. **Test transformer isolation**: Feed the same input to both diffusers and vllm-omni transformers, compare outputs
5. **Binary search for bugs**: Comment out blocks/layers to isolate where divergence starts
