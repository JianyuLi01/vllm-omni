export VLLM_OMNI_WAN_DUMMY_TEXT_ENCODER=1
export VLLM_OMNI_SKIP_DUMMY_RUN=1
export VLLM_OMNI_WAN_PROFILE_TRANSFORMER_ONLY=1
export VLLM_TORCH_PROFILER_DIR="./"

MODEL_PATH="/mnt/disk2/hf_models/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH="/mnt/disk2/hf_models/Wan2.2-T2V-A14B-Diffusers"

python text_to_video.py \
  --model "$MODEL_PATH" \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 128 --width 128 --num-frames 9 \
  --guidance-scale 1.0 --guidance-scale-high 1.0 \
  --boundary-ratio 0.0 --flow-shift 12.0 \
  --num-inference-steps 40 --fps 16 \
  --output t2v_out.mp4 \
  --enforce-eager

#  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./perf"}'
