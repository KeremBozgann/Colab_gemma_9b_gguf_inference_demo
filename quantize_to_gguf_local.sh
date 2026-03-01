# THE FOLLOWING DOES NOT WORK: TRIES TO DEQUANTIZE THE MODEL FIRST
cd "/Users/kerembozgan/Library/CloudStorage/GoogleDrive-bozgan.kerem4@gmail.com/My Drive/training-embedding"

export HF_HUB_DISABLE_XET=1
# optional if not already in .env:
# export HF_TOKEN="hf_..."


uv run --active python hf_quantized_to_gguf.py \
  --quantized-model-dir "/Users/kerembozgan/Desktop/models/turkish-gemma-9b-v01-4bit" \
  --gguf-path "/Users/kerembozgan/Desktop/models/Turkish-Gemma-9b-v0.1.from-hf-4bit.F16.gguf" \
  --conversion-mode dequantize \
  --dequantized-model-dir "/Users/kerembozgan/Desktop/models/turkish-gemma-9b-v01-dequantized" \
  --llama-cpp-dir "$HOME/llama.cpp"


