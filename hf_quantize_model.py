"""
Download full model weights, quantize with bitsandbytes, and save quantized weights.

Example:
python hf_quantize_model.py \
  --model-id ytu-ce-cosmos/Turkish-Gemma-9b-v0.1 \
  --download-dir /content/drive/MyDrive/models/turkish-gemma-9b-full \
  --output-dir /content/drive/MyDrive/models/turkish-gemma-9b-4bit \
  --bits 4
"""

from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Gemma model and save locally.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id for full-precision model.")
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional local folder where full model files are downloaded before quantization.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save quantized model and tokenizer.")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, help="Target quantization bits.")
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Compute dtype used while loading before save.",
    )
    return parser.parse_args()


def download_source_checkpoint(model_id: str, download_dir: str | None) -> str:
    if download_dir:
        return snapshot_download(
            repo_id=model_id,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    return snapshot_download(
        repo_id=model_id,
        resume_download=True,
    )


def build_quant_config(bits: int) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def persist_quantization_config(model: AutoModelForCausalLM, quant_config: BitsAndBytesConfig) -> None:
    """
    Ensure config.json contains a valid non-null quantization_config for reload.
    """
    model.config.quantization_config = quant_config.to_dict()


def normalize_generation_config_for_save(model: AutoModelForCausalLM) -> None:
    """
    Some checkpoints ship with sampling params set while do_sample=False.
    Recent Transformers versions reject saving such invalid generation configs.
    """
    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        return

    # Keep generation config consistent for recent Transformers validation.
    if hasattr(gen_config, "use_cache"):
        gen_config.use_cache = True

    if getattr(gen_config, "do_sample", False):
        return

    # Reset sample-only params to validation-safe defaults for greedy mode.
    if hasattr(gen_config, "temperature"):
        gen_config.temperature = 1.0
    if hasattr(gen_config, "top_p"):
        gen_config.top_p = 1.0
    if hasattr(gen_config, "top_k"):
        gen_config.top_k = 50
    if hasattr(gen_config, "typical_p"):
        gen_config.typical_p = 1.0
    if hasattr(gen_config, "min_p"):
        gen_config.min_p = None
    if hasattr(gen_config, "epsilon_cutoff"):
        gen_config.epsilon_cutoff = 0.0
    if hasattr(gen_config, "eta_cutoff"):
        gen_config.eta_cutoff = 0.0


def main() -> None:
    args = parse_args()
    source_checkpoint = download_source_checkpoint(args.model_id, args.download_dir)

    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    quant_config = build_quant_config(args.bits)

    tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_checkpoint,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    persist_quantization_config(model, quant_config)
    normalize_generation_config_for_save(model)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    print(f"Saved {args.bits}-bit quantized model to: {args.output_dir}")


if __name__ == "__main__":
    main()
