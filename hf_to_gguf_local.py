"""
Local wrapper for hf_to_gguf.py.

Purpose:
- Run HF -> GGUF conversion and quantization on a local machine.
- Explicitly avoid Ollama registration (local registration is usually unnecessary for your workflow).

Example:
python hf_to_gguf_local.py \
  --model-id ytu-ce-cosmos/Turkish-Gemma-9b-v0.1 \
  --work-dir ~/hf_local_artifacts \
  --quantization-type q4_k_m
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local wrapper for hf_to_gguf.py (no Ollama registration)."
    )
    parser.add_argument(
        "--model-id",
        default="ytu-ce-cosmos/Turkish-Gemma-9b-v0.1",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--work-dir",
        default="~/hf_local_artifacts",
        help="Base local directory for full model + GGUF outputs.",
    )
    parser.add_argument(
        "--full-model-dir",
        default="",
        help="Override full-model directory. Default: <work-dir>/<model-name>-full",
    )
    parser.add_argument(
        "--gguf-path",
        default="",
        help="Override F16 GGUF path. Default: <work-dir>/<model-name>.F16.gguf",
    )
    parser.add_argument(
        "--quantized-gguf-path",
        default="",
        help="Override quantized GGUF path. Default: <work-dir>/<model-name>.<QTYPE>.gguf",
    )
    parser.add_argument(
        "--quantization-type",
        default="q4_k_m",
        help="llama.cpp quantization type (e.g. q4_k_m, q5_k_m, q8_0).",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="~/llama.cpp",
        help="Path where llama.cpp is cloned/built locally.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional HF token. Precedence: --hf-token > HF_TOKEN env > .env HF_TOKEN.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Enable HF Xet backend. By default this wrapper disables it for stability.",
    )
    return parser.parse_args()


def read_dotenv_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export ") :].strip()
        if k != key:
            continue

        return v.strip().strip('"').strip("'")

    return ""


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    model_name = args.model_id.split("/")[-1]
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    full_model_dir = (
        Path(args.full_model_dir).expanduser().resolve()
        if args.full_model_dir
        else work_dir / f"{model_name}-full"
    )
    gguf_path = (
        Path(args.gguf_path).expanduser().resolve()
        if args.gguf_path
        else work_dir / f"{model_name}.F16.gguf"
    )
    quantized_gguf_path = (
        Path(args.quantized_gguf_path).expanduser().resolve()
        if args.quantized_gguf_path
        else work_dir / f"{model_name}.{args.quantization_type.upper()}.gguf"
    )

    script_path = Path(__file__).resolve().with_name("hf_to_gguf.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Expected sibling script not found: {script_path}")

    dotenv_path = script_dir / ".env"
    resolved_hf_token = (
        args.hf_token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or read_dotenv_value(dotenv_path, "HF_TOKEN")
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--model-id",
        args.model_id,
        "--full-model-dir",
        str(full_model_dir),
        "--gguf-path",
        str(gguf_path),
        "--quantized-gguf-path",
        str(quantized_gguf_path),
        "--quantization-type",
        args.quantization_type,
        "--llama-cpp-dir",
        str(Path(args.llama_cpp_dir).expanduser().resolve()),
    ]

    if resolved_hf_token:
        cmd.extend(["--hf-token", resolved_hf_token])
    env = os.environ.copy()
    if not args.enable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    if resolved_hf_token:
        env["HF_TOKEN"] = resolved_hf_token

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    print("\nDone.")
    print(f"Full model dir:      {full_model_dir}")
    print(f"GGUF (F16):          {gguf_path}")
    print(f"GGUF (quantized):    {quantized_gguf_path}")
    print("Upload only the quantized GGUF file to Drive for Colab registration.")


if __name__ == "__main__":
    main()
