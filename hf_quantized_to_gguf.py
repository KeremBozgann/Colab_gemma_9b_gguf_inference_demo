"""
Convert an already-quantized HF checkpoint folder to a single GGUF output.

This script supports two conversion modes:
1) direct:      pass the quantized HF folder directly to llama.cpp converter.
2) dequantize:  load quantized model in Transformers, dequantize to float weights,
                save a temporary HF folder, then convert that folder to GGUF.

Examples:

Direct conversion:
python hf_quantized_to_gguf.py \
  --quantized-model-dir /path/to/hf-4bit-model \
  --gguf-path /path/to/model.F16.gguf

Fallback path (more compatible, heavier RAM usage):
python hf_quantized_to_gguf.py \
  --quantized-model-dir /path/to/hf-4bit-model \
  --gguf-path /path/to/model.F16.gguf \
  --conversion-mode dequantize
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert quantized HF model to GGUF.")
    parser.add_argument(
        "--quantized-model-dir",
        required=True,
        help="Path to quantized HF model directory (e.g., output of hf_quantize_model.py).",
    )
    parser.add_argument(
        "--gguf-path",
        required=True,
        help="Output path for converted GGUF (typically F16).",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=os.environ.get("LLAMA_CPP_DIR", ""),
        help=(
            "Path to llama.cpp. If omitted, uses /content/llama.cpp when present, "
            "otherwise ~/llama.cpp."
        ),
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="GGUF output type for convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--conversion-mode",
        choices=["direct", "dequantize"],
        default="direct",
        help=(
            "'direct': convert quantized HF folder directly. "
            "'dequantize': dequantize HF model first, then convert."
        ),
    )
    parser.add_argument(
        "--dequantized-model-dir",
        default="",
        help="Optional folder to store intermediate dequantized HF model.",
    )
    parser.add_argument(
        "--dequantized-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype used for saved dequantized model in --conversion-mode dequantize.",
    )
    parser.add_argument(
        "--keep-dequantized",
        action="store_true",
        help="Keep intermediate dequantized folder when using --conversion-mode dequantize.",
    )
    parser.add_argument(
        "--register-ollama",
        action="store_true",
        help="Register resulting GGUF into Ollama after conversion.",
    )
    parser.add_argument(
        "--ollama-model-name",
        default="quantized-hf-gguf",
        help="Ollama model name used with --register-ollama.",
    )
    parser.add_argument(
        "--ollama-bin",
        default=os.environ.get("OLLAMA_BIN", "ollama"),
        help="Path to ollama binary used with --register-ollama.",
    )
    parser.add_argument(
        "--ollama-models-dir",
        default=os.environ.get("OLLAMA_MODELS", ""),
        help="OLLAMA_MODELS directory used with --register-ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
        help="OLLAMA_HOST used with --register-ollama.",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=2048,
        help="num_ctx value written in generated Modelfile for --register-ollama.",
    )
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def resolve_llama_cpp_dir(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if Path("/content/llama.cpp").exists():
        return Path("/content/llama.cpp")
    return Path("~/llama.cpp").expanduser().resolve()


def ensure_parents(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def ensure_llama_cpp_tools(llama_cpp_dir: Path) -> Path:
    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError("git is required.")

    if not (llama_cpp_dir / ".git").exists():
        run([git_bin, "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(llama_cpp_dir)])

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing converter script: {convert_script}")

    return convert_script


def dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def build_dequantized_dir(args: argparse.Namespace, quantized_model_dir: Path) -> Path:
    if args.dequantized_model_dir:
        path = Path(args.dequantized_model_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    temp = tempfile.mkdtemp(prefix=f"{quantized_model_dir.name}-dequantized-")
    return Path(temp)


def copy_tokenizer_assets(src_dir: Path, dst_dir: Path) -> None:
    # Keep tokenizer metadata aligned with the original quantized checkpoint.
    patterns = [
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template*",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "*.model",
    ]
    copied: set[Path] = set()
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if not src.is_file() or src in copied:
                continue
            shutil.copy2(src, dst_dir / src.name)
            copied.add(src)


def dequantize_model_dir(src_dir: Path, dst_dir: Path, dtype_name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_dtype = dtype_from_name(dtype_name)

    print(f"Loading quantized model from: {src_dir}")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(src_dir), use_fast=True)
    except Exception as fast_error:  # noqa: BLE001 - keep fallback broad for HF tokenizer edge cases.
        print(f"Fast tokenizer load failed, retrying with slow tokenizer: {fast_error}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(src_dir), use_fast=False)
        except Exception as slow_error:  # noqa: BLE001
            print(f"Tokenizer load failed; will copy tokenizer files directly: {slow_error}")
            tokenizer = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(src_dir),
            device_map="cpu",
            torch_dtype=target_dtype,
        )
    except ValueError as error:
        if "requires `accelerate`" not in str(error):
            raise
        print("accelerate not found for device_map=cpu; retrying model load without device_map.")
        model = AutoModelForCausalLM.from_pretrained(
            str(src_dir),
            torch_dtype=target_dtype,
        )

    if not hasattr(model, "dequantize"):
        raise RuntimeError(
            "Loaded model does not expose dequantize(). "
            "Try --conversion-mode direct or use a compatible quantized checkpoint."
        )

    print("Dequantizing model in memory...")
    model.dequantize()
    model.to(target_dtype)

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dequantized model to: {dst_dir}")
    model.save_pretrained(str(dst_dir), safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(dst_dir))
    else:
        copy_tokenizer_assets(src_dir, dst_dir)

    # Release memory early.
    del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_to_gguf(convert_script: Path, source_dir: Path, gguf_path: Path, outtype: str) -> None:
    run(
        [
            sys.executable,
            str(convert_script),
            str(source_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            outtype,
        ]
    )


def register_ollama_model(
    model_path: Path,
    model_name: str,
    ollama_bin: str,
    ollama_models_dir: str,
    ollama_host: str,
    ollama_num_ctx: int,
) -> None:
    bin_path = Path(ollama_bin).expanduser().resolve() if "/" in ollama_bin else None
    executable = str(bin_path) if bin_path else ollama_bin

    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host
    if ollama_models_dir:
        Path(ollama_models_dir).mkdir(parents=True, exist_ok=True)
        env["OLLAMA_MODELS"] = ollama_models_dir

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".modelfile") as f:
        modelfile_path = Path(f.name)
        f.write(f"FROM {model_path}\n")
        f.write(f"PARAMETER num_ctx {ollama_num_ctx}\n")

    try:
        run([executable, "create", model_name, "-f", str(modelfile_path)], env=env)
        run([executable, "list"], env=env)
    finally:
        modelfile_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    quantized_model_dir = Path(args.quantized_model_dir).expanduser().resolve()
    gguf_path = Path(args.gguf_path).expanduser().resolve()
    llama_cpp_dir = resolve_llama_cpp_dir(args.llama_cpp_dir)

    if not quantized_model_dir.exists():
        raise FileNotFoundError(f"Quantized model dir not found: {quantized_model_dir}")

    ensure_parents(gguf_path)

    convert_script = ensure_llama_cpp_tools(llama_cpp_dir)

    dequantized_dir: Path | None = None
    source_dir = quantized_model_dir
    if args.conversion_mode == "dequantize":
        dequantized_dir = build_dequantized_dir(args, quantized_model_dir)
        dequantize_model_dir(quantized_model_dir, dequantized_dir, args.dequantized_dtype)
        source_dir = dequantized_dir

    convert_to_gguf(convert_script, source_dir, gguf_path, args.outtype)

    if args.register_ollama:
        register_ollama_model(
            model_path=gguf_path,
            model_name=args.ollama_model_name,
            ollama_bin=args.ollama_bin,
            ollama_models_dir=args.ollama_models_dir,
            ollama_host=args.ollama_host,
            ollama_num_ctx=args.ollama_num_ctx,
        )

    if dequantized_dir and not args.keep_dequantized and not args.dequantized_model_dir:
        shutil.rmtree(dequantized_dir, ignore_errors=True)

    print("\nCompleted.")
    print(f"Source quantized HF dir: {quantized_model_dir}")
    print(f"Converted GGUF:          {gguf_path}")


if __name__ == "__main__":
    main()
