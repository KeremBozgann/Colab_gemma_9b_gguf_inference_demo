"""
Download a HF model checkpoint, convert it to GGUF (F16), then quantize GGUF.

Example:
python hf_to_gguf.py \
  --model-id ytu-ce-cosmos/Turkish-Gemma-9b-v0.1 \
  --full-model-dir /content/models/turkish-gemma-9b-v01-full \
  --gguf-path /content/drive/MyDrive/training-embedding/ollama/downloads/Turkish-Gemma-9b-v0.1.F16.gguf \
  --quantized-gguf-path /content/drive/MyDrive/training-embedding/ollama/downloads/Turkish-Gemma-9b-v0.1.Q4_K_M.gguf \
  --quantization-type q4_k_m
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF checkpoint to GGUF and generate quantized GGUF."
    )
    parser.add_argument(
        "--model-id",
        default="ytu-ce-cosmos/Turkish-Gemma-9b-v0.1",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--full-model-dir",
        required=True,
        help="Local path to store full Hugging Face model checkpoint.",
    )
    parser.add_argument(
        "--gguf-path",
        required=True,
        help="Output path for F16 GGUF.",
    )
    parser.add_argument(
        "--quantized-gguf-path",
        required=True,
        help="Output path for quantized GGUF.",
    )
    parser.add_argument(
        "--quantization-type",
        default="q4_k_m",
        help="llama-quantize type (e.g. q4_k_m, q5_k_m, q8_0).",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="/content/llama.cpp",
        help="Path where llama.cpp is cloned/built.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional HF token. If empty, uses HF_TOKEN env var.",
    )
    parser.add_argument(
        "--register-ollama",
        action="store_true",
        help="After quantization, register the quantized GGUF as an Ollama model.",
    )
    parser.add_argument(
        "--ollama-model-name",
        default="turkish-gemma-v01-q4km",
        help="Target Ollama model name used with --register-ollama.",
    )
    parser.add_argument(
        "--ollama-bin",
        default=os.environ.get("OLLAMA_BIN", "/content/ollama-runtime/bin/ollama"),
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


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_parents(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def resolve_hf_token(explicit_token: str) -> Optional[str]:
    token = explicit_token.strip() or os.environ.get("HF_TOKEN", "").strip()
    return token or None


def ensure_llama_cpp(llama_cpp_dir: Path) -> tuple[Path, Path]:
    git_bin = shutil.which("git")
    cmake_bin = shutil.which("cmake")
    if not git_bin or not cmake_bin:
        raise RuntimeError("git and cmake are required. Install them before running this script.")

    if not (llama_cpp_dir / ".git").exists():
        run([git_bin, "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(llama_cpp_dir)])

    run([cmake_bin, "-S", str(llama_cpp_dir), "-B", str(llama_cpp_dir / "build"), "-DGGML_CUDA=OFF"])
    run([cmake_bin, "--build", str(llama_cpp_dir / "build"), "--target", "llama-quantize", "-j"])

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        raise FileNotFoundError(f"Missing converter script: {convert_script}")
    if not quantize_bin.exists():
        raise FileNotFoundError(f"Missing quantizer binary: {quantize_bin}")

    return convert_script, quantize_bin


def has_complete_checkpoint(full_model_dir: Path) -> bool:
    if not full_model_dir.exists() or not full_model_dir.is_dir():
        return False

    if not (full_model_dir / "config.json").exists():
        return False

    if list(full_model_dir.rglob("*.incomplete")):
        return False

    index_path = full_model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shard_names = {name for name in weight_map.values() if isinstance(name, str)}
        if not shard_names:
            return False
        return all((full_model_dir / shard_name).exists() for shard_name in shard_names)

    has_any_weight = bool(
        list(full_model_dir.glob("*.safetensors"))
        or list(full_model_dir.glob("*.bin"))
        or list(full_model_dir.glob("*.pth"))
    )
    return has_any_weight


def ensure_full_model_downloaded(model_id: str, full_model_dir: Path, hf_token: Optional[str]) -> None:
    if has_complete_checkpoint(full_model_dir):
        print(f"Found complete local checkpoint, reusing: {full_model_dir}")
        return

    full_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading or resuming checkpoint into: {full_model_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(full_model_dir),
        token=hf_token,
    )
    if not has_complete_checkpoint(full_model_dir):
        raise RuntimeError(
            f"Checkpoint directory is still incomplete after download: {full_model_dir}"
        )


def convert_to_gguf(convert_script: Path, full_model_dir: Path, gguf_path: Path) -> None:
    run(
        [
            sys.executable,
            str(convert_script),
            str(full_model_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            "f16",
        ]
    )


def quantize_gguf(quantize_bin: Path, gguf_path: Path, quantized_gguf_path: Path, quantization_type: str) -> None:
    run([str(quantize_bin), str(gguf_path), str(quantized_gguf_path), quantization_type])


def register_ollama_model(
    quantized_gguf_path: Path,
    ollama_bin: Path,
    ollama_model_name: str,
    ollama_models_dir: str,
    ollama_host: str,
    ollama_num_ctx: int,
) -> None:
    if not ollama_bin.exists():
        raise FileNotFoundError(f"Ollama binary not found: {ollama_bin}")
    if not os.access(ollama_bin, os.X_OK):
        raise PermissionError(f"Ollama binary is not executable: {ollama_bin}")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host
    if ollama_models_dir:
        Path(ollama_models_dir).mkdir(parents=True, exist_ok=True)
        env["OLLAMA_MODELS"] = ollama_models_dir

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".modelfile") as handle:
        modelfile_path = Path(handle.name)
        handle.write(f"FROM {quantized_gguf_path}\n")
        handle.write(f"PARAMETER num_ctx {ollama_num_ctx}\n")

    try:
        print(f"Registering Ollama model: {ollama_model_name}")
        subprocess.run(
            [str(ollama_bin), "create", ollama_model_name, "-f", str(modelfile_path)],
            env=env,
            check=True,
        )
        subprocess.run([str(ollama_bin), "list"], env=env, check=True)
    finally:
        modelfile_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    full_model_dir = Path(args.full_model_dir).expanduser().resolve()
    gguf_path = Path(args.gguf_path).expanduser().resolve()
    quantized_gguf_path = Path(args.quantized_gguf_path).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()

    ensure_parents(gguf_path, quantized_gguf_path)
    hf_token = resolve_hf_token(args.hf_token)

    ensure_full_model_downloaded(args.model_id, full_model_dir, hf_token)
    convert_script, quantize_bin = ensure_llama_cpp(llama_cpp_dir)
    convert_to_gguf(convert_script, full_model_dir, gguf_path)
    quantize_gguf(quantize_bin, gguf_path, quantized_gguf_path, args.quantization_type)
    if args.register_ollama:
        register_ollama_model(
            quantized_gguf_path=quantized_gguf_path,
            ollama_bin=Path(args.ollama_bin).expanduser().resolve(),
            ollama_model_name=args.ollama_model_name,
            ollama_models_dir=args.ollama_models_dir,
            ollama_host=args.ollama_host,
            ollama_num_ctx=args.ollama_num_ctx,
        )

    print("\nCompleted.")
    print(f"HF checkpoint dir:   {full_model_dir}")
    print(f"GGUF (F16):          {gguf_path}")
    print(f"GGUF (quantized):    {quantized_gguf_path}")


if __name__ == "__main__":
    main()
