import argparse
import fnmatch
import json
from pathlib import Path

from llama_cpp import Llama

# Model + generation settings
CONFIG = {
    "repo_id": "ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF",
    "filename": "*Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_threads": 4,
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.05,
    "max_history_turns": 8,
    "pre_prompt_prefix": "",
    "pre_prompt": "",
    "pre_prompt_suffix": "",
    "input_prefix": "<start_of_turn>user\n",
    "input_suffix": "<end_of_turn>\n<start_of_turn>model\n",
}

HISTORY_PATH = Path("cosmos_t1_chat_history.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run interactive GGUF chat. Optionally cache/load model at --save_load_path "
            "to avoid re-downloading each Colab restart."
        )
    )
    parser.add_argument(
        "--save_load_path",
        "--save-load-path",
        default="",
        help=(
            "Optional file or directory path for GGUF persistence. "
            "If omitted, uses Llama.from_pretrained default behavior."
        ),
    )
    parser.add_argument(
        "--use_gpu",
        "--use-gpu",
        action="store_true",
        help="Enable GPU offload for llama.cpp (uses --gpu_layers, default all layers).",
    )
    parser.add_argument(
        "--gpu_layers",
        "--gpu-layers",
        type=int,
        default=-1,
        help=(
            "Number of layers to offload when --use_gpu is set. "
            "-1 means offload all supported layers."
        ),
    )
    return parser.parse_args()


def load_history() -> list[tuple[str, str]]:
    if not HISTORY_PATH.exists():
        print(f"History file not found, starting fresh: {HISTORY_PATH}")
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"Warning: failed to load history from {HISTORY_PATH}: {error}")
        return []
    history: list[tuple[str, str]] = []
    for item in data:
        user = str(item.get("user", "")).strip()
        assistant = sanitize_assistant_text(str(item.get("assistant", "")).strip())
        if user and assistant:
            history.append((user, assistant))
    return history


def save_history(history: list[tuple[str, str]]) -> None:
    payload = [{"user": u, "assistant": a} for u, a in history]
    HISTORY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_prompt(history: list[tuple[str, str]], user_input: str) -> str:
    prompt = (
        f"{CONFIG['pre_prompt_prefix']}"
        f"{CONFIG['pre_prompt']}"
        f"{CONFIG['pre_prompt_suffix']}"
    )
    for user, assistant in history[-CONFIG["max_history_turns"] :]:
        prompt += (
            f"{CONFIG['input_prefix']}{user}"
            f"{CONFIG['input_suffix']}{assistant}<end_of_turn>\n"
        )

    prompt += f"{CONFIG['input_prefix']}{user_input}{CONFIG['input_suffix']}"
    return prompt


def sanitize_assistant_text(text: str) -> str:
    """
    Keep only the visible final answer for memory.
    This avoids stuffing <think> blocks into history across turns.
    """
    raw = text.strip()
    if not raw:
        return raw

    if "</think>" in raw:
        # Keep only content after the last closing think tag.
        tail = raw.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail

    return raw


def resolve_repo_filename(repo_id: str, pattern: str) -> str:
    if "*" not in pattern and "?" not in pattern and "[" not in pattern:
        return pattern

    try:
        from huggingface_hub import list_repo_files
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to resolve wildcard filename patterns. "
            "Install it with: pip install huggingface_hub"
        ) from error

    candidates = [name for name in list_repo_files(repo_id) if fnmatch.fnmatch(name, pattern)]
    if not candidates:
        raise FileNotFoundError(f"No files in {repo_id} match pattern: {pattern}")
    candidates.sort()
    return candidates[0]


def create_model(save_load_path: str, use_gpu: bool, gpu_layers: int) -> Llama:
    n_gpu_layers = gpu_layers if use_gpu else 0
    print(
        "Backend mode: "
        + ("GPU offload enabled" if use_gpu else "CPU-only")
        + f" (n_gpu_layers={n_gpu_layers})"
    )

    if not save_load_path:
        return Llama.from_pretrained(
            repo_id=CONFIG["repo_id"],
            filename=CONFIG["filename"],
            n_ctx=CONFIG["n_ctx"],
            n_threads=CONFIG["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    target = Path(save_load_path).expanduser().resolve()

    # If user gives a file path, use it directly when present; otherwise download there.
    if target.suffix.lower() == ".gguf":
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            selected_name = resolve_repo_filename(CONFIG["repo_id"], CONFIG["filename"])
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise RuntimeError(
                    "huggingface_hub is required for download mode. "
                    "Install it with: pip install huggingface_hub"
                ) from error
            print(f"Downloading {selected_name} to {target}")
            downloaded_path = Path(
                hf_hub_download(
                repo_id=CONFIG["repo_id"],
                filename=selected_name,
                local_dir=str(target.parent),
                local_files_only=False,
            )
            )
            if downloaded_path.resolve() != target.resolve() and downloaded_path.exists():
                downloaded_path.replace(target)
        print(f"Loading GGUF from local file: {target}")
        return Llama(
            model_path=str(target),
            n_ctx=CONFIG["n_ctx"],
            n_threads=CONFIG["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    # Directory path: reuse existing match first; otherwise download into directory.
    target.mkdir(parents=True, exist_ok=True)
    matches = sorted(target.glob(CONFIG["filename"]))
    if matches:
        gguf_file = matches[0]
        print(f"Loading cached GGUF: {gguf_file}")
        return Llama(
            model_path=str(gguf_file),
            n_ctx=CONFIG["n_ctx"],
            n_threads=CONFIG["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    selected_name = resolve_repo_filename(CONFIG["repo_id"], CONFIG["filename"])
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for download mode. "
            "Install it with: pip install huggingface_hub"
        ) from error

    print(f"No cached GGUF found in {target}. Downloading: {selected_name}")
    downloaded_path = Path(
        hf_hub_download(
        repo_id=CONFIG["repo_id"],
        filename=selected_name,
        local_dir=str(target),
        local_files_only=False,
    )
    )
    gguf_file = downloaded_path
    print(f"Loading downloaded GGUF: {gguf_file}")
    return Llama(
        model_path=str(gguf_file),
        n_ctx=CONFIG["n_ctx"],
        n_threads=CONFIG["n_threads"],
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def main() -> None:
    args = parse_args()
    model = create_model(args.save_load_path, args.use_gpu, args.gpu_layers)

    history = load_history()
    if history:
        print(f"Loaded {len(history)} turns from {HISTORY_PATH}")

    print("Chat started. Commands: /reset, /exit")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_input == "/reset":
            history = []
            save_history(history)
            print("History cleared.")
            continue

        prompt = build_prompt(history, user_input)
        response = model(
            prompt,
            max_tokens=CONFIG["max_tokens"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            top_k=CONFIG["top_k"],
            min_p=CONFIG["min_p"],
            repeat_penalty=CONFIG["repeat_penalty"],
            stop=["<end_of_turn>"],
        )
        choice = response["choices"][0]
        raw_assistant_text = choice["text"].strip()
        assistant_text = sanitize_assistant_text(raw_assistant_text)
        finish_reason = choice.get("finish_reason", "")

        if raw_assistant_text.lstrip().startswith("<think>") and "</think>" not in raw_assistant_text:
            print(
                "Warning: output ended inside a <think> block. "
                "Answer may be incomplete."
            )
        if finish_reason == "length":
            print("Warning: generation hit max token limit before natural stop.")

        print(f"Assistant: {assistant_text}\n")

        history.append((user_input, assistant_text))
        history = history[-CONFIG["max_history_turns"] :]
        save_history(history)


if __name__ == "__main__":
    main()
