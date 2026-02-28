import argparse
import json
from pathlib import Path

from llama_cpp import Llama

# Model + generation settings
CONFIG = {
    "repo_id": "ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF",
    "filename": "Turkish-Gemma-9b-T1.Q4_K_M.gguf",
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
            "Run interactive GGUF chat using a required local/Drive GGUF cache path "
            "via --save_load_path."
        )
    )
    parser.add_argument(
        "--save_load_path",
        "--save-load-path",
        required=True,
        help=(
            "Required file or directory path for GGUF persistence/download cache. "
            "The script downloads via huggingface_hub and loads the local GGUF file."
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
        return ""

    if "</think>" in raw:
        # Keep only content after the last closing think tag.
        tail = raw.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail
        return ""

    return raw


def get_repo_filename() -> str:
    filename = str(CONFIG["filename"]).strip()
    if not filename:
        raise ValueError("CONFIG['filename'] must be a non-empty GGUF filename.")
    return filename


def create_model(save_load_path: str, use_gpu: bool, gpu_layers: int) -> Llama:
    n_gpu_layers = gpu_layers if use_gpu else 0
    print(
        "Backend mode: "
        + ("GPU offload enabled" if use_gpu else "CPU-only")
        + f" (n_gpu_layers={n_gpu_layers})"
    )

    target = Path(save_load_path).expanduser().resolve()
    repo_filename = get_repo_filename()

    # If user gives a file path, use it directly when present; otherwise download there.
    if target.suffix.lower() == ".gguf":
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if target.name != repo_filename:
                raise FileNotFoundError(
                    "Target GGUF file does not exist and filename does not match the configured "
                    f"repo filename. Expected '{repo_filename}', got '{target.name}'. "
                    "Use a matching file path or update CONFIG['filename']."
                )
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise RuntimeError(
                    "huggingface_hub is required for download mode. "
                    "Install it with: pip install huggingface_hub"
                ) from error
            print(f"Downloading {repo_filename} to {target.parent}")
            target = Path(
                hf_hub_download(
                    repo_id=CONFIG["repo_id"],
                    filename=repo_filename,
                    local_dir=str(target.parent),
                    local_files_only=False,
                )
            )
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
    gguf_file = target / repo_filename
    if gguf_file.exists():
        print(f"Loading cached GGUF: {gguf_file}")
        return Llama(
            model_path=str(gguf_file),
            n_ctx=CONFIG["n_ctx"],
            n_threads=CONFIG["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for download mode. "
            "Install it with: pip install huggingface_hub"
        ) from error

    print(f"No cached GGUF found at {gguf_file}. Downloading: {repo_filename}")
    gguf_file = Path(
        hf_hub_download(
            repo_id=CONFIG["repo_id"],
            filename=repo_filename,
            local_dir=str(target),
            local_files_only=False,
        )
    )
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
