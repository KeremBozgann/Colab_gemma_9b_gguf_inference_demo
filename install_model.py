import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Optional


def _read_hf_token_from_dotenv(dotenv_path: Path) -> Optional[str]:
    if not dotenv_path.exists():
        return None

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key != "HF_TOKEN":
            continue

        token = value.strip().strip('"').strip("'")
        return token or None

    return None


HF_REPO_ID = (
    os.environ.get("HF_REPO_ID", "ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF").strip()
    or "ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF"
)
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "turkish-gemma").strip() or "turkish-gemma"
DRIVE_MOUNT_POINT = "/content/drive"
PROJECT_ROOT = Path(__file__).resolve().parent


def _is_drive_root_usable(candidate: Path) -> bool:
    if not candidate.exists() or not candidate.is_dir():
        return False

    probe = candidate / ".ollama_write_probe"
    try:
        probe.mkdir(parents=False, exist_ok=False)
        probe.rmdir()
        return True
    except OSError:
        return False


def _detect_drive_root() -> Path:
    candidates = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]
    for candidate in candidates:
        if _is_drive_root_usable(candidate):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DRIVE_ROOT = _detect_drive_root()

_ollama_root_env = os.environ.get("OLLAMA_ROOT", "").strip()
_ollama_models_env = os.environ.get("OLLAMA_MODELS", "").strip()
if _ollama_models_env:
    OLLAMA_MODELS_DIR = Path(_ollama_models_env)
    OLLAMA_ROOT = Path(_ollama_root_env) if _ollama_root_env else OLLAMA_MODELS_DIR.parent
else:
    default_root = PROJECT_ROOT / "ollama"
    OLLAMA_ROOT = Path(_ollama_root_env) if _ollama_root_env else default_root
    OLLAMA_MODELS_DIR = OLLAMA_ROOT / "models"

OLLAMA_BIN_DIR = OLLAMA_ROOT / "bin"
OLLAMA_BIN_DRIVE = OLLAMA_BIN_DIR / "ollama"
OLLAMA_BIN_RUNTIME = Path("/content/ollama-runtime/bin/ollama")
OLLAMA_BIN_RUNTIME_LEGACY = Path("/content/ollama-bin/ollama")
OLLAMA_BIN = Path(os.environ.get("OLLAMA_BIN", str(OLLAMA_BIN_DRIVE)))
OLLAMA_BLOBS_DIR = OLLAMA_MODELS_DIR / "blobs"
OLLAMA_MANIFESTS_DIR = OLLAMA_MODELS_DIR / "manifests"
DOWNLOADS_DIR = OLLAMA_ROOT / "downloads"
MODELFILE_PATH = OLLAMA_ROOT / "Modelfile"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
KEEP_SOURCE_GGUF = os.environ.get("KEEP_SOURCE_GGUF", "0") == "1"
GGUF_FILENAME_OVERRIDE = os.environ.get("GGUF_FILENAME", "").strip() or None
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or _read_hf_token_from_dotenv(PROJECT_ENV_PATH)
if HF_TOKEN and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HF_TOKEN

PREFERRED_QUANT_TAGS = [
    "q3_k_m",
    "q4_k_m",
    "q4_0",
    "q5_k_m",
    "q8_0",
    "q2_k",
    "q3_k_s",
    "q4_k_s",
    "q5_k_s",
]


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This script is intended for Google Colab where /content/drive is available."
        ) from exc
    drive.mount(DRIVE_MOUNT_POINT, force_remount=False)


def ensure_directories() -> None:
    def _ensure_real_dir(path: Path) -> None:
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            if not resolved.exists() or not resolved.is_dir():
                path.unlink(missing_ok=True)
                path.mkdir(parents=True, exist_ok=True)
                return
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Expected directory at {path}, found a non-directory entry.")
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        OLLAMA_ROOT,
        OLLAMA_MODELS_DIR,
        OLLAMA_BLOBS_DIR,
        OLLAMA_MANIFESTS_DIR,
        DOWNLOADS_DIR,
    ):
        _ensure_real_dir(path)


def build_ollama_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        f"{OLLAMA_BIN.parent}:{OLLAMA_BIN_RUNTIME.parent}:{OLLAMA_BIN_RUNTIME_LEGACY.parent}:"
        f"{OLLAMA_BIN_DIR}:{env.get('PATH', '')}"
    )
    env["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)
    env["OLLAMA_HOME"] = str(OLLAMA_ROOT)
    env["OLLAMA_HOST"] = OLLAMA_HOST
    return env


def ollama_server_is_ready() -> bool:
    for endpoint in ("/api/version", "/api/tags"):
        try:
            with urllib.request.urlopen(f"http://{OLLAMA_HOST}{endpoint}", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def ensure_ollama_binary_exists() -> None:
    global OLLAMA_BIN

    def _can_execute(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            result = subprocess.run(
                [str(path), "-v"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return result.returncode == 0
        except OSError:
            return False

    binary_candidates = [
        OLLAMA_BIN,
        OLLAMA_BIN_RUNTIME,
        OLLAMA_BIN_RUNTIME_LEGACY,
        OLLAMA_BIN_DRIVE,
    ]
    for candidate in binary_candidates:
        if _can_execute(candidate):
            OLLAMA_BIN = candidate
            os.environ["OLLAMA_BIN"] = str(candidate)
            return

    if OLLAMA_BIN_DRIVE.exists():
        try:
            OLLAMA_BIN_DRIVE.chmod(0o755)
        except OSError:
            pass

        if _can_execute(OLLAMA_BIN_DRIVE):
            OLLAMA_BIN = OLLAMA_BIN_DRIVE
            os.environ["OLLAMA_BIN"] = str(OLLAMA_BIN_DRIVE)
            return

        OLLAMA_BIN_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLLAMA_BIN_DRIVE, OLLAMA_BIN_RUNTIME)
        OLLAMA_BIN_RUNTIME.chmod(0o755)

        if _can_execute(OLLAMA_BIN_RUNTIME):
            OLLAMA_BIN = OLLAMA_BIN_RUNTIME
            os.environ["OLLAMA_BIN"] = str(OLLAMA_BIN_RUNTIME)
            print("Using runtime Ollama binary at /content/ollama-runtime/bin/ollama.")
            return

    raise FileNotFoundError(
        f"Ollama binary is not executable (checked {OLLAMA_BIN}). "
        f"Expected a runnable binary at {OLLAMA_BIN_RUNTIME} or {OLLAMA_BIN_DRIVE}. "
        "Run your startup install cell first."
    )


def ensure_ollama_server_running() -> None:
    if ollama_server_is_ready():
        return
    raise RuntimeError(
        f"Ollama server is not reachable at http://{OLLAMA_HOST}. "
        "Start it with your startup cell before running this script."
    )


def get_repo_gguf_files(repo_id: str) -> Dict[str, Optional[int]]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    gguf_files: Dict[str, Optional[int]] = {}
    for sibling in data.get("siblings", []):
        filename = sibling.get("rfilename")
        if isinstance(filename, str) and filename.lower().endswith(".gguf"):
            size = sibling.get("size")
            gguf_files[filename] = int(size) if isinstance(size, int) else None

    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {repo_id}.")
    return gguf_files


def choose_gguf_file(gguf_files: Dict[str, Optional[int]]) -> str:
    if GGUF_FILENAME_OVERRIDE:
        if GGUF_FILENAME_OVERRIDE not in gguf_files:
            raise ValueError(
                f"GGUF_FILENAME={GGUF_FILENAME_OVERRIDE} was not found in {HF_REPO_ID}."
            )
        return GGUF_FILENAME_OVERRIDE

    def _norm(name: str) -> str:
        return name.lower().replace("-", "_").replace(".", "_")

    def _is_f16(name: str) -> bool:
        normalized = _norm(name)
        return "f16" in normalized or "fp16" in normalized

    def _is_quantized(name: str) -> bool:
        return re.search(r"(^|_)q\d", _norm(name)) is not None

    def _pick_smallest(candidates: list[str]) -> str:
        with_size = [(candidate, gguf_files[candidate]) for candidate in candidates]
        sized = [(name, size) for name, size in with_size if isinstance(size, int)]
        if sized:
            sized.sort(key=lambda item: item[1])
            return sized[0][0]
        return sorted(candidates)[0]

    all_files = list(gguf_files.keys())
    quant_candidates = [name for name in all_files if _is_quantized(name) and not _is_f16(name)]

    for tag in PREFERRED_QUANT_TAGS:
        tagged = [name for name in quant_candidates if tag in _norm(name)]
        if tagged:
            return _pick_smallest(tagged)

    if quant_candidates:
        return _pick_smallest(quant_candidates)

    non_f16_candidates = [name for name in all_files if not _is_f16(name)]
    if non_f16_candidates:
        return _pick_smallest(non_f16_candidates)

    raise RuntimeError(
        "Only F16/FP16 GGUF files were detected. "
        "Set GGUF_FILENAME=<quantized-file.gguf> explicitly."
    )


def format_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "unknown size"
    gib = size_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def download_gguf(repo_id: str, filename: str) -> Path:
    target_path = DOWNLOADS_DIR / filename
    if target_path.exists():
        print(f"Using existing GGUF: {target_path}")
        return target_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=1"
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    print(f"Downloading {filename}...")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        tmp_path.replace(target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return target_path


def write_modelfile(gguf_path: Path) -> None:
    content = f"FROM {gguf_path}\nPARAMETER num_ctx 2048\n"
    MODELFILE_PATH.write_text(content, encoding="utf-8")


def create_ollama_model(env: Dict[str, str]) -> None:
    subprocess.run(
        [str(OLLAMA_BIN), "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
        env=env,
        check=True,
    )


def ollama_model_exists(env: Dict[str, str], model_name: str) -> bool:
    result = subprocess.run(
        [str(OLLAMA_BIN), "show", model_name],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def cleanup_source_file(gguf_path: Path) -> None:
    if KEEP_SOURCE_GGUF:
        return

    if gguf_path.exists():
        gguf_path.unlink()
        print("Removed source GGUF after model import to save disk.")


def main() -> None:
    mount_drive()
    ensure_directories()
    ensure_ollama_binary_exists()
    env = build_ollama_env()
    ensure_ollama_server_running()

    if ollama_model_exists(env, OLLAMA_MODEL_NAME):
        print(f"Model '{OLLAMA_MODEL_NAME}' already exists. Skipping download/import.")
    else:
        if HF_TOKEN:
            print("Using authenticated Hugging Face requests (HF_TOKEN detected).")
        gguf_files = get_repo_gguf_files(HF_REPO_ID)
        chosen_file = choose_gguf_file(gguf_files)
        print(f"Selected GGUF: {chosen_file} ({format_size(gguf_files.get(chosen_file))})")

        gguf_path = download_gguf(HF_REPO_ID, chosen_file)
        write_modelfile(gguf_path)
        create_ollama_model(env)
        cleanup_source_file(gguf_path)

    print("Model setup complete.")
    print(f"Ollama binary: {OLLAMA_BIN}")
    print(f"Ollama models directory: {OLLAMA_MODELS_DIR}")
    print(f"Model name: {OLLAMA_MODEL_NAME}")
    print("To keep the source GGUF file as well, set KEEP_SOURCE_GGUF=1 before running.")
    print("To force a specific quant file, set GGUF_FILENAME=<exact .gguf filename> before running.")


if __name__ == "__main__":
    main()
