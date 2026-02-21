from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import urllib.request
from pathlib import Path

OLLAMA_HOST = "127.0.0.1:11434"


def _run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs: dict = {"check": check, "text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, **kwargs)


def _resolve_project_root() -> Path:
    candidates = [
        Path("/content/drive/MyDrive/training-embedding"),
        Path("/content/drive/My Drive/training-embedding"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError("training-embedding folder not found under mounted Drive.")


def _map_arch() -> str:
    machine = os.uname().machine
    if machine == "x86_64":
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported architecture: {machine}")


def _ensure_zstd() -> None:
    if shutil.which("zstd"):
        return

    _run(["apt-get", "update", "-qq"])
    _run(["apt-get", "install", "-y", "-qq", "zstd"])


def _validate_tarball(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    result = subprocess.run(
        ["tar", "--zstd", "-tf", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _ensure_tarball(path: Path, arch: str) -> None:
    if _validate_tarball(path):
        return

    path.unlink(missing_ok=True)
    _run(
        [
            "curl",
            "-fL",
            f"https://ollama.com/download/ollama-linux-{arch}.tar.zst",
            "-o",
            str(path),
        ]
    )


def _extract_runtime(runtime_root: Path, runtime_bin: Path, tarball: Path) -> None:
    shutil.rmtree(runtime_root, ignore_errors=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    _run(["tar", "--zstd", "-xf", str(tarball), "-C", str(runtime_root)])
    runtime_bin.chmod(0o755)


def _start_ollama_server(runtime_bin: Path, models_dir: Path, log_path: Path) -> None:
    subprocess.run(
        ["pkill", "-f", f"{runtime_bin} serve"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_MODELS": str(models_dir),
            "OLLAMA_HOST": OLLAMA_HOST,
        }
    )

    with log_path.open("ab") as log_file:
        subprocess.Popen(
            [str(runtime_bin), "serve"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _endpoint_ok(path: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{OLLAMA_HOST}{path}", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _wait_for_server(log_path: Path, timeout_seconds: int = 30) -> None:
    for _ in range(timeout_seconds):
        if _endpoint_ok("/api/version"):
            return
        import time

        time.sleep(1)

    print("Ollama server did not become ready. Last logs:")
    subprocess.run(["tail", "-n", "120", str(log_path)], check=False)
    raise RuntimeError("Ollama server failed to start.")


def _get_json(path: str) -> dict:
    with urllib.request.urlopen(f"http://{OLLAMA_HOST}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _ensure_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        if path.is_symlink():
            path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _run_install_model(project_root: Path, models_dir: Path, runtime_bin: Path) -> None:
    os.environ["OLLAMA_MODELS"] = str(models_dir)
    os.environ["OLLAMA_BIN"] = str(runtime_bin)
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST

    script_path = project_root / "install_model.py"
    if not script_path.exists():
        raise FileNotFoundError(f"install_model.py not found: {script_path}")

    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    global project_root, ollama_root, ollama_models, ollama_bin

    project_root = str(_resolve_project_root())
    project_root_path = Path(project_root)

    ollama_root = str(project_root_path / "ollama")
    ollama_models = str(Path(ollama_root) / "models")

    arch = _map_arch()
    tarball = Path(f"/content/ollama-linux-{arch}.tar.zst")
    runtime_root = Path("/content/ollama-runtime")
    runtime_bin_path = runtime_root / "bin" / "ollama"
    ollama_bin = str(runtime_bin_path)
    log_path = Path("/content/ollama-serve.log")

    _ensure_zstd()

    _ensure_dir(Path(ollama_root))
    _ensure_dir(Path(ollama_models))
    _ensure_dir(Path(ollama_models) / "blobs")
    _ensure_dir(Path(ollama_models) / "manifests")

    _ensure_tarball(tarball, arch)
    _extract_runtime(runtime_root, runtime_bin_path, tarball)
    _start_ollama_server(runtime_bin_path, Path(ollama_models), log_path)
    _wait_for_server(log_path)

    print(f"project_root={project_root}")
    print(f"ollama_models={ollama_models}")
    _run([str(runtime_bin_path), "-v"], check=True)
    print(json.dumps(_get_json("/api/version"), ensure_ascii=False))

    print(json.dumps(_get_json("/api/tags"), ensure_ascii=False))
    _run_install_model(project_root_path, Path(ollama_models), runtime_bin_path)
    print(json.dumps(_get_json("/api/tags"), ensure_ascii=False))


if __name__ == "__main__":
    main()
