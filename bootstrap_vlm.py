"""
bootstrap_vlm.py
Smart CUDA 12.x binary installer & verifier for llama.cpp (llama-server & llama-mtmd-cli)
Targeted for Google Colab NVIDIA Tesla T4 (15 GiB VRAM, Compute Capability 7.5).
"""

import os
import sys
import glob
import shutil
import tarfile
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

DEFAULT_BIN_ROOT = Path("/content/llama-bin")


def detect_cuda_version() -> str:
    """Detect active CUDA runtime version via torch or nvcc, defaulting to 12.6."""
    try:
        import torch
        if torch.cuda.is_available() and torch.version.cuda:
            return str(torch.version.cuda)
    except Exception:
        pass
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            import re
            m = re.search(r"release\s+(\d+\.\d+)", res.stdout)
            if m:
                return m.group(1)
    except Exception:
        pass
    return "12.6"


def link_nvidia_cuda_libs(bin_dir: str) -> str:
    """
    Symlink libcudart.so.12, libcublas.so.12, libcublasLt.so.12 from Python dist-packages
    into the llama.cpp binary directory and update LD_LIBRARY_PATH.
    """
    nvidia_lib_dirs = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")
    system_cuda_dirs = [
        d for d in ["/usr/local/cuda/lib64", "/usr/local/cuda-12/lib64", "/usr/lib/x86_64-linux-gnu"]
        if os.path.isdir(d)
    ]
    all_lib_dirs = [bin_dir] + nvidia_lib_dirs + system_cuda_dirs

    for lib_dir in nvidia_lib_dirs:
        for so_file in glob.glob(os.path.join(lib_dir, "*.so*")):
            dest = os.path.join(bin_dir, os.path.basename(so_file))
            if not os.path.exists(dest):
                try:
                    os.symlink(so_file, dest)
                except Exception:
                    pass

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    extra_ld = ":".join(all_lib_dirs)
    new_ld = f"{extra_ld}:{current_ld}" if current_ld else extra_ld
    os.environ["LD_LIBRARY_PATH"] = new_ld
    return new_ld


def _find_binaries(bin_root: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Locate llama-server, llama-mtmd-cli, and llama-cli inside bin_root."""
    root_str = str(bin_root)
    servers = sorted(glob.glob(f"{root_str}/**/llama-server", recursive=True))
    mtmds = sorted(glob.glob(f"{root_str}/**/llama-mtmd-cli", recursive=True))
    clis = sorted(glob.glob(f"{root_str}/**/llama-cli", recursive=True))
    return (
        servers[0] if servers else None,
        mtmds[0] if mtmds else None,
        clis[0] if clis else None,
    )


def _download_and_extract_archive(url: str, dest_dir: Path) -> None:
    """Download a .tar.gz or .zip release asset and extract it safely into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1].split("?")[0]
    archive_path = dest_dir / fname

    print(f"[+] Mengunduh paket binary CUDA: {fname} ...", flush=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=str(dest_dir))
    elif fname.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=str(dest_dir))

    try:
        archive_path.unlink(missing_ok=True)
    except Exception:
        pass


def _select_aidock_cuda12_asset(assets: list, detected_cuda: str) -> Optional[dict]:
    """
    Strictly select a Linux amd64/x64 CUDA 12.x archive from ai-dock/llama.cpp-cuda.
    Never selects cuda-13 or arm64/aarch64 builds on Colab T4.
    """
    candidates = []
    for asset in assets:
        name = asset.get("name", "").lower()
        if "cuda-13" in name or "arm64" in name or "aarch64" in name or "darwin" in name or "win" in name:
            continue
        if (name.endswith("-amd64.tar.gz") or name.endswith("-x64.tar.gz") or name.endswith(".tar.gz")) and "cuda-12" in name:
            candidates.append(asset)

    if not candidates:
        return None

    priority_tags = [f"cuda-{detected_cuda}", "cuda-12.6", "cuda-12.8", "cuda-12.4", "cuda-12.2", "cuda-12"]
    for ptag in priority_tags:
        for cand in candidates:
            if ptag in cand["name"].lower():
                return cand
    return candidates[0]


def _install_from_ggml_official(dest_dir: Path) -> bool:
    """
    Fallback installer from official ggml-org/llama.cpp releases.
    Downloads both llama-bXXXX-bin-ubuntu-cuda-12.x-x64 and cudart-llama-bXXXX-bin-ubuntu-cuda-12.x-x64.
    """
    resp = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", timeout=25)
    resp.raise_for_status()
    assets = resp.json().get("assets", [])

    main_asset = None
    cudart_asset = None
    for asset in assets:
        name = asset.get("name", "").lower()
        if "cuda-13" in name or "arm64" in name or "aarch64" in name:
            continue
        if "ubuntu" in name and "cuda-12" in name and ("x64" in name or "amd64" in name):
            if name.startswith("cudart-"):
                cudart_asset = asset
            elif name.startswith("llama-"):
                main_asset = asset

    if not main_asset:
        return False

    if cudart_asset:
        _download_and_extract_archive(cudart_asset["browser_download_url"], dest_dir)
    _download_and_extract_archive(main_asset["browser_download_url"], dest_dir)
    return True


def verify_binary_execution(bin_path: str) -> Tuple[bool, str]:
    """Run `<binary> --version` to verify shared libraries and executable permissions."""
    if not bin_path or not os.path.exists(bin_path):
        return False, "Binary path does not exist"
    try:
        os.chmod(bin_path, 0o755)
        res = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=15,
        )
        out = (res.stdout + "\n" + res.stderr).strip()
        if res.returncode == 0 or "version" in out.lower() or "llama" in out.lower():
            return True, out.splitlines()[0] if out else "OK"
        return False, out
    except Exception as e:
        return False, str(e)


def bootstrap_llama_cpp(bin_root: Path = DEFAULT_BIN_ROOT, force_reinstall: bool = False) -> Dict[str, str]:
    """
    Ensure pre-compiled CUDA 12.x llama-server and llama-mtmd-cli are installed, linked, and verified.
    Returns dictionary with resolved paths and environment details.
    """
    if force_reinstall and bin_root.exists():
        shutil.rmtree(bin_root, ignore_errors=True)

    detected_cuda = detect_cuda_version()
    server_bin, mtmd_bin, cli_bin = _find_binaries(bin_root)

    if not server_bin or (not mtmd_bin and not cli_bin):
        print(f"[+] Menyiapkan instalasi llama.cpp CUDA (Runtime CUDA terdeteksi: {detected_cuda})...", flush=True)
        installed = False
        try:
            resp = requests.get("https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest", timeout=25)
            resp.raise_for_status()
            assets = resp.json().get("assets", [])
            selected = _select_aidock_cuda12_asset(assets, detected_cuda)
            if selected:
                _download_and_extract_archive(selected["browser_download_url"], bin_root)
                installed = True
        except Exception as e:
            print(f"[!] Peringatan unduhan ai-dock ({e}), beralih ke rilis resmi ggml-org/llama.cpp...", flush=True)

        if not installed:
            if not _install_from_ggml_official(bin_root):
                raise RuntimeError(
                    "Gagal menemukan paket biner Linux CUDA 12.x baik di ai-dock/llama.cpp-cuda maupun ggml-org/llama.cpp."
                )

        server_bin, mtmd_bin, cli_bin = _find_binaries(bin_root)

    if not server_bin:
        raise RuntimeError(f"Tidak menemukan binary 'llama-server' di dalam {bin_root}.")

    # Fallback if an older archive only has llama-cli or only llama-mtmd-cli
    effective_mtmd = mtmd_bin or cli_bin
    effective_cli = cli_bin or mtmd_bin

    bin_dir = os.path.dirname(server_bin)
    for b in filter(None, [server_bin, effective_mtmd, effective_cli]):
        try:
            os.chmod(b, 0o755)
        except Exception:
            pass

    ld_path = link_nvidia_cuda_libs(bin_dir)

    ok_srv, ver_srv = verify_binary_execution(server_bin)
    if not ok_srv:
        raise RuntimeError(f"Verifikasi llama-server gagal ({server_bin}): {ver_srv}")

    if effective_mtmd:
        verify_binary_execution(effective_mtmd)

    os.environ["VLM_LLAMA_SERVER"] = server_bin
    os.environ["VLM_LLAMA_MTMD_CLI"] = effective_mtmd or ""
    os.environ["VLM_LLAMA_CLI"] = effective_cli or ""

    print(f"[✓] llama-server siap   : {server_bin} ({ver_srv})", flush=True)
    print(f"[✓] llama-mtmd-cli siap : {effective_mtmd}", flush=True)

    return {
        "llama_server": server_bin,
        "llama_mtmd_cli": effective_mtmd or "",
        "llama_cli": effective_cli or "",
        "bin_dir": bin_dir,
        "cuda_version": detected_cuda,
        "ld_library_path": ld_path,
    }


if __name__ == "__main__":
    info = bootstrap_llama_cpp()
    print(info)
