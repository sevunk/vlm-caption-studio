"""
core/dataset_manager.py
Dataset & File Operations Manager:
- Fast dataset scanning with image dimensions (width/height, low-res <512px flag), word counts, and token estimates
- Safe `.zip` extraction ignoring `__MACOSX`, `.DS_Store`, `._*`, hidden files, with collision-free flattening and Pillow verification
- Single-item caption save & single-item image+txt deletion
- Bulk Trigger Word & Find/Replace with Auto-Backup & Restore
- Smart File Manager (9 Actions + One-Click Google Drive Export/Import)
"""

import os
import glob
import time
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from core.text_cleaner import process_caption_text, estimate_token_count


def _resolve_writable_default(preferred_path: str, fallback_subdir: str) -> str:
    """Use `/content/...` on Google Colab when writable; otherwise fall back cleanly to `/tmp/...`."""
    try:
        os.makedirs(preferred_path, exist_ok=True)
        return preferred_path
    except PermissionError:
        alt = os.path.join(tempfile.gettempdir(), fallback_subdir)
        os.makedirs(alt, exist_ok=True)
        return alt


DEFAULT_DATASET_DIR = _resolve_writable_default(
    os.environ.get("VLM_DATASET_DIR", "/content/dataset"), "vlm_dataset"
)
BACKUP_DIR = _resolve_writable_default(
    os.environ.get("VLM_BACKUP_DIR", "/content/txt_backup"), "vlm_txt_backup"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def verify_and_inspect_image(img_path: Path) -> Tuple[bool, int, int]:
    """
    Safely verify an image file with Pillow (`verify()` on first handle, dimensions on second handle).
    If corrupted, returns `(False, 0, 0)`.
    """
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im2:
            w, h = im2.size
            return True, int(w), int(h)
    except Exception:
        return False, 0, 0


def scan_dataset_items(folder_path: str = DEFAULT_DATASET_DIR) -> List[Dict[str, Any]]:
    """
    Scan dataset directory and return sorted list of image items with caption, words,
    estimated tokens, image dimensions (`width`, `height`), and quality flags (`low_res`).
    """
    folder = Path(folder_path)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        folder = Path(DEFAULT_DATASET_DIR)
        folder.mkdir(parents=True, exist_ok=True)

    imgs = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )

    items: List[Dict[str, Any]] = []
    for p in imgs:
        txt_p = p.with_suffix(".txt")
        cap = ""
        if txt_p.is_file():
            try:
                cap = txt_p.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                pass

        width, height = 0, 0
        try:
            with Image.open(p) as im:
                width, height = im.size
        except Exception:
            pass

        stats = estimate_token_count(cap)
        low_res = bool(width > 0 and height > 0 and min(width, height) < 512)

        items.append({
            "name": p.name,
            "img_path": str(p),
            "txt_path": str(txt_p),
            "caption": cap,
            "words": stats["words"],
            "tokens": stats["tokens"],
            "width": width,
            "height": height,
            "low_res": low_res,
        })
    return items


def save_single_caption(img_path: str, caption: str) -> Dict[str, Any]:
    """Save or delete `.txt` caption file corresponding to `img_path`."""
    p = Path(img_path)
    txt_path = p.with_suffix(".txt")
    clean_cap = (caption or "").strip()
    if clean_cap:
        txt_path.write_text(clean_cap, encoding="utf-8")
    elif txt_path.exists():
        txt_path.unlink(missing_ok=True)
    stats = estimate_token_count(clean_cap)
    return {"ok": True, "words": stats["words"], "tokens": stats["tokens"]}


def delete_single_item(img_path: str) -> Dict[str, Any]:
    """Delete an image and its companion `.txt` file from the dataset."""
    p = Path(img_path)
    txt_p = p.with_suffix(".txt")
    removed = False
    if p.is_file():
        p.unlink(missing_ok=True)
        removed = True
    if txt_p.is_file():
        txt_p.unlink(missing_ok=True)
    return {"ok": removed, "deleted_path": str(p)}


def _is_junk_archive_member(member_name: str) -> bool:
    """Return True if a ZIP entry is a macOS/Linux hidden or metadata artifact."""
    norm = member_name.replace("\\", "/")
    parts = [pt for pt in norm.split("/") if pt]
    if not parts:
        return True
    for pt in parts:
        if pt in ("__MACOSX", ".DS_Store", "Thumbs.db") or pt.startswith("._") or pt.startswith("."):
            return True
    return False


def _unique_stem_in_folder(folder: Path, desired_stem: str, img_ext: str) -> str:
    """Return a stem that does not collide with any existing image or txt file in `folder`."""
    candidate = desired_stem
    counter = 1
    while (folder / f"{candidate}{img_ext}").exists() or (folder / f"{candidate}.txt").exists():
        candidate = f"{desired_stem}_{counter}"
        counter += 1
    return candidate


def safe_extract_zip_to_dataset(zip_path: Path, target_folder: Path) -> Tuple[int, int]:
    """
    Safely extract a `.zip` archive into `target_folder`:
    - Ignores `__MACOSX`, `.DS_Store`, `._*`, and hidden files.
    - Flattens subdirectories while keeping image + `.txt` pairs matched even when renamed for collision avoidance.
    - Verifies every extracted image with Pillow and deletes corrupted image files automatically.
    Returns `(added_images_count, rejected_corrupt_count)`.
    """
    tmp_extract = target_folder / f"__tmp_zip_{int(time.time() * 1000)}"
    tmp_extract.mkdir(parents=True, exist_ok=True)
    added = 0
    corrupt = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir() or _is_junk_archive_member(member.filename):
                    continue
                zf.extract(member, path=str(tmp_extract))

        all_files = [p for p in tmp_extract.rglob("*") if p.is_file() and not _is_junk_archive_member(p.name)]
        img_files = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
        orphan_txts = set(p for p in all_files if p.suffix.lower() == ".txt")

        for img_p in sorted(img_files, key=lambda x: str(x).lower()):
            ok_img, _, _ = verify_and_inspect_image(img_p)
            companion_txt = img_p.with_suffix(".txt")
            if companion_txt in orphan_txts:
                orphan_txts.remove(companion_txt)

            if not ok_img:
                corrupt += 1
                img_p.unlink(missing_ok=True)
                continue

            ext = img_p.suffix.lower()
            safe_stem = _unique_stem_in_folder(target_folder, img_p.stem, ext)
            dst_img = target_folder / f"{safe_stem}{ext}"
            shutil.move(str(img_p), str(dst_img))
            added += 1

            if companion_txt.is_file():
                dst_txt = target_folder / f"{safe_stem}.txt"
                shutil.move(str(companion_txt), str(dst_txt))

        for txt_p in orphan_txts:
            dst_txt = target_folder / txt_p.name
            if not dst_txt.exists():
                shutil.move(str(txt_p), str(dst_txt))

    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

    return added, corrupt


def process_uploaded_file_bytes(
    folder_path: str,
    filename: str,
    content_bytes: bytes,
) -> Tuple[int, int]:
    """
    Save an uploaded file (image, `.txt`, or `.zip`) into `folder_path` with corruption verification.
    Returns `(added_count, corrupt_count)`.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename or f"upload_{int(time.time())}.png").name
    if _is_junk_archive_member(safe_name):
        return 0, 0

    suffix = Path(safe_name).suffix.lower()
    if suffix == ".zip":
        tmp_zip = folder / f"__upload_{int(time.time() * 1000)}_{safe_name}"
        tmp_zip.write_bytes(content_bytes)
        return safe_extract_zip_to_dataset(tmp_zip, folder)

    if suffix in IMAGE_EXTENSIONS:
        dst_path = folder / safe_name
        dst_path.write_bytes(content_bytes)
        ok_img, _, _ = verify_and_inspect_image(dst_path)
        if not ok_img:
            dst_path.unlink(missing_ok=True)
            return 0, 1
        return 1, 0

    if suffix == ".txt":
        dst_path = folder / safe_name
        dst_path.write_bytes(content_bytes)
        return 1, 0

    return 0, 0


def create_dataset_zip(folder_path: str = DEFAULT_DATASET_DIR) -> Path:
    """Create a downloadable `.zip` archive of all valid files in `folder_path`."""
    fdir = Path(folder_path)
    files = [
        p for p in fdir.iterdir()
        if p.is_file() and not p.name.startswith(".") and not p.name.startswith("__tmp_")
    ] if fdir.exists() else []

    zip_dir = Path("/content") if os.access("/content", os.W_OK) else Path(tempfile.gettempdir())
    out_zip = zip_dir / f"dataset_labeled_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files, key=lambda x: x.name.lower()):
            zf.write(p, arcname=p.name)
    return out_zip


def run_bulk_caption_edit(
    folder_path: str,
    action_type: str = "apply",
    trigger: str = "",
    pos: str = "front",
    find_str: str = "",
    replace_str: str = "",
    lowercase: bool = False,
    skip_dup: bool = True,
    clean_commas: bool = True,
    backup: bool = True,
    selected_img_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply bulk trigger word insertion and/or find-and-replace across `.txt` files in `folder_path`
    (or only on `selected_img_paths` if provided), with automatic backup to `BACKUP_DIR`.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    if action_type == "restore":
        bfiles = glob.glob(os.path.join(BACKUP_DIR, "*.txt"))
        for bf in bfiles:
            shutil.copy2(bf, os.path.join(folder_path, os.path.basename(bf)))
        return {
            "ok": True,
            "msg": f"Berhasil memulihkan {len(bfiles)} file .txt dari cadangan (backup).",
            "items": scan_dataset_items(folder_path),
        }

    if selected_img_paths:
        target_txts = []
        for img_p in selected_img_paths:
            tp = Path(img_p).with_suffix(".txt")
            if not tp.exists() and trigger.strip():
                tp.write_text("", encoding="utf-8")
            if tp.exists():
                target_txts.append(str(tp))
    else:
        target_txts = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

    pos_code = "front" if pos == "front" else "back"
    mod_cnt = 0

    for fpath in target_txts:
        fname = os.path.basename(fpath)
        orig = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        if backup:
            bp = os.path.join(BACKUP_DIR, fname)
            if not os.path.exists(bp):
                Path(bp).write_text(orig, encoding="utf-8")

        cur = orig.replace(find_str, replace_str) if find_str else orig
        if trigger.strip():
            updated = process_caption_text(cur, trigger, pos_code, lowercase, skip_dup, clean_commas)
        else:
            updated = process_caption_text(cur, "", pos_code, lowercase, False, clean_commas)

        if updated is not None and updated != orig:
            Path(fpath).write_text(updated, encoding="utf-8")
            mod_cnt += 1

    return {
        "ok": True,
        "msg": f"Berhasil memperbarui {mod_cnt} file caption (.txt).",
        "items": scan_dataset_items(folder_path),
    }


def run_smart_file_manager(
    action: str,
    source_path: str,
    destination_path: str = "/content/dataset.zip",
    rename_prefix: str = "dataset_img",
    active_dataset_dir: str = DEFAULT_DATASET_DIR,
) -> Dict[str, Any]:
    """
    Execute any of the 9 Smart File Manager actions + Google Drive Export/Import actions.
    """
    log = [f"=== SMART FILE MANAGER: {action.upper()} ===", f"[+] Source Path      : {source_path}"]
    if destination_path:
        log.append(f"[+] Destination Path : {destination_path}")

    try:
        if action == "Count Statistics":
            if not os.path.exists(source_path):
                log.append("[!] Path sumber tidak ditemukan.")
            elif os.path.isfile(source_path):
                log.append(f"-> Ukuran File: {os.path.getsize(source_path) / (1024 * 1024):.2f} MB")
            else:
                all_f = [
                    f for f in glob.glob(os.path.join(source_path, "**/*"), recursive=True)
                    if os.path.isfile(f)
                ]
                ext_counts: Dict[str, int] = {}
                tot_sz = 0
                for f in all_f:
                    ext = os.path.splitext(f)[1].lower() or "no_ext"
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    tot_sz += os.path.getsize(f)
                items = scan_dataset_items(source_path)
                cap_cnt = sum(1 for x in items if x["caption"])
                low_res_cnt = sum(1 for x in items if x["low_res"])
                log.append(f"-> Total File          : {len(all_f)}")
                log.append(f"-> Total Ukuran        : {tot_sz / (1024 * 1024):.2f} MB")
                log.append(f"-> Gambar Ber-Caption  : {cap_cnt} / {len(items)} gambar")
                log.append(f"-> Gambar Low-Res <512 : {low_res_cnt} gambar")
                for ext, c in sorted(ext_counts.items()):
                    log.append(f"   • {ext:<10} : {c} file")

        elif action == "Zip Folder":
            if os.path.isdir(source_path):
                out_zip = destination_path if destination_path.endswith(".zip") else destination_path + ".zip"
                os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
                shutil.make_archive(out_zip[:-4], "zip", source_path)
                log.append(f"[✓] Berhasil di-ZIP ke '{out_zip}' ({os.path.getsize(out_zip) / (1024 * 1024):.2f} MB)")
            else:
                log.append("[!] Folder sumber tidak ditemukan.")

        elif action == "Unzip File":
            if os.path.isfile(source_path) and source_path.endswith(".zip"):
                dest_dir = destination_path[:-4] if destination_path.endswith(".zip") else destination_path
                os.makedirs(dest_dir, exist_ok=True)
                tmp_copy = Path(dest_dir) / f"__tmp_mgr_{int(time.time())}.zip"
                shutil.copy2(source_path, tmp_copy)
                added, corrupt = safe_extract_zip_to_dataset(tmp_copy, Path(dest_dir))
                log.append(f"[✓] Ekstraksi aman selesai ke '{dest_dir}' ({added} gambar valid, {corrupt} korup diabaikan).")
            else:
                log.append("[!] File ZIP sumber tidak ditemukan.")

        elif action in ("Copy", "Export to Google Drive", "Import from Google Drive"):
            if not os.path.exists(source_path):
                log.append(f"[!] Path sumber '{source_path}' tidak ditemukan. Pastikan Google Drive sudah di-mount jika mengakses /content/drive.")
            else:
                if os.path.isdir(source_path):
                    os.makedirs(destination_path, exist_ok=True)
                    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                log.append(f"[✓] Berhasil disalin ke '{destination_path}'")

        elif action == "Move":
            if os.path.exists(source_path):
                os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                shutil.move(source_path, destination_path)
                log.append(f"[✓] Berhasil dipindahkan ke '{destination_path}'")
            else:
                log.append("[!] Path sumber tidak ditemukan.")

        elif action == "Delete":
            if os.path.isdir(source_path):
                shutil.rmtree(source_path)
                os.makedirs(source_path, exist_ok=True)
                log.append(f"[✓] Isi folder '{source_path}' berhasil dikosongkan.")
            elif os.path.isfile(source_path):
                os.remove(source_path)
                log.append(f"[✓] File '{source_path}' berhasil dihapus.")
            else:
                log.append("[!] Path sumber tidak ditemukan.")

        elif action == "Sync Images & Captions":
            if os.path.isdir(source_path):
                img_map, txt_map = {}, {}
                for f in os.listdir(source_path):
                    p = os.path.join(source_path, f)
                    if os.path.isfile(p):
                        b, e = os.path.splitext(f)
                        if e.lower() in IMAGE_EXTENSIONS:
                            img_map[b] = f
                        elif e.lower() == ".txt":
                            txt_map[b] = f
                orphaned_txt = [txt_map[b] for b in txt_map if b not in img_map and b != "classes"]
                for f in orphaned_txt:
                    os.remove(os.path.join(source_path, f))
                    log.append(f"   [Deleted Orphan TXT] {f}")
                log.append(f"[✓] Sinkronisasi selesai ({len(orphaned_txt)} file .txt yatim dihapus).")
            else:
                log.append("[!] Folder sumber tidak ditemukan.")

        elif action == "Batch Rename":
            if os.path.isdir(source_path):
                pairs, seen = [], set()
                for f in sorted(os.listdir(source_path)):
                    p = os.path.join(source_path, f)
                    if os.path.isfile(p):
                        b, e = os.path.splitext(f)
                        if e.lower() in IMAGE_EXTENSIONS and b not in seen:
                            txt_f = b + ".txt"
                            has_txt = os.path.exists(os.path.join(source_path, txt_f))
                            pairs.append((f, txt_f if has_txt else None))
                            seen.add(b)
                temp_pairs = []
                for idx, (img_f, txt_f) in enumerate(pairs, 1):
                    new_b = f"{rename_prefix}_{str(idx).zfill(3)}"
                    _, ext = os.path.splitext(img_f)
                    t_img = f"__tmp_{new_b}{ext}"
                    os.rename(os.path.join(source_path, img_f), os.path.join(source_path, t_img))
                    t_txt = None
                    if txt_f:
                        t_txt = f"__tmp_{new_b}.txt"
                        os.rename(os.path.join(source_path, txt_f), os.path.join(source_path, t_txt))
                    temp_pairs.append((t_img, t_txt, new_b + ext, new_b + ".txt" if txt_f else None))
                for t_img, t_txt, f_img, f_txt in temp_pairs:
                    os.rename(os.path.join(source_path, t_img), os.path.join(source_path, f_img))
                    if t_txt and f_txt:
                        os.rename(os.path.join(source_path, t_txt), os.path.join(source_path, f_txt))
                log.append(f"[✓] Batch rename selesai untuk {len(pairs)} pasang gambar + caption!")
            else:
                log.append("[!] Folder sumber tidak ditemukan.")

        elif action == "Clean Empty Files":
            if os.path.isdir(source_path):
                empty = [
                    f for f in glob.glob(os.path.join(source_path, "*"))
                    if os.path.isfile(f) and os.path.getsize(f) == 0
                ]
                for f in empty:
                    os.remove(f)
                    log.append(f"   [Deleted Empty] {os.path.basename(f)}")
                log.append(f"[✓] Dihapus {len(empty)} file berukuran 0 byte.")
            else:
                log.append("[!] Folder sumber tidak ditemukan.")

    except Exception as e:
        log.append(f"[!] Error saat menjalankan '{action}': {e}")

    return {
        "ok": True,
        "log": "\n".join(log),
        "items": scan_dataset_items(active_dataset_dir),
    }
