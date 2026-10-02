# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download helpers shared by the per-source preparation modules.

Everything a preparer downloads goes to ``<data_root>/.raw/<repo>/`` first and is then
moved/extracted into ``<data_root>/<benchmark>/``. Raw files are removed afterwards unless
``--keep-raw`` is given. All helpers retry transient network errors and are idempotent:
files that already exist (and, for images, decode) are not downloaded again.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


USER_AGENT = "Awesome-Perception-Aware-RLVR-eval-prepare/1.0 (+https://github.com)"


@dataclass
class PrepareContext:
    data_root: Path
    force: bool = False
    keep_raw: bool = False
    hf_workers: int = 8
    http_workers: int = 16
    retries: int = 5
    skip_space_check: bool = False
    log: Callable[[str], None] = print
    _downloaded_raw: list[Path] = field(default_factory=list)

    @property
    def raw_root(self) -> Path:
        return self.data_root / ".raw"

    def raw_dir(self, repo_id: str) -> Path:
        path = self.raw_root / repo_id.replace("/", "__")
        if path not in self._downloaded_raw:
            self._downloaded_raw.append(path)
        return path

    def discard_raw(self, path: Path) -> None:
        """Delete a raw download once its content has been extracted (unless --keep-raw)."""
        if self.keep_raw:
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    def finalize(self) -> None:
        """Remove the raw download folders used by this run (unless --keep-raw)."""
        if self.keep_raw:
            return
        for path in self._downloaded_raw:
            shutil.rmtree(path, ignore_errors=True)
        try:
            self.raw_root.rmdir()  # only succeeds when nothing else is left
        except OSError:
            pass


class PrepareError(RuntimeError):
    """A preparation step failed in a way the user has to act on."""


def bypass_proxy_for_mirror() -> None:
    """hf-mirror.com redirects to huggingface.co when it is reached through an HTTP proxy, which
    makes downloads fail; talk to the mirror directly unless HF_MIRROR_BYPASS_PROXY=0 (same rule
    as scripts/data/prepare_train_data.py)."""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror" not in endpoint or os.environ.get("HF_MIRROR_BYPASS_PROXY", "1") == "0":
        return
    hosts = ["hf-mirror.com", ".hf-mirror.com", ".hf.co"]
    for key in ("NO_PROXY", "no_proxy"):
        current = [item for item in os.environ.get(key, "").split(",") if item]
        os.environ[key] = ",".join(current + [host for host in hosts if host not in current])


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def marker_path(target_dir: Path, key: str) -> Path:
    return target_dir / f".prepared-{key}.json"


def is_prepared(target_dir: Path, key: str, outputs: Iterable[Path]) -> bool:
    if not marker_path(target_dir, key).exists():
        return False
    return all(path.exists() for path in outputs)


def write_marker(target_dir: Path, key: str, info: dict[str, Any]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": key, "prepared_at": datetime.now().astimezone().isoformat(timespec="seconds"), **info}
    path = marker_path(target_dir, key)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Hugging Face downloads
# ---------------------------------------------------------------------------


def _non_retryable_hf_errors() -> tuple[type[BaseException], ...]:
    try:
        from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover - very old huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError
    return (EntryNotFoundError, GatedRepoError, RepositoryNotFoundError)


def hf_download(ctx: PrepareContext, repo_ids: str | list[str], filename: str) -> Path:
    """Download one file of a dataset repo into the raw area; ``repo_ids`` lists fallbacks."""
    from huggingface_hub import hf_hub_download

    candidates = [repo_ids] if isinstance(repo_ids, str) else list(repo_ids)
    non_retryable = _non_retryable_hf_errors()
    last_error: BaseException | None = None
    for repo_id in candidates:
        for attempt in range(1, ctx.retries + 1):
            try:
                ctx.log(f"  [hf] {repo_id}/{filename}")
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(ctx.raw_dir(repo_id)),
                )
                return Path(path)
            except non_retryable as exc:
                last_error = exc
                break
            except Exception as exc:  # network errors surface as many exception types
                last_error = exc
                if attempt == ctx.retries:
                    break
                wait = min(5 * attempt, 30)
                ctx.log(f"  [retry {attempt}/{ctx.retries - 1}] {type(exc).__name__}: {exc}; sleeping {wait}s")
                time.sleep(wait)
    raise PrepareError(f"could not download {filename} from {candidates}: {last_error}")


def hf_snapshot(
    ctx: PrepareContext, repo_ids: str | list[str], patterns: list[str], *, local_dir: Path | None = None
) -> Path:
    """Download the files matching ``patterns``; retries reuse already finished files.

    ``local_dir`` downloads in place (for large sources, so that an interrupted run resumes
    without re-downloading files that were already moved out of the raw area).
    """
    from huggingface_hub import snapshot_download

    candidates = [repo_ids] if isinstance(repo_ids, str) else list(repo_ids)
    non_retryable = _non_retryable_hf_errors()
    last_error: BaseException | None = None
    for repo_id in candidates:
        workers = ctx.hf_workers
        for attempt in range(1, ctx.retries + 1):
            try:
                ctx.log(f"  [hf] {repo_id} {patterns} (workers={workers})")
                target = local_dir if local_dir is not None else ctx.raw_dir(repo_id)
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(target),
                    allow_patterns=patterns,
                    max_workers=workers,
                )
                return target
            except non_retryable as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                if attempt == ctx.retries:
                    break
                workers = max(1, workers // 2)
                wait = min(5 * attempt, 30)
                ctx.log(
                    f"  [retry {attempt}/{ctx.retries - 1}] {type(exc).__name__}: {exc}; retrying with {workers} worker(s)"
                )
                time.sleep(wait)
    raise PrepareError(f"could not download {patterns} from {candidates}: {last_error}")


# ---------------------------------------------------------------------------
# Plain HTTP downloads (COCO / Visual Genome images)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadItem:
    urls: tuple[str, ...]
    dest: Path


def is_valid_image_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_one(item: DownloadItem, retries: int, timeout: float) -> str | None:
    """Return None on success or a short error description."""
    if is_valid_image_file(item.dest):
        return None
    errors = []
    for url in item.urls:
        for attempt in range(1, retries + 1):
            try:
                payload = _fetch(url, timeout)
                from PIL import Image

                with Image.open(BytesIO(payload)) as image:
                    image.verify()
                item.dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = item.dest.with_name(item.dest.name + f".tmp{threading.get_ident()}")
                tmp.write_bytes(payload)
                os.replace(tmp, item.dest)
                return None
            except urllib.error.HTTPError as exc:
                errors.append(f"{url}: HTTP {exc.code}")
                if exc.code in {403, 404, 410}:
                    break  # try the next mirror URL
            except Exception as exc:  # noqa: BLE001 - timeouts, truncated payloads, DNS, ...
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
    return "; ".join(errors[-3:]) or "unknown error"


def http_download_many(
    ctx: PrepareContext,
    items: list[DownloadItem],
    *,
    desc: str,
    timeout: float = 60.0,
) -> dict[Path, str]:
    """Download ``items`` concurrently. Returns ``{dest: error}`` for the ones that failed."""
    pending = [item for item in items if not is_valid_image_file(item.dest)]
    ctx.log(f"  [http] {desc}: {len(items) - len(pending)} present, {len(pending)} to download")
    failures: dict[Path, str] = {}
    if not pending:
        return failures
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, ctx.http_workers)) as pool:
        futures = {pool.submit(_download_one, item, ctx.retries, timeout): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            error = future.result()
            if error is not None:
                failures[item.dest] = error
            done += 1
            if done % 200 == 0 or done == len(pending):
                ctx.log(f"  [http] {desc}: {done}/{len(pending)} ({len(failures)} failed)")
    return failures


# ---------------------------------------------------------------------------
# Partial extraction from a remote zip (HTTP range requests)
# ---------------------------------------------------------------------------


class RangeNotSupported(PrepareError):
    """The server ignored the Range header; the archive has to be downloaded in full."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int


class RemoteZip:
    """Read single members of a large zip on the Hugging Face Hub without downloading all of it.

    Only the central directory (a few MB) and the bytes of the requested members are fetched,
    with HTTP range requests against the file's resolve URL (hf-mirror and the Hub's CDN both
    support ranges). The signed CDN URL that the resolve URL redirects to is cached and
    refreshed when it expires.
    """

    _MAX_URL_AGE = 1800.0

    def __init__(self, repo_id: str, filename: str, *, retries: int = 5, timeout: float = 120.0):
        from huggingface_hub import hf_hub_url

        self.source_url = hf_hub_url(repo_id, filename, repo_type="dataset")
        self.retries = retries
        self.timeout = timeout
        self._url: str | None = None
        self._resolved_at = 0.0
        self._lock = threading.Lock()
        self.size = self._content_length()

    def _request(self, url: str, start: int, end: int) -> tuple[bytes, str, int]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read(), response.geturl(), response.status

    def _range(self, start: int, end: int) -> bytes:
        """Bytes ``start``..``end`` (inclusive)."""
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            with self._lock:
                url = self._url if self._url and time.time() - self._resolved_at < self._MAX_URL_AGE else None
            try:
                payload, final_url, status = self._request(url or self.source_url, start, end)
                if status != 206:
                    raise RangeNotSupported(f"{self.source_url} does not support HTTP range requests")
                if url is None:
                    with self._lock:
                        self._url, self._resolved_at = final_url, time.time()
                if len(payload) != end - start + 1:
                    raise OSError(f"short read: {len(payload)} of {end - start + 1} bytes")
                return payload
            except RangeNotSupported:
                raise
            except Exception as exc:  # noqa: BLE001 - expired signature, timeout, reset, ...
                last_error = exc
                with self._lock:
                    self._url = None
                if attempt < self.retries:
                    time.sleep(min(2 * attempt, 10))
        raise PrepareError(f"range request {start}-{end} on {self.source_url} failed: {last_error}")

    def _content_length(self) -> int:
        request = urllib.request.Request(self.source_url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206:
                        raise RangeNotSupported(f"{self.source_url} does not support HTTP range requests")
                    content_range = response.headers.get("Content-Range", "")
                    self._url, self._resolved_at = response.geturl(), time.time()
                    return int(content_range.rsplit("/", 1)[-1])
            except RangeNotSupported:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 * attempt, 10))
        raise PrepareError(f"could not reach {self.source_url}: {last_error}")

    def entries(self) -> list[ZipEntry]:
        """Parse the (zip64-aware) central directory."""
        import struct

        tail_size = min(self.size, 1 << 16)
        tail = self._range(self.size - tail_size, self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise PrepareError(f"{self.source_url}: end of central directory not found")
        _, _, _, _, total, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[eocd : eocd + 22])
        zip64 = tail.rfind(b"PK\x06\x06")
        if zip64 >= 0 and (cd_offset == 0xFFFFFFFF or total == 0xFFFF or cd_size == 0xFFFFFFFF):
            values = struct.unpack("<IQHHIIQQQQ", tail[zip64 : zip64 + 56])
            total, cd_size, cd_offset = values[7], values[8], values[9]
        directory = self._range(cd_offset, cd_offset + cd_size - 1)
        entries = []
        pos = 0
        for _ in range(total):
            fields = struct.unpack("<IHHHHHHIIIHHHHHII", directory[pos : pos + 46])
            method, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
            name_len, extra_len, comment_len, offset = fields[10], fields[11], fields[12], fields[16]
            name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
            extra = directory[pos + 46 + name_len : pos + 46 + name_len + extra_len]
            if 0xFFFFFFFF in (usize, csize, offset):
                cursor = 0
                while cursor + 4 <= len(extra):
                    tag, length = struct.unpack("<HH", extra[cursor : cursor + 4])
                    if tag == 0x0001:
                        values = list(
                            struct.unpack(f"<{length // 8}Q", extra[cursor + 4 : cursor + 4 + (length // 8) * 8])
                        )
                        if usize == 0xFFFFFFFF:
                            usize = values.pop(0)
                        if csize == 0xFFFFFFFF:
                            csize = values.pop(0)
                        if offset == 0xFFFFFFFF:
                            offset = values.pop(0)
                        break
                    cursor += 4 + length
            if not name.endswith("/"):
                entries.append(ZipEntry(name, offset, csize, usize, method, crc))
            pos += 46 + name_len + extra_len + comment_len
        return entries

    def read(self, entry: ZipEntry) -> bytes:
        import struct
        import zlib

        guess = 30 + len(entry.name.encode("utf-8")) + 256
        chunk = self._range(entry.header_offset, entry.header_offset + guess + entry.compress_size - 1)
        if chunk[:4] != b"PK\x03\x04":
            raise PrepareError(f"{entry.name}: bad local file header")
        name_len, extra_len = struct.unpack("<HH", chunk[26:30])
        start = 30 + name_len + extra_len
        if len(chunk) < start + entry.compress_size:
            chunk += self._range(
                entry.header_offset + len(chunk), entry.header_offset + start + entry.compress_size - 1
            )
        data = chunk[start : start + entry.compress_size]
        if entry.compress_type == 8:
            data = zlib.decompressobj(-15).decompress(data)
        elif entry.compress_type != 0:
            raise PrepareError(f"{entry.name}: unsupported compression method {entry.compress_type}")
        if zlib.crc32(data) & 0xFFFFFFFF != entry.crc:
            raise PrepareError(f"{entry.name}: CRC mismatch")
        return data


def extract_remote_zip_members(
    ctx: PrepareContext,
    repo_id: str,
    filename: str,
    wanted: dict[str, Path],
    *,
    desc: str,
) -> int:
    """Extract the members whose basename is a key of ``wanted`` to the mapped paths.

    Existing (decodable) files are skipped, so interrupted runs resume. Falls back to
    downloading the whole archive when the server does not honour range requests.
    Returns the number of files written.
    """
    pending = {name: path for name, path in wanted.items() if not is_valid_image_file(path)}
    ctx.log(
        f"  [zip] {desc}: {len(wanted) - len(pending)} present, {len(pending)} to extract from {repo_id}/{filename}"
    )
    if not pending:
        return 0
    try:
        archive = RemoteZip(repo_id, filename, retries=ctx.retries)
        by_name = {PurePosixPath(entry.name).name: entry for entry in archive.entries()}
    except RangeNotSupported as exc:
        ctx.log(f"  [zip] {exc}; downloading the whole archive instead")
        return _extract_from_full_zip(ctx, repo_id, filename, pending)
    missing = sorted(name for name in pending if name not in by_name)
    if missing:
        raise PrepareError(f"{len(missing)} file(s) not found in {repo_id}/{filename}, e.g. {missing[:3]}")

    def fetch(name: str) -> None:
        payload = archive.read(by_name[name])
        destination = pending[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + f".tmp{threading.get_ident()}")
        tmp.write_bytes(payload)
        os.replace(tmp, destination)

    done = 0
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, ctx.http_workers)) as pool:
        futures = {pool.submit(fetch, name): name for name in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures[futures[future]] = f"{type(exc).__name__}: {exc}"
            done += 1
            if done % 500 == 0 or done == len(pending):
                ctx.log(f"  [zip] {desc}: {done}/{len(pending)} ({len(failures)} failed)")
    if failures:
        examples = "; ".join(f"{name}: {error}" for name, error in list(failures.items())[:3])
        raise PrepareError(f"{desc}: {len(failures)} member(s) could not be extracted ({examples}); re-run to resume")
    return len(pending)


def _extract_from_full_zip(ctx: PrepareContext, repo_id: str, filename: str, pending: dict[str, Path]) -> int:
    archive_path = hf_download(ctx, repo_id, filename)
    written = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = PurePosixPath(_safe_member(info.filename)).name
            destination = pending.get(name)
            if destination is None or info.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    missing = [name for name, path in pending.items() if not path.exists()]
    if missing:
        raise PrepareError(f"{len(missing)} file(s) not found in {repo_id}/{filename}, e.g. {missing[:3]}")
    ctx.discard_raw(archive_path)
    return written


# ---------------------------------------------------------------------------
# Archives and file placement
# ---------------------------------------------------------------------------


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise PrepareError(f"refusing to extract unsafe archive member: {name!r}")
    return path


def extract_zip(zip_path: Path, dest: Path, log: Callable[[str], None] = print) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            member = _safe_member(info.filename)
            if info.is_dir():
                continue
            target = dest.joinpath(*member.parts)
            if target.exists() and target.stat().st_size == info.file_size:
                count += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    log(f"  [unzip] {zip_path.name}: {count} files -> {dest}")
    return count


def extract_tar(tar_path: Path, dest: Path, log: Callable[[str], None] = print) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(tar_path) as archive:
        members = []
        for member in archive.getmembers():
            _safe_member(member.name)
            if member.isfile():
                members.append(member)
        if hasattr(tarfile, "data_filter"):
            archive.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            archive.extractall(dest, members=members)
        count = len(members)
    log(f"  [untar] {tar_path.name}: {count} files -> {dest}")
    return count


def place_referenced_files(root: Path, relative_paths: Iterable[str], *, search_root: Path | None = None) -> None:
    """Make sure every ``root/<relative path>`` exists, moving files found by basename if needed.

    Archives are not always laid out exactly like the annotation paths (extra top-level
    folder, flat layout, ...). Missing files are looked up by basename below ``search_root``
    (default ``root``) and moved into place; anything still missing raises.
    """
    expected = [root / rel for rel in dict.fromkeys(relative_paths)]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return
    search_root = search_root or root
    index: dict[str, list[Path]] = {}
    for path in search_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    still_missing = []
    for path in missing:
        candidates = [candidate for candidate in index.get(path.name, []) if candidate != path]
        if len(candidates) == 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(path))
        else:
            still_missing.append(path)
    if still_missing:
        examples = ", ".join(str(path) for path in still_missing[:5])
        raise PrepareError(f"{len(still_missing)} referenced file(s) missing after extraction, e.g. {examples}")


def move_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


def copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def image_extension(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG"):
        return ".png"
    if payload[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    if payload[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".png"


def free_disk_gb(path: Path) -> float:
    """Free space of the filesystem holding ``path`` (or its closest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
