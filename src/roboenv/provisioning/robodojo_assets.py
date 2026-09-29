"""Pinned public RoboDojo assets, independent of policy packages and Isaac imports."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from roboenv.provisioning.assets import asset_cache
from roboenv.provisioning.progress import stage

REPOSITORY = "RoboDojo-Benchmark/RoboDojo"
REVISION = "0a709fe7863f6c869a8d141676585a883fce4270"


class ResolverPacer:
    """Share request spacing and server-directed cooldown across download threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.cancelled = threading.Event()

    def wait(self):
        with self.lock:
            delay = max(0.0, self.next_request - time.monotonic())
            self.next_request = time.monotonic() + delay + 0.15
        if self.cancelled.wait(delay):
            raise concurrent.futures.CancelledError("Asset acquisition cancelled after failure")

    def limited(self, headers):
        retry = headers.get("retry-after", "")
        reset = re.search(r"(?:^|;)t=(\d+)", headers.get("ratelimit", ""))
        delay = float(retry) if retry.isdigit() else float(reset[1]) + 1 if reset else 60.0
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + min(delay, 600.0))


def verified(path, entry):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        return False
    lfs = entry.get("lfs")
    checksum = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        checksum.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest() == (lfs["sha256"] if lfs else entry["blobId"])


def safe_relative(name):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "\\" in name
        or relative.parts[0] != "Assets"
    ):
        raise ValueError("Unsafe RoboDojo asset path")
    return Path(*relative.parts)


def download_file(root, entry, pacer=None):
    """Verify or acquire one pinned file inside the selected Assets directory."""
    path = root / safe_relative(entry["rfilename"]).relative_to("Assets")
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset destination follows a symlink outside its depot")
    if path.exists() or path.is_symlink():
        if not verified(path, entry):
            raise ValueError(f"Existing asset differs from its source pin: {entry['rfilename']}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset destination follows a symlink outside its depot")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{quote(entry['rfilename'])}"
    pacer = pacer or ResolverPacer()
    failures = 0
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        pacer.wait()
        try:
            with tempfile.TemporaryDirectory(
                prefix=".roboenv-download-", dir=path.parent
            ) as temporary:
                pending = Path(temporary) / "data"
                with urlopen(url, timeout=90) as response, pending.open("xb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if not verified(pending, entry):
                    raise ValueError(
                        "Downloaded RoboDojo asset failed source checksum verification"
                    )
                # Publish without overwriting any existing file, including a concurrent download.
                try:
                    os.link(pending, path)
                except FileExistsError:
                    if not verified(path, entry):
                        raise ValueError(
                            "Concurrent asset differs from the pinned source"
                        ) from None
                return
        except HTTPError as error:
            if error.code == 429:
                pacer.limited(error.headers)
                continue
            failures += 1
            if error.code < 500 or failures >= 3:
                raise
            time.sleep(2**failures)
        except (URLError, TimeoutError, ConnectionError):
            failures += 1
            if failures >= 3:
                raise
            time.sleep(2**failures)
    raise TimeoutError("RoboDojo asset could not be acquired within its download deadline")


def prepare_assets(project, *, progress=None, cache=None, workers=4):
    project = Path(project).resolve()
    link = project / "third_party/RoboDojo/Assets"
    if (link.exists() or link.is_symlink()) and not link.is_dir():
        raise ValueError("Preserve existing RoboDojo Assets; expected a readable directory")
    if cache is None:
        cache = asset_cache(project, "robodojo")
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source_path = cache / "source.json"
    if not source_path.exists():
        url = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
        with urlopen(url, timeout=90) as response:
            source = json.load(response)
        with source_path.open("x") as output:
            json.dump(source, output)
    source = json.loads(source_path.read_text())
    if source["sha"] != REVISION or source.get("gated") or source.get("private"):
        raise ValueError("Expected the pinned public, ungated RoboDojo dataset")
    entries = [entry for entry in source["siblings"] if entry["rfilename"].startswith("Assets/")]
    if not entries or len({item["rfilename"] for item in entries}) != len(entries):
        raise ValueError("Empty or duplicated RoboDojo asset inventory")
    for entry in entries:
        safe_relative(entry["rfilename"])
    # Existing user directories and links are authoritative locations, while
    # every file still has to match the selected public source pin.
    assets = link.resolve() if link.is_dir() else cache / REVISION / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    missing = sum(
        entry["size"]
        for entry in entries
        if not (assets / safe_relative(entry["rfilename"]).relative_to("Assets")).exists()
    )
    if shutil.disk_usage(assets).free < missing + 2 * 1024**3:
        raise ValueError("Insufficient asset space; set ROBOENV_ASSET_CACHE to a larger data depot")
    with stage(progress, f"Verifying/downloading {len(entries)} RoboDojo source assets"):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        pacer = ResolverPacer()
        try:
            jobs = {
                executor.submit(download_file, assets, entry, pacer): entry for entry in entries
            }
            for count, job in enumerate(concurrent.futures.as_completed(jobs), 1):
                job.result()
                if progress and (count % 250 == 0 or count == len(entries)):
                    progress(f"RoboDojo assets verified: {count}/{len(entries)}")
        finally:
            pacer.cancelled.set()
            executor.shutdown(wait=True, cancel_futures=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != assets:
            raise ValueError(
                "Preserve existing RoboDojo Assets; select or verify that depot explicitly"
            )
    else:
        link.symlink_to(assets, target_is_directory=True)
    return {
        "environment_assets_verified": True,
        "public_assets_verified": True,
        "source_revision": REVISION,
        "files": len(entries),
        "asset_depot": str(assets),
        "asset_bytes": sum(entry["size"] for entry in entries),
        "gpu_replay_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    report = prepare_assets(
        args.project, cache=args.cache, progress=lambda message: print(message, flush=True)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
