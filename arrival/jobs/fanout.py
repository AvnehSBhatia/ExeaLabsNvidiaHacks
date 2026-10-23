"""Fan sim shards out to a job backend with resumable storage (C1, offline slice).

`run_fanout` splits a sim run into n shards (the `arrival sim --shard i/n` machinery of L5), submits each shard that has
no done-marker to a `Backend`, polls, retries a failed shard, copies a finished shard's output into the backend's object
storage and writes that shard's done-marker LAST, so a marker means the shard's files are all there. A rerun after a
kill skips every shard with a marker. When all markers exist the shards are downloaded and joined with `merge_shards`.

`LocalBackend` runs shards as local subprocesses, with a directory standing in for object storage. `NebiusBackend`
(arrival/jobs/nebius.py) runs them as Nebius Serverless AI Jobs with an Object Storage bucket; it is tested only
against fakes and has never run against Nebius. `NebiusBackendStub` below is the earlier stub, which still raises."""

from __future__ import annotations

import json
import re

import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from arrival.jobs.shards import ShardError, merge_shards

RUNNING, SUCCEEDED, FAILED = "running", "succeeded", "failed"


@dataclass(frozen=True)
class FanoutConfig:
    """What every shard runs: the `arrival sim` settings (identical across shards) and the shard count."""
    run_id: str
    shards: int
    room: str = "hero_kitchen"
    families: str = "counter_to_sink,open_drawer"
    episodes: int = 8
    people: int = 0
    res: int = 128
    seed: int = 0
    scans: str = "scans"
    dart: float = 0.0
    base: str = "fixed"
    exocentric: bool = False

    def __post_init__(self) -> None:
        # The run id becomes both a scratch-directory component and an object-storage prefix.
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.run_id):
            raise ShardError("run_id must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit")
        for name, minimum in (("shards", 1), ("episodes", 1), ("people", 0), ("res", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ShardError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int:
            raise ShardError("seed must be an integer")
        if not isinstance(self.room, str) or not self.room.strip():
            raise ShardError("room must be a non-empty name")
        if (not isinstance(self.families, str) or not self.families.strip() or
                any(not part.strip() for part in self.families.split(","))):
            raise ShardError("families must be comma-separated non-empty names")
        if not isinstance(self.scans, str) or not self.scans.strip():
            raise ShardError("scans must be a non-empty path")
        if type(self.dart) not in (int, float) or not 0 <= self.dart < float("inf"):
            raise ShardError("dart must be a finite number >= 0")
        if self.base not in ("fixed", "mobile"):
            raise ShardError("base must be fixed or mobile")
        if type(self.exocentric) is not bool:
            raise ShardError("exocentric must be true or false")

    def sim_args(self, i: int) -> list[str]:
        args = ["sim", "--room", self.room, "--families", self.families, "--episodes", str(self.episodes),
                "--people", str(self.people), "--res", str(self.res), "--seed", str(self.seed),
                "--scans", str(self.scans), "--shard", f"{i}/{self.shards}", "--resume"]
        if self.dart:
            args += ["--dart", str(self.dart)]
        if self.base != "fixed":
            args += ["--base", self.base]
        if self.exocentric:
            args.append("--exocentric")
        return args


class Backend(Protocol):
    def submit(self, cfg: FanoutConfig, i: int) -> str:
        """Start shard i; return a job id."""

    def poll(self, job_id: str) -> str:
        """RUNNING, SUCCEEDED or FAILED."""

    def fetch_result(self, job_id: str, prefix: str) -> None:
        """Put a succeeded job's shard output (the `shards/<i>of<n>` tree) in object storage under `prefix/`."""

    def put_object(self, key: str, data: bytes) -> None: ...

    def get_object(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list_objects(self, prefix: str) -> list[str]: ...


class LocalBackend:
    """Shards as local subprocesses (`python -m arrival sim ...`); `store` is a directory standing in for object
    storage. `fail_first={i}` makes shard i's first submission exit non-zero without running (test hook)."""

    def __init__(self, store: str | Path, scratch: str | Path, fail_first=(), command: Callable | None = None):
        self.store, self.scratch = Path(store), Path(scratch)
        self.store.mkdir(parents=True, exist_ok=True)
        self.fail_first = set(fail_first)
        self.command = command  # (cfg, i, out_dir) -> argv; default runs arrival sim
        self.submissions: list[tuple[int, int]] = []  # (shard, attempt number) in order
        self._jobs: dict[str, tuple[subprocess.Popen, int, Path, object]] = {}

    def submit(self, cfg: FanoutConfig, i: int) -> str:
        n = sum(1 for s, _ in self.submissions if s == i) + 1
        self.submissions.append((i, n))
        out = self.scratch / cfg.run_id / f"{i}of{cfg.shards}"  # kept between attempts, so --resume has a journal
        out.mkdir(parents=True, exist_ok=True)
        if i in self.fail_first and n == 1:
            argv = [sys.executable, "-c", "import sys; sys.exit(3)"]
        elif self.command:
            argv = self.command(cfg, i, out)
        else:
            argv = [sys.executable, "-m", "arrival", *cfg.sim_args(i), "--out", str(out)]
        root = Path(__file__).resolve().parents[2]
        log = open(out / f"attempt{n}.log", "wb")
        proc = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        job_id = f"{cfg.run_id}-{i}-{n}"
        self._jobs[job_id] = (proc, i, out, log)
        return job_id

    def poll(self, job_id: str) -> str:
        proc, _, _, log = self._jobs[job_id]
        code = proc.poll()
        if code is None:
            return RUNNING
        log.close()
        return SUCCEEDED if code == 0 else FAILED

    def fetch_result(self, job_id: str, prefix: str) -> None:
        _, i, out, _ = self._jobs[job_id]
        expected = Path(prefix).name
        src = out / "shards" / expected
        if expected.split("of", 1)[0] != str(i) or not src.is_dir():
            raise ShardError(f"job {job_id} reported success but wrote no expected shard folder {src}")
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            self.put_object(f"{prefix}/{f.relative_to(src).as_posix()}", f.read_bytes())

    def _path(self, key: str) -> Path:
        p = (self.store / key).resolve()
        if self.store.resolve() not in p.parents:
            raise ShardError(f"object key {key!r} leaves the store")
        return p

    def put_object(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)  # a kill never leaves a half-written object under its real key

    def get_object(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_objects(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        return sorted(p.relative_to(self.store).as_posix() for p in base.rglob("*")
                      if p.is_file() and not p.name.endswith(".part"))

    def describe(self, job_id: str) -> dict:
        """What the status record keeps for a job. A local subprocess has no resource preset."""
        proc, _, out, _ = self._jobs[job_id]
        code = proc.poll()
        log = out / f"attempt{job_id.rsplit('-', 1)[-1]}.log"
        return {"state": "running" if code is None else f"exited {code}",
                "reason": f"exit code {code}, log {log}" if code else None, "resources": None}


class NebiusBackendStub:
    """UNTESTED interface stub for Nebius Serverless Jobs plus Nebius Object Storage. Not implemented, never run
    against Nebius (no account was available). A real backend would map `submit` to creating a job from the arrival
    container with the `sim` args, `poll` to its status, and the object calls to the S3-compatible API."""

    def __getattr__(self, name):
        raise NotImplementedError(f"NebiusBackendStub.{name}: no Nebius backend is implemented or tested")


# ---- orchestration ----

def marker_key(cfg: FanoutConfig, i: int) -> str:
    return f"{cfg.run_id}/done/{i}of{cfg.shards}.json"


def shard_prefix(cfg: FanoutConfig, i: int) -> str:
    return f"{cfg.run_id}/shards/{i}of{cfg.shards}"


def require_shard_artifacts(backend: Backend, cfg: FanoutConfig, i: int) -> None:
    base = f"{shard_prefix(cfg, i)}/datasets/{cfg.room}_sim/meta"
    for name in ("info.json", "episodes.jsonl"):
        if not backend.exists(f"{base}/{name}"):
            raise ShardError(f"shard {i}/{cfg.shards} is missing {base}/{name}; done marker is not trustworthy")


def run_fanout(backend: Backend, cfg: FanoutConfig, out_dir: str | Path, *, parallel: int = 2, max_attempts: int = 3,
               poll_seconds: float = 0.2, on_shard_done: Callable[[int], None] | None = None, log=print) -> dict:
    """Run every shard without a done-marker, retry failures up to `max_attempts` submissions each, then download and
    merge. Returns merge_shards' {"dataset", "report", "info"} plus "ran" and "skipped" shard lists. Raises ShardError
    when a shard fails every attempt (finished shards keep their markers, so a rerun resumes)."""
    if cfg.shards < 1:
        raise ShardError("shards must be at least 1")
    if type(parallel) is not int or parallel < 1:
        raise ShardError("parallel must be a positive integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ShardError("max_attempts must be a positive integer")
    if type(poll_seconds) not in (int, float) or not math.isfinite(poll_seconds) or poll_seconds < 0:
        raise ShardError("poll_seconds must be a finite, nonnegative number")
    manifest = {"config": cfg.__dict__}
    mkey = f"{cfg.run_id}/run.json"
    if backend.exists(mkey):
        try:
            saved_manifest = json.loads(backend.get_object(mkey))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ShardError(f"invalid run manifest {mkey}: {exc}") from exc
        if saved_manifest != manifest:
            raise ShardError(f"run {cfg.run_id!r} in storage was started with different settings; use a new run id")
    else:
        backend.put_object(mkey, json.dumps(manifest, sort_keys=True).encode())
    todo = []
    for i in range(cfg.shards):
        key = marker_key(cfg, i)
        if not backend.exists(key):
            todo.append(i)
            continue
        try:
            marker = json.loads(backend.get_object(key))
        except (ValueError, UnicodeDecodeError) as e:
            raise ShardError(f"invalid done marker {key}: {e}") from e
        if (not isinstance(marker, dict) or type(marker.get("shard")) is not int or marker["shard"] != i or
                type(marker.get("attempts")) is not int or marker["attempts"] < 1):
            raise ShardError(f"invalid done marker {key}: shard or attempts do not match")
        require_shard_artifacts(backend, cfg, i)
    skipped = [i for i in range(cfg.shards) if i not in todo]
    if skipped:
        log(f"fanout: shards {skipped} already done, skipping")
    attempts = {i: 0 for i in todo}
    pending, running, ran = list(todo), {}, []
    while pending or running:
        while pending and len(running) < parallel:
            i = pending.pop(0)
            attempts[i] += 1
            running[backend.submit(cfg, i)] = i
        time.sleep(poll_seconds)
        for job_id, i in list(running.items()):
            state = backend.poll(job_id)
            if state == RUNNING:
                continue
            if state not in (SUCCEEDED, FAILED):
                raise ShardError(f"job {job_id} returned unknown state {state!r}")
            del running[job_id]
            if state == SUCCEEDED:
                backend.fetch_result(job_id, shard_prefix(cfg, i))
                require_shard_artifacts(backend, cfg, i)
                backend.put_object(marker_key(cfg, i), json.dumps({"shard": i, "attempts": attempts[i]}).encode())
                ran.append(i)
                log(f"fanout: shard {i}/{cfg.shards} done (attempt {attempts[i]})")
                if on_shard_done:
                    on_shard_done(i)
            elif attempts[i] < max_attempts:
                log(f"fanout: shard {i}/{cfg.shards} failed (attempt {attempts[i]}), retrying")
                pending.append(i)
            else:
                raise ShardError(f"shard {i}/{cfg.shards} failed {attempts[i]} times; finished shards are kept, "
                                 f"rerun to resume")
    collected = Path(out_dir) / "collected"
    if collected.exists():
        shutil.rmtree(collected)
    for i in range(cfg.shards):
        prefix = shard_prefix(cfg, i)
        for key in backend.list_objects(prefix):
            dest = collected / "shards" / f"{i}of{cfg.shards}" / key[len(prefix) + 1:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(backend.get_object(key))
    res = merge_shards([collected / "shards"], out_dir, log=log)
    return {**res, "ran": sorted(ran), "skipped": skipped}
