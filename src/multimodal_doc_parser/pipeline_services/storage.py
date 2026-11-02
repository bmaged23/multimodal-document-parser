"""Durable idempotent jobs and publish-on-completion shared artifact storage."""

import contextlib
import hashlib
import json
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class JobStore:
    def __init__(self, root: Path, stage: str) -> None:
        self.path = root / "jobs" / (stage + ".sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request TEXT "
                "NOT NULL, request_hash TEXT NOT NULL, idempotency_key TEXT UNIQUE, "
                "status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, "
                "heartbeat REAL, worker TEXT, result TEXT, error TEXT)"
            )

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(
        self, request: dict[str, Any], key: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        body = json.dumps(request, sort_keys=True)
        digest = hashlib.sha256(body.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                previous = db.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?", (key,)
                ).fetchone()
                if previous:
                    if previous["request_hash"] != digest:
                        raise ValueError("Idempotency key already belongs to a different request")
                    return self.decode(previous), False
            identity = uuid.uuid4().hex
            now = time.time()
            db.execute(
                (
                    "INSERT INTO "
                    "jobs(id,request,request_hash,idempotency_key,status,created,updated) "
                    "VALUES (?,?,?,?,?,?,?)"
                ),
                (identity, body, digest, key, "queued", now, now),
            )
        return self.get(identity), True

    @staticmethod
    def decode(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["request"] = json.loads(data["request"])
        data["result"] = json.loads(data["result"]) if data["result"] else None
        return data

    def get(self, identity: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise KeyError("Unknown job")
        return self.decode(row)

    def claim(self, worker: str) -> dict[str, Any] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                (
                    "UPDATE jobs SET status='interrupted', error='Worker lease expired; "
                    "inspect partial artifacts and explicitly resubmit. No automatic "
                    "inference retry.', updated=? WHERE status='running' AND heartbeat < ?"
                ),
                (time.time(), time.time() - 120),
            )
            row = db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='running',worker=?,updated=?,heartbeat=? WHERE id=?",
                (worker, time.time(), time.time(), row["id"]),
            )
        return self.get(row["id"])

    def heartbeat(self, identity: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET heartbeat=? WHERE id=? AND status='running'",
                (time.time(), identity),
            )

    def finish(
        self, identity: str, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        with self.connect() as db:
            db.execute(
                (
                    "UPDATE jobs SET status=?,result=?,error=?,updated=? WHERE id=? AND "
                    "status='running'"
                ),
                (
                    "failed" if error else "completed",
                    json.dumps(result) if result is not None else None,
                    error,
                    time.time(),
                    identity,
                ),
            )


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, identity: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", identity):
            raise ValueError("Invalid artifact ID")
        path = self.root / identity
        if path.is_symlink():
            raise ValueError("Artifact directory must not be a symlink")
        return path

    def reserve(self) -> tuple[str, Path]:
        identity = uuid.uuid4().hex
        path = self.directory(identity)
        path.mkdir()
        return identity, path

    def publish(self, identity: str, kind: str, bundle: Path, **metadata: Any) -> dict[str, Any]:
        path = self.directory(identity)
        relative = bundle.resolve().relative_to(path.resolve())
        record = dict(artifact_id=identity, kind=kind, bundle=relative.as_posix(), **metadata)
        temporary = path / ".artifact.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path / "artifact.json")
        return record

    def metadata(self, identity: str) -> dict[str, Any]:
        record: dict[str, Any] = json.loads(
            (self.directory(identity) / "artifact.json").read_text()
        )
        return record

    def resolve(self, identity: str, kind: str | None = None) -> Path:
        record = self.metadata(identity)
        if kind is not None and record["kind"] != kind:
            raise ValueError(f"Expected a {kind} artifact")
        root = self.directory(identity).resolve()
        path = (root / str(record["bundle"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact escaped its storage directory")
        return path
