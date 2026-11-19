import gzip
import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from bson import encode

EVALUATION = Path(__file__).resolve().parent
ROOT = EVALUATION.parents[1]


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    evaluation = repo / "evaluations/search_providers"
    evaluation.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", "README.md", ".python-version"):
        shutil.copy(ROOT / name, repo / name)
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    shutil.copy(EVALUATION / "prepare_dataset.py", evaluation)
    shutil.copy(EVALUATION / "run.sh", evaluation)

    with closing(sqlite3.connect(tmp_path / "logs.db")) as db, db:
        db.execute(
            "CREATE TABLE logs (id, timestamp, provider, model,"
            " input_history, output_message)"
        )
        db.execute(
            "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
            (
                "request",
                "2026-11-13",
                "test",
                "model",
                "[]",
                json.dumps(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call", "function": {"name": "web_search"}}
                        ],
                    }
                ),
            ),
        )
    collections: dict[str, dict[str, Any]] = {
        "messages": {
            "conversationId": "conversation",
            "messageId": "message",
            "isCreatedByUser": False,
            "content": [
                {"tool_call": {"id": "call", "name": "web_search"}},
                {"type": "text", "text": "Original answer"},
            ],
        },
        "conversations": {
            "conversationId": "conversation",
            "lastResponseMessageId": "message",
        },
    }
    header = b"\x6d\xe2\x99\x81" + encode({}) + b"\xff" * 4
    blocks = {}
    for name, document in collections.items():
        blocks[name] = (
            encode({"db": "librechat", "collection": name})
            + encode(document)
            + b"\xff" * 4
        )
        (tmp_path / f"{name}.gz").write_bytes(
            gzip.compress(header + blocks[name])
        )
    (tmp_path / "combined.gz").write_bytes(
        gzip.compress(header + b"".join(blocks.values()))
    )

    ssh = tmp_path / "ssh"
    ssh.write_text(
        """#!/bin/bash
set -eu
echo "$*" >> "$FIXTURES/ssh.log"
case "$*" in
    *' messages')
        if [[ ${FAIL_DOWNLOAD:-} ]]; then exit 1; fi
        cat "$FIXTURES/messages.gz"
        ;;
    *' conversations') cat "$FIXTURES/conversations.gz" ;;
    *) cd "$FIXTURES"; tar -cf - logs.db* ;;
esac
"""
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("FIXTURES", str(tmp_path))
    for key, value in {
        "EVAL_SSH_HOST": "eval-host",
        "EVAL_SSH_USER": "eval-operator",
        "BIFROST_CONTAINER": "bifrost",
        "BIFROST_DB": "/app/data/logs.db",
        "LIBRECHAT_MONGO_CONTAINER": "mongodb",
        "LIBRECHAT_DB": "librechat",
    }.items():
        monkeypatch.setenv(key, value)
    return evaluation


def run(evaluation: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(evaluation / "run.sh"), *args],
        cwd=evaluation.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )


def test_fresh_run_prepares_scenarios(
    deployment: Path, tmp_path: Path
) -> None:
    with closing(sqlite3.connect(tmp_path / "logs.db")) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("UPDATE logs SET model = 'committed-in-wal'")
        live.commit()
        completed = run(deployment)

    assert completed.returncode == 0, completed.stderr
    (dataset,) = (deployment / "data").glob("*/scratch/*.jsonl")
    assert f"Dataset prepared: {dataset} (1 scenarios)" in completed.stdout
    record = json.loads(dataset.read_text())
    assert record["assistantMessageId"] == "message"
    assert record["model"] == "committed-in-wal"
    connections = (tmp_path / "ssh.log").read_text().splitlines()
    assert len(connections) == 3
    assert all("-l eval-operator -- eval-host " in line for line in connections)


def test_reuse_only_shares_snapshots(
    deployment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = run(deployment)
    assert fresh.returncode == 0, fresh.stderr
    (original,) = (deployment / "data").glob("*/scratch/*.jsonl")
    monkeypatch.delenv("EVAL_SSH_HOST")
    monkeypatch.delenv("EVAL_SSH_USER")

    reused = run(deployment, "--reuse-inputs")

    assert reused.returncode == 0, reused.stderr
    datasets = set((deployment / "data").glob("*/scratch/*.jsonl"))
    (new,) = datasets - {original}
    assert new.read_bytes() == original.read_bytes()
    assert (new.parents[1] / "snapshots").resolve() == (
        original.parents[1] / "snapshots"
    )
    assert len((tmp_path / "ssh.log").read_text().splitlines()) == 3


def test_reuse_requires_prior_snapshots(
    deployment: Path, tmp_path: Path
) -> None:
    completed = run(deployment, "--reuse-inputs")

    assert completed.returncode != 0
    assert "No reusable snapshots" in completed.stderr
    assert not (tmp_path / "ssh.log").exists()


@pytest.mark.parametrize("failed_stage", ["download", "preparation"])
def test_failed_refresh_preserves_previous_snapshots(
    deployment: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    fresh = run(deployment)
    assert fresh.returncode == 0, fresh.stderr
    (original,) = (deployment / "data").glob("*/scratch/*.jsonl")
    if failed_stage == "download":
        monkeypatch.setenv("FAIL_DOWNLOAD", "1")
    else:
        (tmp_path / "messages.gz").write_bytes(b"not an archive")

    failed = run(deployment)

    assert failed.returncode != 0
    assert "Dataset prepared:" not in failed.stdout
    reused = run(deployment, "--reuse-inputs")
    assert reused.returncode == 0, reused.stderr
    datasets = set((deployment / "data").glob("*/scratch/*.jsonl"))
    (new,) = datasets - {original}
    assert new.read_bytes() == original.read_bytes()


def test_combined_archive_produces_the_same_scenarios(
    deployment: Path, tmp_path: Path
) -> None:
    fresh = run(deployment)
    assert fresh.returncode == 0, fresh.stderr
    (scoped,) = (deployment / "data").glob("*/scratch/*.jsonl")
    combined = tmp_path / "combined.jsonl"

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(deployment.parents[1]),
            "--group",
            "search-providers",
            "python",
            str(deployment / "prepare_dataset.py"),
            "--db",
            str(tmp_path / "logs.db"),
            "--librechat",
            str(tmp_path / "combined.gz"),
            "--out",
            str(combined),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert combined.read_bytes() == scoped.read_bytes()
