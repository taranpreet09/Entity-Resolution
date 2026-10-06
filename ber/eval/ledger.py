"""
ber.eval.ledger — Append-Only Experiment Run Ledger
=====================================================
Every experiment run is recorded in an append-only Parquet table backed by
DuckDB.  No run record is ever modified or deleted.

Schema (one row per run):
    run_id          VARCHAR       UUID v4
    timestamp       TIMESTAMP     UTC wall-clock at start_run()
    config_hash     VARCHAR       SHA-256 of the serialised config dict
    dataset         VARCHAR       dataset identifier string, e.g. "fodors_zagats"
    dataset_version VARCHAR       git tag / file hash / "unknown"
    git_commit      VARCHAR       HEAD commit SHA or "untracked"
    seed            INTEGER       RNG seed used for the run (−1 = unset)
    metrics         VARCHAR       JSON-serialised metrics dict (populated later)
    status          VARCHAR       "running" | "completed" | "failed"

Usage::

    from ber.eval.ledger import RunLedger

    ledger = RunLedger("experiments/runs.parquet")
    run_id = ledger.start_run(config={...}, dataset="fodors_zagats", seed=42)
    ledger.log_metrics(run_id, {"recall@10": 0.87, "zero_cand_rate": 0.02})

CLI::

    python -m ber.eval.ledger list [--ledger PATH]
    python -m ber.eval.ledger show RUN_ID [--ledger PATH]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LEDGER_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("config_hash", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("dataset_version", pa.string()),
        pa.field("git_commit", pa.string()),
        pa.field("seed", pa.int32()),
        pa.field("metrics", pa.string()),   # JSON
        pa.field("status", pa.string()),
    ]
)

DEFAULT_LEDGER_PATH = Path("experiments") / "runs.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serialised config dict."""
    serialised = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def _git_commit() -> str:
    """Return the current HEAD commit SHA, or 'untracked' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "untracked"


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# RunLedger
# ---------------------------------------------------------------------------

class RunLedger:
    """Append-only experiment run ledger backed by a single Parquet file.

    Parameters
    ----------
    path:
        Path to the Parquet file.  Created on first write.
    """

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        config: dict[str, Any],
        dataset: str,
        seed: int = -1,
        dataset_version: str = "unknown",
    ) -> str:
        """Record the start of a new experiment run.

        Parameters
        ----------
        config:
            Arbitrary dict describing all hyperparameters and choices.
            Hashed and stored; the caller is responsible for reproducibility.
        dataset:
            Dataset identifier, e.g. ``"fodors_zagats"``.
        seed:
            Integer RNG seed used for the run.  Pass -1 if no seed was set.
        dataset_version:
            A string identifying the dataset snapshot (file hash, date, tag).

        Returns
        -------
        str
            A UUID4 run_id that can be passed to :meth:`log_metrics`.
        """
        run_id = str(uuid.uuid4())
        row = pa.table(
            {
                "run_id": [run_id],
                "timestamp": pa.array(
                    [_now_utc()], type=pa.timestamp("us", tz="UTC")
                ),
                "config_hash": [_config_hash(config)],
                "dataset": [dataset],
                "dataset_version": [dataset_version],
                "git_commit": [_git_commit()],
                "seed": pa.array([seed], type=pa.int32()),
                "metrics": ["{}"],
                "status": ["running"],
            },
            schema=LEDGER_SCHEMA,
        )
        self._append(row)
        return run_id

    def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, Any],
        status: str = "completed",
    ) -> None:
        """Append a metrics update for an existing run.

        Because the ledger is append-only, this adds a new row with the same
        run_id and updated metrics/status rather than modifying the original.
        Queries should use the most-recent row per run_id (MAX timestamp).

        Parameters
        ----------
        run_id:
            The run_id returned by :meth:`start_run`.
        metrics:
            Dict of metric name → value.  Values must be JSON-serialisable.
        status:
            ``"completed"`` or ``"failed"``.
        """
        # Read the original row to carry forward non-metrics fields
        original = self._get_run(run_id)
        if original is None:
            raise KeyError(f"run_id '{run_id}' not found in ledger at {self.path}")

        row = pa.table(
            {
                "run_id": [run_id],
                "timestamp": pa.array(
                    [_now_utc()], type=pa.timestamp("us", tz="UTC")
                ),
                "config_hash": [original["config_hash"]],
                "dataset": [original["dataset"]],
                "dataset_version": [original["dataset_version"]],
                "git_commit": [original["git_commit"]],
                "seed": pa.array([original["seed"]], type=pa.int32()),
                "metrics": [json.dumps(metrics, default=str)],
                "status": [status],
            },
            schema=LEDGER_SCHEMA,
        )
        self._append(row)

    def list_runs(self) -> list[dict[str, Any]]:
        """Return the latest state of every run as a list of dicts."""
        if not self.path.exists():
            return []
        con = duckdb.connect()
        rows = con.execute(
            f"""
            SELECT DISTINCT ON (run_id)
                run_id, timestamp, config_hash, dataset,
                dataset_version, git_commit, seed, metrics, status
            FROM read_parquet('{self.path.as_posix()}')
            ORDER BY run_id, timestamp DESC
            """
        ).fetchall()
        cols = [
            "run_id", "timestamp", "config_hash", "dataset",
            "dataset_version", "git_commit", "seed", "metrics", "status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the latest state of a single run, or None if not found."""
        return self._get_run(run_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append(self, table: pa.Table) -> None:
        """Append *table* to the Parquet file, creating it if absent."""
        if self.path.exists():
            existing = pq.read_table(self.path, schema=LEDGER_SCHEMA)
            combined = pa.concat_tables([existing, table])
        else:
            combined = table
        pq.write_table(combined, self.path, compression="snappy")

    def _get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        con = duckdb.connect()
        rows = con.execute(
            f"""
            SELECT config_hash, dataset, dataset_version, git_commit, seed
            FROM read_parquet('{self.path.as_posix()}')
            WHERE run_id = '{run_id}'
            ORDER BY timestamp ASC
            LIMIT 1
            """
        ).fetchall()
        if not rows:
            return None
        cols = ["config_hash", "dataset", "dataset_version", "git_commit", "seed"]
        return dict(zip(cols, rows[0]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_list(ledger_path: Path) -> None:
    ledger = RunLedger(ledger_path)
    runs = ledger.list_runs()
    if not runs:
        print("No runs recorded.")
        return
    header = f"{'run_id':<38} {'status':<12} {'dataset':<20} {'timestamp'}"
    print(header)
    print("-" * len(header))
    for r in runs:
        ts = str(r["timestamp"])[:19]
        print(f"{r['run_id']:<38} {r['status']:<12} {r['dataset']:<20} {ts}")


def _cli_show(run_id: str, ledger_path: Path) -> None:
    ledger = RunLedger(ledger_path)
    runs = ledger.list_runs()
    matches = [r for r in runs if r["run_id"].startswith(run_id)]
    if not matches:
        print(f"No run found matching '{run_id}'")
        sys.exit(1)
    r = matches[0]
    for k, v in r.items():
        if k == "metrics":
            try:
                v = json.dumps(json.loads(v), indent=2)
            except Exception:
                pass
        print(f"{k:20s}: {v}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ber.eval.ledger",
        description="Inspect the BER experiment run ledger.",
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help="Path to the runs.parquet ledger file.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all recorded runs.")

    show_p = sub.add_parser("show", help="Show details for a specific run.")
    show_p.add_argument("run_id", help="run_id (or unique prefix) to display.")

    args = parser.parse_args(argv)
    ledger_path = Path(args.ledger)

    if args.cmd == "list" or args.cmd is None:
        _cli_list(ledger_path)
    elif args.cmd == "show":
        _cli_show(args.run_id, ledger_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
