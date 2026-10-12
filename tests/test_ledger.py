"""
Tests for ber.eval.ledger.RunLedger
=====================================
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.eval.ledger import RunLedger, _config_hash


class TestRunLedger:

    def test_start_run_returns_uuid(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        run_id = ledger.start_run(config={"k": 10}, dataset="test_ds", seed=42)
        assert len(run_id) == 36  # UUID4 length with dashes
        assert run_id.count("-") == 4

    def test_ledger_file_created(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        ledger.start_run(config={}, dataset="test_ds")
        assert (tmp_path / "runs.parquet").exists()

    def test_list_runs_empty(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        assert ledger.list_runs() == []

    def test_list_runs_after_start(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        run_id = ledger.start_run(config={"x": 1}, dataset="ds1", seed=7)
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["dataset"] == "ds1"
        assert runs[0]["status"] == "running"

    def test_log_metrics_updates_status(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        run_id = ledger.start_run(config={}, dataset="ds1")
        ledger.log_metrics(run_id, {"recall@10": 0.85})
        runs = ledger.list_runs()
        assert len(runs) == 1  # DISTINCT ON should return latest row
        run = runs[0]
        assert run["status"] == "completed"
        metrics = json.loads(run["metrics"])
        assert metrics["recall@10"] == pytest.approx(0.85)

    def test_log_metrics_unknown_run_raises(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        ledger.start_run(config={}, dataset="ds1")
        with pytest.raises(KeyError):
            ledger.log_metrics("nonexistent-uuid", {"x": 1})

    def test_config_hash_deterministic(self):
        h1 = _config_hash({"k": 10, "alpha": 0.5, "name": "test"})
        h2 = _config_hash({"name": "test", "alpha": 0.5, "k": 10})
        assert h1 == h2

    def test_config_hash_changes_with_content(self):
        h1 = _config_hash({"k": 10})
        h2 = _config_hash({"k": 20})
        assert h1 != h2

    def test_multiple_runs_stored(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        ids = [
            ledger.start_run(config={"i": i}, dataset=f"ds{i}", seed=i)
            for i in range(5)
        ]
        runs = ledger.list_runs()
        assert len(runs) == 5
        recorded_ids = {r["run_id"] for r in runs}
        for rid in ids:
            assert rid in recorded_ids

    def test_get_run(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        run_id = ledger.start_run(
            config={"foo": "bar"}, dataset="ds_x", seed=1,
            dataset_version="v1.0"
        )
        rec = ledger.get_run(run_id)
        assert rec is not None
        assert rec["dataset"] == "ds_x"
        assert rec["dataset_version"] == "v1.0"
        assert rec["seed"] == 1

    def test_failed_status(self, tmp_path):
        ledger = RunLedger(tmp_path / "runs.parquet")
        run_id = ledger.start_run(config={}, dataset="ds1")
        ledger.log_metrics(run_id, {"error": "oops"}, status="failed")
        runs = ledger.list_runs()
        assert runs[0]["status"] == "failed"

    def test_dirty_tree_logging_and_refusal(self, tmp_path, monkeypatch):
        ledger = RunLedger(tmp_path / "runs.parquet")

        # Mock clean tree
        monkeypatch.setattr("ber.eval.ledger.subprocess.run", lambda *args, **kwargs: (
            type("Result", (), {"returncode": 0, "stdout": "abc1234\n" if "rev-parse" in args[0] else ""})()
        ))
        run_id_clean = ledger.start_run(config={}, dataset="ds1", require_clean=True)
        rec_clean = ledger.get_run(run_id_clean)
        assert rec_clean["git_commit"] == "abc1234"

        # Mock dirty tree
        monkeypatch.setattr("ber.eval.ledger.subprocess.run", lambda *args, **kwargs: (
            type("Result", (), {"returncode": 0, "stdout": "abc1234\n" if "rev-parse" in args[0] else " M file.py\n"})()
        ))
        # require_clean=False logs dirty:sha
        run_id_dirty = ledger.start_run(config={}, dataset="ds1", require_clean=False)
        rec_dirty = ledger.get_run(run_id_dirty)
        assert rec_dirty["git_commit"] == "dirty:abc1234"

        # require_clean=True raises RuntimeError
        with pytest.raises(RuntimeError, match="Cannot log run from dirty working tree"):
            ledger.start_run(config={}, dataset="ds1", require_clean=True)

