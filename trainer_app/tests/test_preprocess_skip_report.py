"""Standalone tests for _find_all_windows skip-report logic.

Does NOT import numpy/torch/netCDF4 — replicates core algorithm with pure
stdlib to verify correctness.
"""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from bisect import bisect_left


def searchsorted(sorted_list, target):
    """Pure-Python equivalent of np.searchsorted."""
    return bisect_left(sorted_list, target)


# ── Algorithm mirror (matches DataPreprocessor._find_all_windows) ──────

class FakePreprocessor:
    def __init__(self, history_steps=12, future_steps=36, tolerance_s=150):
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.total_steps = history_steps + future_steps
        self.radar_tolerance_s = tolerance_s
        self._report = []

    def find_all_windows(self, rows, nc_files):
        """Mirror of _find_all_windows using pure Python."""
        n = len(rows)
        if n < self.total_steps:
            return [], f"only {n} CSV rows, need ≥{self.total_steps}"

        nc_ts_list = sorted(nc_files.keys())

        def find_nearest_nc(t):
            target_ts = int(t.timestamp())
            idx = searchsorted(nc_ts_list, target_ts)
            best = None
            best_diff = self.radar_tolerance_s
            for i in [idx - 1, idx, idx + 1]:
                if 0 <= i < len(nc_ts_list):
                    diff = abs(nc_ts_list[i] - target_ts)
                    if diff <= best_diff:
                        best_diff = diff
                        best = nc_files[nc_ts_list[i]]
            return best

        matched_indices = []
        for i, row in enumerate(rows):
            if find_nearest_nc(row["time"]) is not None:
                matched_indices.append(i)

        if len(matched_indices) < self.total_steps:
            return [], (
                f"only {len(matched_indices)}/{n} CSV rows have radar matches, "
                f"need ≥{self.total_steps}"
            )

        stride = self.history_steps
        windows = []

        for start in range(0, len(matched_indices) - self.total_steps + 1, stride):
            indices = matched_indices[start:start + self.total_steps]
            time_span = rows[indices[-1]]["time"] - rows[indices[0]]["time"]
            expected_span = timedelta(minutes=5 * (self.total_steps - 1))
            if time_span > expected_span + timedelta(minutes=5):
                continue

            matched_rows = [rows[i] for i in indices]
            matched_files = [find_nearest_nc(rows[i]["time"]) for i in indices]
            if all(f is not None for f in matched_files):
                windows.append((matched_rows, matched_files))

        if not windows:
            first_idx = matched_indices[0]
            last_idx = matched_indices[-1]
            actual_span = rows[last_idx]["time"] - rows[first_idx]["time"]
            return [], (
                f"no contiguous window of {self.total_steps} radar-matched rows "
                f"({len(matched_indices)} matched over {n} CSV rows, "
                f"span {actual_span}, need ≤{expected_span + timedelta(minutes=5)})"
            )

        return windows, None

    def write_report(self, storms, total_samples, output_dir):
        """Mirror of _write_report."""
        success_count = sum(1 for s in storms if s["status"] == "success")
        skipped_count = sum(1 for s in storms if s["status"] != "success")

        reason_counts = {}
        for s in storms:
            if s["status"] == "skipped":
                reason = s.get("skip_reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            elif s["status"] == "error":
                reason_counts["exception"] = reason_counts.get("exception", 0) + 1

        report = {
            "summary": {
                "total_storms": len(storms),
                "successful": success_count,
                "skipped": skipped_count,
                "total_samples": total_samples,
                "skip_reasons": reason_counts,
            },
            "config": {
                "history_steps": self.history_steps,
                "future_steps": self.future_steps,
                "total_steps": self.total_steps,
                "radar_time_tolerance_s": self.radar_tolerance_s,
            },
            "storms": storms,
        }

        path = Path(output_dir) / "preprocess_report.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return path


# ── helpers ─────────────────────────────────────────────────────────────

def make_rows(n, start_time=None):
    if start_time is None:
        start_time = datetime(2023, 5, 1, 12, 0, 0)
    return [
        {
            "time": start_time + timedelta(minutes=5 * i),
            "lon": -95.0 + i * 0.001,
            "lat": 35.0 + i * 0.001,
            "u": 10.0, "v": 5.0, "ef": 1,
        }
        for i in range(n)
    ]


def make_nc_files(rows, offset_s=0):
    return {
        int((r["time"] + timedelta(seconds=offset_s)).timestamp()): Path(f"/fake/radar_{r['time'].strftime('%Y%m%dT%H%M')}.nc")
        for r in rows
    }


# ── tests ───────────────────────────────────────────────────────────────

def test_too_few_csv_rows():
    pp = FakePreprocessor()
    rows = make_rows(20)
    nc = make_nc_files(rows)
    windows, reason = pp.find_all_windows(rows, nc)
    assert windows == [], windows
    assert "20" in reason, reason
    assert "48" in reason, reason
    print("  PASS test_too_few_csv_rows")


def test_no_radar_files():
    pp = FakePreprocessor()
    rows = make_rows(60)
    windows, reason = pp.find_all_windows(rows, {})
    assert windows == [], windows
    assert "0/" in reason, reason
    print("  PASS test_no_radar_files")


def test_insufficient_matches():
    pp = FakePreprocessor()
    rows = make_rows(60)
    # Only give radar files for 30 rows
    nc = make_nc_files(rows[:30])
    windows, reason = pp.find_all_windows(rows, nc)
    assert windows == [], windows
    assert "30/" in reason, reason
    print("  PASS test_insufficient_matches")


def test_minimum_success():
    pp = FakePreprocessor()
    rows = make_rows(48)
    nc = make_nc_files(rows)
    windows, reason = pp.find_all_windows(rows, nc)
    assert reason is None, f"Unexpected reason: {reason}"
    assert len(windows) == 1, f"Expected 1, got {len(windows)}"
    print("  PASS test_minimum_success")


def test_sliding_windows():
    pp = FakePreprocessor()
    rows = make_rows(60)
    nc = make_nc_files(rows)
    windows, reason = pp.find_all_windows(rows, nc)
    assert reason is None, f"Unexpected reason: {reason}"
    # stride=12, 60 matched rows, 60-48+1=13 possible starts, step by 12: 0, 12
    assert len(windows) == 2, f"Expected 2 windows, got {len(windows)}"
    print("  PASS test_sliding_windows")


def test_time_gap_rejected():
    pp = FakePreprocessor()
    # 24 rows, 3h gap, 24 rows
    rows = make_rows(24, datetime(2023, 5, 1, 12, 0, 0))
    rows += make_rows(24, datetime(2023, 5, 1, 15, 0, 0))
    nc = make_nc_files(rows)
    windows, reason = pp.find_all_windows(rows, nc)
    assert windows == [], f"Expected empty, got {len(windows)}"
    assert "span" in reason.lower() or "contiguous" in reason.lower(), reason
    print("  PASS test_time_gap_rejected")


def test_tolerance_boundary():
    """CSV time exactly 150s from radar → should match (<= tolerance)."""
    pp = FakePreprocessor(tolerance_s=150)
    rows = make_rows(48)
    # Offset each radar file by exactly tolerance (150s)
    nc = make_nc_files(rows, offset_s=150)
    windows, reason = pp.find_all_windows(rows, nc)
    assert reason is None, f"Should match at boundary, got: {reason}"
    assert len(windows) == 1
    print("  PASS test_tolerance_boundary")


def test_write_report():
    pp = FakePreprocessor()
    storms = [
        {"storm_id": "1_20230501", "csv_rows": 60, "status": "success",
         "windows_found": 2, "samples_saved": 2,
         "time_range": "2023-05-01T12:00Z ~ 2023-05-01T16:55Z"},
        {"storm_id": "2_20230501", "csv_rows": 20, "status": "skipped",
         "skip_reason": "only 20 CSV rows, need ≥48",
         "time_range": "2023-05-01T12:00Z ~ 2023-05-01T13:35Z"},
        {"storm_id": "3_20230501", "csv_rows": 80, "status": "skipped",
         "skip_reason": "only 5/80 CSV rows have radar matches, need ≥48",
         "time_range": "2023-05-01T12:00Z ~ 2023-05-01T18:35Z"},
        {"storm_id": "4_20230501", "csv_rows": 50, "status": "error",
         "skip_reason": "ValueError: something broke",
         "time_range": "2023-05-01T12:00Z ~ 2023-05-01T16:05Z"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pp.write_report(storms, total_samples=2, output_dir=tmpdir)
        assert path.exists(), f"Not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        s = data["summary"]
        assert s["total_storms"] == 4
        assert s["successful"] == 1
        assert s["skipped"] == 3
        assert s["total_samples"] == 2
        assert len(data["storms"]) == 4

        reasons = s["skip_reasons"]
        assert reasons["only 20 CSV rows, need ≥48"] == 1
        assert reasons["only 5/80 CSV rows have radar matches, need ≥48"] == 1
        assert reasons["exception"] == 1

    print("  PASS test_write_report")


# ── main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Skip-report tests (stdlib only):")
    test_too_few_csv_rows()
    test_no_radar_files()
    test_insufficient_matches()
    test_minimum_success()
    test_sliding_windows()
    test_time_gap_rejected()
    test_tolerance_boundary()
    test_write_report()
    print("\nAll 8 tests passed.")
