"""Data preprocessing for NEXRAD Level II 3D radar and storm track CSV.

Dataset 1: NEXRAD 3D .nc files (5-min timesteps, sparse index on 480x480x29 grid)
Dataset 2: Storm track .csv files (Storm Number, Time, Lon, Lat, u/v-motion, EF)
"""
import os
import csv
import re
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
from collections.abc import Callable
from collections import defaultdict

from scipy.ndimage import gaussian_filter


# --- CSV column mapping ---
_CSV_REQUIRED = {
    "storm_num": "Storm Number",
    "time": "Time",
    "lon": "Longitude",
    "lat": "Latitude",
    "u_motion": "u-motion",
    "v_motion": "v-motion",
}
_CSV_OPTIONAL = {
    "ef": "Max Tor Intensity",
    "tor_count": "Tor Count",
    "refl_max": "Column-Max Refl.",
    "echo_top_30": "30-dBZ Echo Top",
}

# --- NetCDF variable names ---
_NC_SPARSE_VARS = [
    "Reflectivity", "SpectrumWidth", "AzShear", "Divergence",
    "DifferentialReflectivity", "CorrelationCoefficient",
]
_NC_REQUIRED = ["Longitude", "Latitude", "Altitude", "index", "Nradobs"]


def _parse_nc_time(filename: str) -> datetime | None:
    """Parse radar timestamp from NEXRAD filename."""
    m = re.search(r"(\d{8}T\d{6}Z)", filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ")
    m = re.search(r"(\d{12,14})", filename)
    if m:
        ts = m.group(1)
        if len(ts) == 14:
            return datetime.strptime(ts, "%Y%m%d%H%M%S")
        return datetime.strptime(ts[:12], "%Y%m%d%H%M")
    return None


def _parse_csv_time(time_str: str) -> datetime | None:
    """Parse CSV time string YYYYMMDDTHHMMZ."""
    s = time_str.strip()
    try:
        return datetime.strptime(s, "%Y%m%dT%H%MZ")
    except ValueError:
        pass
    m = re.search(r"(\d{8}T\d{4}Z?)", s)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%MZ")
    return None


def _safe_nc_open(nc_path: Path):
    """安全打开 NetCDF 文件，自动处理含中文路径的编码问题。

    netCDF4 底层 C 库在 Windows 上可能无法处理含非 ASCII 字符的路径，
    此时将文件复制到临时目录再打开。
    """
    from netCDF4 import Dataset
    try:
        return Dataset(str(nc_path), "r")
    except (FileNotFoundError, OSError):
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp_path = tmp.name
        tmp.close()
        shutil.copy2(str(nc_path), tmp_path)
        return Dataset(tmp_path, "r")


def _reconstruct_dense(nc_path: Path, var_names: list[str],
                       grid_shape: tuple = (29, 480, 480)) -> dict[str, np.ndarray]:
    """Reconstruct dense 3D grid from sparse index-based NEXRAD .nc files.

    Args:
        nc_path: path to .nc file
        var_names: sparse variable names to reconstruct
        grid_shape: (L, H, W) = (29, 480, 480)

    Returns:
        {var_name: dense np.ndarray(L, H, W), NaN for unobserved cells}
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        return {}

    L, H, W = grid_shape
    grid_flat_size = L * H * W

    try:
        with _safe_nc_open(nc_path) as ds:
            idx = ds.variables["index"][:]
            valid = (idx >= 0) & (idx < grid_flat_size)

            results = {}
            for vname in var_names:
                if vname not in ds.variables:
                    continue
                values = ds.variables[vname][:]
                if values.ndim != 1 or len(values) != len(idx):
                    continue
                dense_flat = np.full(grid_flat_size, np.nan, dtype=np.float32)
                dense_flat[idx[valid]] = values[valid]
                results[vname] = dense_flat.reshape(L, H, W)

            if "Nradobs" in ds.variables:
                results["Nradobs"] = ds.variables["Nradobs"][:].astype(np.float32)

            return results
    except Exception as e:
        import sys
        print(f"[WARNING] Failed to read NetCDF file {nc_path.name}: {e}", file=sys.stderr)
        return {}


def _reconstruct_crop(nc_path: Path, var_names: list[str],
                      lat_sub: np.ndarray, lon_sub: np.ndarray,
                      l_use: int, grid_shape: tuple = (29, 480, 480),
                      vertical_stride: int = 1) -> dict[str, np.ndarray]:
    """Reconstruct only the cropped region from sparse NEXRAD .nc data.

    Avoids building the full 29×480×480 grid (~26 MB/channel) when only a
    128×128 crop is needed — reduces per-channel allocation by ~93%.

    Args:
        nc_path: path to .nc file
        var_names: sparse variable names to reconstruct
        lat_sub: array of original y-indices (length H_crop)
        lon_sub: array of original x-indices (length W_crop)
        l_use: number of vertical levels to include
        grid_shape: (L, H, W) of the original full grid
        vertical_stride: stride for vertical downsampling

    Returns:
        {var_name: np.ndarray(L_crop, H_crop, W_crop), NaN for unobserved}
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        return {}

    L_full, H_full, W_full = grid_shape
    H_crop = len(lat_sub)
    W_crop = len(lon_sub)

    # Effective levels after stride
    l_indices = np.arange(0, l_use, vertical_stride)
    L_crop = len(l_indices)

    # Lookup tables: original index → crop position (-1 = not in crop)
    y_to_crop = np.full(H_full, -1, dtype=np.int32)
    x_to_crop = np.full(W_full, -1, dtype=np.int32)
    l_to_crop = np.full(L_full, -1, dtype=np.int32)
    y_to_crop[lat_sub] = np.arange(H_crop)
    x_to_crop[lon_sub] = np.arange(W_crop)
    for crop_l, orig_l in enumerate(l_indices):
        l_to_crop[orig_l] = crop_l

    try:
        with _safe_nc_open(nc_path) as ds:
            idx = ds.variables["index"][:]
            grid_flat_size = L_full * H_full * W_full

            # Flat → (l, y, x)
            l_idx = idx // (H_full * W_full)
            remainder = idx % (H_full * W_full)
            y_idx = remainder // W_full
            x_idx = remainder % W_full

            # Map to crop coordinates
            crop_l = l_to_crop[l_idx]
            crop_y = y_to_crop[y_idx]
            crop_x = x_to_crop[x_idx]

            in_crop = (
                (crop_l >= 0) & (crop_y >= 0) & (crop_x >= 0)
                & (idx >= 0) & (idx < grid_flat_size)
            )

            crop_flat = (
                crop_l[in_crop] * H_crop * W_crop
                + crop_y[in_crop] * W_crop
                + crop_x[in_crop]
            )

            results: dict[str, np.ndarray] = {}
            crop_total = L_crop * H_crop * W_crop
            for vname in var_names:
                if vname not in ds.variables:
                    continue
                values = ds.variables[vname][:]
                if values.ndim != 1 or len(values) != len(idx):
                    continue
                dense_crop = np.full(crop_total, np.nan, dtype=np.float32)
                dense_crop[crop_flat] = values[in_crop]
                results[vname] = dense_crop.reshape(L_crop, H_crop, W_crop)

            if "Nradobs" in ds.variables:
                results["Nradobs"] = ds.variables["Nradobs"][:].astype(np.float32)

            return results
    except Exception as e:
        import sys
        print(f"[WARNING] Failed to read NetCDF file {nc_path.name}: {e}", file=sys.stderr)
        return {}


def _fill_nan(dense: np.ndarray) -> np.ndarray:
    """Fill NaN: per-level mean fill, all-NaN level -> 0.
    
    Kept for backward compatibility; new pipeline prefers _check_nan_and_drop.
    """
    result = dense.copy()
    for k in range(result.shape[0]):
        layer = result[k]
        nan_mask = np.isnan(layer)
        if nan_mask.all():
            result[k] = 0.0
        elif nan_mask.any():
            layer[nan_mask] = np.nanmean(layer)
    return result


def _check_nan_excessive(dense: np.ndarray, max_nan_ratio: float = 0.3) -> bool:
    """Check if any layer has excessive NaN ratio.
    
    Returns True if the sample should be dropped.
    Per 方法.docx: missing values → delete the sample.
    """
    for k in range(dense.shape[0]):
        nan_ratio = np.isnan(dense[k]).mean()
        if nan_ratio > max_nan_ratio:
            return True
    return False


class DataPreprocessor:
    """NEXRAD Level II radar + storm track data preprocessor."""

    def __init__(self, config: dict):
        data_cfg = config.get("data", {})

        self.history_steps = data_cfg.get("history_steps", 12)
        self.future_steps = data_cfg.get("future_steps", 36)
        self.total_steps = self.history_steps + self.future_steps

        self.spatial_degree = data_cfg.get("spatial_degree", 2.56)
        self.grid_size = data_cfg.get("grid_size", 128)
        self.vertical_layers = data_cfg.get("vertical_layers", None)
        self.vertical_stride = data_cfg.get("vertical_stride", 1)

        self.channel_variables = data_cfg.get("channel_variables",
            ["Reflectivity", "SpectrumWidth", "AzShear", "Divergence"])
        self.target_variable = data_cfg.get("target_variable", "AzShear")

        # P1: Storm motion channels
        self.use_storm_motion = data_cfg.get("use_storm_motion", False)
        self.storm_motion_channels = data_cfg.get(
            "storm_motion_channels", ["storm_u", "storm_v"])
        # Normalize storm motion to this range (m/s)
        self.storm_motion_range = data_cfg.get("storm_motion_range", [-50.0, 50.0])

        # P2: Environmental parameter channels (requires HRRR or pre-computed file)
        self.use_env_params = data_cfg.get("use_env_params", False)
        self.env_param_channels = data_cfg.get(
            "env_param_channels", ["ebwd", "srh500"])
        self.env_params_path = data_cfg.get("env_params_path", "")
        self._env_params_cache: dict[str, dict] | None = None
        self.env_param_ranges = data_cfg.get("env_param_ranges", {
            "ebwd": [0.0, 50.0],      # m/s, effective bulk wind difference
            "srh500": [0.0, 500.0],    # m²/s², 0-500m SRH
        })

        norm_cfg = data_cfg.get("normalization", {})
        self.norm_config = norm_cfg
        self.nan_max_ratio = data_cfg.get("nan_max_ratio", 0.3)
        self.gaussian_sigma = data_cfg.get("gaussian_sigma", 1.0)

        self.processed_dir = Path(data_cfg.get("processed_dir", "./processed"))
        self.radar_time_tolerance = timedelta(minutes=2.5)

        stride_raw = data_cfg.get("window_stride", None)
        self.window_stride = stride_raw if stride_raw is not None else self.history_steps
        self.num_workers = int(data_cfg.get("num_workers", 0))

        self.lon_arr: np.ndarray | None = None
        self.lat_arr: np.ndarray | None = None
        self._report: list[dict] = []
        self._report_lock: object | None = None  # threading.Lock when using workers

    def _load_env_params_cache(self):
        """Load pre-computed environmental parameters from JSON/CSV file.
        
        Expected format: {"storm_id_timestep": {"ebwd": 25.3, "srh500": 180.0}, ...}
        or CSV with columns: storm_num, time, ebwd, srh500, ...
        """
        if self._env_params_cache is not None:
            return
        if not self.env_params_path:
            self._env_params_cache = {}
            return

        import json
        ep_path = Path(self.env_params_path)
        if not ep_path.exists():
            self._env_params_cache = {}
            return

        cache: dict[str, dict] = {}
        if ep_path.suffix == '.json':
            with open(ep_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        elif ep_path.suffix == '.csv':
            with open(ep_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = [h.strip().lower() for h in next(reader)]
                for row in reader:
                    storm_num = row[0].strip()
                    time_str = row[1].strip()
                    key = f"{storm_num}_{time_str}"
                    vals = {}
                    for ch in self.env_param_channels:
                        if ch in header:
                            idx = header.index(ch)
                            vals[ch] = float(row[idx]) if idx < len(row) else 0.0
                    if vals:
                        cache[key] = vals
        self._env_params_cache = cache

    def _get_env_params(self, storm_num: str, dt: datetime) -> dict[str, float]:
        """Get environmental parameters for a specific storm at a specific time."""
        if self._env_params_cache is None:
            self._load_env_params_cache()
        if not self._env_params_cache:
            return {}
        time_str = dt.strftime("%Y%m%dT%H%MZ")
        key = f"{storm_num}_{time_str}"
        return self._env_params_cache.get(key, {})

    def validate_datasets(self, nexrad_dir: str, tracks_dir: str) -> dict:
        """Validate that both dataset directories are readable.

        Returns:
            {"valid": bool, "nc_count": int, "csv_count": int, "errors": [str]}
        """
        result = {"valid": True, "nc_count": 0, "csv_count": 0, "errors": []}

        # Validate NetCDF directory
        nc_dir = Path(nexrad_dir)
        if not nc_dir.exists():
            result["valid"] = False
            result["errors"].append(f"NetCDF dir not found: {nexrad_dir}")
        else:
            nc_files = sorted(nc_dir.glob("*.nc"))
            result["nc_count"] = len(nc_files)
            if len(nc_files) == 0:
                result["valid"] = False
                result["errors"].append(f"No .nc files found: {nexrad_dir}")
            else:
                try:
                    from netCDF4 import Dataset
                    with _safe_nc_open(nc_files[0]) as ds:
                        missing = [v for v in _NC_REQUIRED if v not in ds.variables]
                    if missing:
                        result["valid"] = False
                        result["errors"].append(
                            f"NetCDF missing required vars: {missing}, "
                            f"actual: {list(ds.variables.keys())}"
                        )
                except ImportError:
                    result["valid"] = False
                    result["errors"].append("Need netCDF4: pip install netCDF4")
                except Exception as e:
                    result["valid"] = False
                    result["errors"].append(f"NetCDF read failed: {e}")

        # Validate CSV directory
        csv_dir = Path(tracks_dir)
        if not csv_dir.exists():
            result["valid"] = False
            result["errors"].append(f"CSV dir not found: {tracks_dir}")
        else:
            csv_files = sorted(csv_dir.glob("*.csv"))
            result["csv_count"] = len(csv_files)
            if len(csv_files) == 0:
                result["valid"] = False
                result["errors"].append(f"No .csv files in: {tracks_dir}")
            else:
                try:
                    with open(csv_files[0], "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header_raw = next(reader)
                    header = [h.strip() for h in header_raw]
                    missing = []
                    for key, col_name in _CSV_REQUIRED.items():
                        if col_name not in header:
                            missing.append(col_name)
                    if missing:
                        result["valid"] = False
                        result["errors"].append(
                            f"CSV missing required cols: {missing}, got: {header[:15]}"
                        )
                except Exception as e:
                    result["valid"] = False
                    result["errors"].append(f"CSV read failed: {e}")

        return result

    def process_events(self, nexrad_dir: str, tracks_dir: str,
                       progress_callback: Callable[[int, int, str], None] | None = None):
        """Process all storm events to generate preprocessed .npz files.

        Each storm can produce multiple samples via sliding windows (stride =
        history_steps).  A 6-hour storm yields ~5 samples instead of 1.

        Supports multi-threaded processing via ``num_workers`` config.

        After processing, writes ``preprocess_report.json`` to ``processed_dir``
        listing every storm and whether it was used or skipped (with reason).

        Args:
            nexrad_dir: NEXRAD .nc file directory
            tracks_dir: storm track .csv file directory
            progress_callback: (current, total, message)
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Index all radar files
        nc_files = self._index_nc_files(nexrad_dir)
        if not nc_files:
            raise RuntimeError(f"No .nc files found: {nexrad_dir}")

        # Pre-load coordinates from first file (shared read-only across threads)
        self._load_coordinates(list(nc_files.values())[0])

        # 2. Parse all CSV files, group by (storm_num, date)
        all_storms = self._parse_all_csvs(tracks_dir)
        storm_ids = sorted(all_storms.keys())
        total = len(storm_ids)

        self._report = []
        self._report_lock = threading.Lock()

        processed = 0
        total_samples = 0
        skipped = 0

        if self.num_workers > 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._process_one_storm, sid,
                                    all_storms[sid], nc_files): sid
                    for sid in storm_ids
                }
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        entry, samples, was_skipped = future.result()
                    except Exception as e:
                        entry = {"storm_id": sid, "status": "error",
                                 "skip_reason": f"worker failed: {e}"}
                        samples = 0
                        was_skipped = True
                        with self._report_lock:
                            self._report.append(entry)

                    processed += 1
                    total_samples += samples
                    if was_skipped:
                        skipped += 1

                    if progress_callback:
                        if was_skipped:
                            progress_callback(processed, total,
                                f"Storm #{sid}: skipped ({entry.get('skip_reason', '')})")
                        else:
                            progress_callback(processed, total,
                                f"Storm #{sid}: {entry.get('samples_saved', 0)} sample(s) "
                                f"from {entry.get('windows_found', 0)} window(s)")
        else:
            for sid in storm_ids:
                entry, samples, was_skipped = self._process_one_storm(
                    sid, all_storms[sid], nc_files)
                processed += 1
                total_samples += samples
                if was_skipped:
                    skipped += 1

                if progress_callback:
                    if was_skipped:
                        progress_callback(processed, total,
                            f"Storm #{sid}: no valid window ({entry.get('skip_reason', '')})")
                    else:
                        progress_callback(processed, total,
                            f"Storm #{sid}: {entry.get('samples_saved', 0)} sample(s) "
                            f"from {entry.get('windows_found', 0)} window(s)")

        # Write skip report
        self._write_report(total, total_samples, skipped)

        # Summary
        if progress_callback:
            progress_callback(total, total,
                f"Done: {total_samples} samples from {processed - skipped}/{processed} storms "
                f"({skipped} skipped)")
            progress_callback(total, total,
                f"Report written to {self.processed_dir / 'preprocess_report.json'}")


    def _process_one_storm(self, sid: str, rows: list[dict],
                           nc_files: dict[int, Path]
                           ) -> tuple[dict, int, bool]:
        """Process a single storm event (thread-safe).

        Args:
            sid: storm event ID
            rows: CSV rows for this storm
            nc_files: indexed radar files {timestamp: path}

        Returns:
            (report_entry, samples_saved, was_skipped)
        """
        entry: dict = {"storm_id": sid, "csv_rows": len(rows)}

        try:
            rows_sorted = sorted(rows, key=lambda r: r["time"])
            entry["time_range"] = (
                rows_sorted[0]["time"].strftime("%Y-%m-%dT%H:%MZ") +
                " ~ " +
                rows_sorted[-1]["time"].strftime("%Y-%m-%dT%H:%MZ")
            )

            windows, skip_reason = self._find_all_windows(rows_sorted, nc_files)
            if not windows:
                entry["status"] = "skipped"
                entry["skip_reason"] = skip_reason
                with self._report_lock:
                    self._report.append(entry)
                return entry, 0, True

            ef_label = max((r.get("ef", 0) or 0) for r in rows_sorted)

            win_success = 0
            win_failed = 0
            for win_idx, (matched_rows, matched_files) in enumerate(windows):
                sample = self._build_sample(sid, matched_rows, matched_files,
                                            ef_label)
                if sample is None:
                    win_failed += 1
                    continue

                if len(windows) > 1:
                    out_path = self.processed_dir / f"storm_{sid}_w{win_idx}.npz"
                else:
                    out_path = self.processed_dir / f"storm_{sid}.npz"
                np.savez_compressed(out_path, **sample)
                win_success += 1

            entry["status"] = "success"
            entry["windows_found"] = len(windows)
            entry["samples_saved"] = win_success
            if win_failed > 0:
                entry["windows_failed"] = win_failed
                entry["fail_reason"] = "radar crop extraction returned None"
            with self._report_lock:
                self._report.append(entry)
            return entry, win_success, False

        except Exception as e:
            entry["status"] = "error"
            entry["skip_reason"] = str(e)
            with self._report_lock:
                self._report.append(entry)
            return entry, 0, True

    def _write_report(self, total: int, total_samples: int, skipped: int):
        """Write ``preprocess_report.json`` to ``processed_dir``."""
        import json

        reason_counts: dict[str, int] = {}
        for entry in self._report:
            if entry["status"] == "skipped":
                reason = entry.get("skip_reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            elif entry["status"] == "error":
                reason_counts["exception"] = reason_counts.get("exception", 0) + 1

        success = total - skipped
        summary = {
            "total_storms": total,
            "successful": success,
            "skipped": skipped,
            "total_samples": total_samples,
            "skip_reasons": reason_counts,
        }

        report = {
            "summary": summary,
            "config": {
                "history_steps": self.history_steps,
                "future_steps": self.future_steps,
                "total_steps": self.total_steps,
                "radar_time_tolerance_s": self.radar_time_tolerance.total_seconds(),
            },
            "storms": self._report,
        }

        report_path = self.processed_dir / "preprocess_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return report_path

    def _index_nc_files(self, nexrad_dir: str) -> dict[int, Path]:
        """Index .nc files by Unix timestamp."""
        nc_path = Path(nexrad_dir)
        result = {}
        for f in sorted(nc_path.glob("*.nc")):
            dt = _parse_nc_time(f.name)
            if dt:
                result[int(dt.timestamp())] = f
        return result

    def _load_coordinates(self, nc_file: Path):
        """Load longitude/latitude coordinates."""
        try:
            from netCDF4 import Dataset
            with _safe_nc_open(nc_file) as ds:
                self.lon_arr = ds.variables["Longitude"][:].astype(np.float64)
                self.lat_arr = ds.variables["Latitude"][:].astype(np.float64)
        except Exception as e:
            import sys
            print(f"[WARNING] Failed to load coordinates from {nc_file.name}, "
                  f"using default linspace: {e}", file=sys.stderr)
            self.lon_arr = np.linspace(-130, -60, 480, dtype=np.float64)
            self.lat_arr = np.linspace(20, 55, 480, dtype=np.float64)

    def _parse_all_csvs(self, tracks_dir: str) -> dict[str, list[dict]]:
        """Parse all CSV files, group by Storm Number + date.

        Storm Number is reused across dates. Use storm_{num}_date_{YYYYMMDD} as event ID.

        Returns:
            {event_id: [{"time": datetime, "lon": float, "lat": float,
                         "u": float, "v": float, "ef": int, ...}, ...]}
        """
        csv_dir = Path(tracks_dir)
        storms = defaultdict(list)

        for csv_file in sorted(csv_dir.glob("*.csv")):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header_raw = next(reader)
                    header = [h.strip() for h in header_raw]
                    try:
                        next(reader)  # skip units row
                    except StopIteration:
                        pass

                    # Resolve column indices
                    col_idx = {}
                    for key, col_name in _CSV_REQUIRED.items():
                        if col_name in header:
                            col_idx[key] = header.index(col_name)
                    for key, col_name in _CSV_OPTIONAL.items():
                        if col_name in header:
                            col_idx[key] = header.index(col_name)

                    if "storm_num" not in col_idx or "time" not in col_idx:
                        continue

                    for row in reader:
                        try:
                            storm_num = int(row[col_idx["storm_num"]].strip())
                        except (ValueError, IndexError):
                            continue

                        dt = _parse_csv_time(row[col_idx["time"]])
                        if dt is None:
                            continue

                        entry = {"time": dt}
                        for key in ["lon", "lat", "u_motion", "v_motion"]:
                            if key in col_idx:
                                try:
                                    entry[key.split("_")[0]] = float(row[col_idx[key]].strip())
                                except (ValueError, IndexError):
                                    entry[key.split("_")[0]] = 0.0

                        for key in ["ef", "tor_count"]:
                            if key in col_idx:
                                try:
                                    entry[key] = int(float(row[col_idx[key]].strip()))
                                except (ValueError, IndexError):
                                    entry[key] = 0

                        entry.setdefault("lon", 0.0)
                        entry.setdefault("lat", 0.0)
                        entry.setdefault("u", 0.0)
                        entry.setdefault("v", 0.0)
                        entry.setdefault("ef", 0)

                        date_str = dt.strftime("%Y%m%d")
                        event_id = f"{storm_num}_{date_str}"
                        storms[event_id].append(entry)
            except Exception as e:
                import sys
                print(f"[WARNING] Failed to parse CSV {csv_file.name}: {e}", file=sys.stderr)
                continue

        return dict(storms)

    def _build_sample(self, storm_id: str, matched_rows: list[dict],
                      matched_files: list[Path], ef_label: int) -> dict | None:
        """Build a single training sample from a matched time window.

        Returns:
            {"input": (T_in,H,W,L,C), "target": (T_out,H,W,L),
             "storm_uv": (T_total,2), "storm_lon_lat": (T_total,2),
             "ef_label": int, "channel_names": [str], "norm_params": dict}
            or None if radar extraction fails.
        """
        half_deg = self.spatial_degree / 2.0
        n_levels = self.vertical_layers or 29

        radar_volumes = []
        storm_uv_list = []
        storm_ll_list = []

        for row, nc_file in zip(matched_rows, matched_files):
            center_lon = row.get("lon", 0.0)
            center_lat = row.get("lat", 0.0)
            storm_u = row.get("u", 0.0)
            storm_v = row.get("v", 0.0)

            vol = self._extract_radar_crop(nc_file, center_lon, center_lat,
                                           half_deg, n_levels)
            if vol is None:
                return None

            radar_volumes.append(vol)
            storm_uv_list.append([storm_u, storm_v])
            storm_ll_list.append([center_lon, center_lat])

        radar_array = np.stack(radar_volumes, axis=0)  # [T_total, H, W, L, C_radar]

        input_data = radar_array[:self.history_steps]     # [T_in, H, W, L, C_radar]
        target_full = radar_array[self.history_steps:self.total_steps]  # [T_out, H, W, L, C_radar]

        H, W, L = input_data.shape[1:4]
        T_in = self.history_steps
        T_out = self.future_steps
        
        # ── P1: Storm motion broadcast channels ──
        extra_channels = []
        extra_channel_names = list(self.channel_variables)
        extra_norm_params: dict[str, dict] = {}

        if self.use_storm_motion:
            storm_uv_array = np.array(storm_uv_list, dtype=np.float32)  # (T_total, 2)
            sm_min, sm_max = self.storm_motion_range
            sm_range = max(sm_max - sm_min, 1e-8)
            for ci, ch_name in enumerate(self.storm_motion_channels):
                sm_val = storm_uv_array[:, ci]  # (T_total,)
                sm_norm = (np.clip(sm_val, sm_min, sm_max) - sm_min) / sm_range
                # Broadcast to (T_total, H, W, L)
                sm_field = sm_norm[:, None, None, None] * np.ones((1, H, W, L), dtype=np.float32)
                sm_in = sm_field[:T_in]   # (T_in, H, W, L)
                sm_out = sm_field[T_in:]  # (T_out, H, W, L)
                extra_channels.append((sm_in, sm_out, ch_name))
                extra_channel_names.append(ch_name)
                extra_norm_params[ch_name] = {
                    "method": "minmax", "min": sm_min, "max": sm_max,
                }

        # ── P2: Environmental parameter broadcast channels ──
        if self.use_env_params:
            storm_num_str = storm_id.split("_")[0]
            env_in_list = []
            env_out_list = []
            for ti, row in enumerate(matched_rows):
                ep = self._get_env_params(storm_num_str, row["time"])
                vals = [ep.get(ch, 0.0) for ch in self.env_param_channels]
                if ti < T_in:
                    env_in_list.append(vals)
                else:
                    env_out_list.append(vals)
            for ci, ch_name in enumerate(self.env_param_channels):
                ev_range = self.env_param_ranges.get(ch_name, [0.0, 100.0])
                ev_min, ev_max = ev_range[0], ev_range[1]
                ev_range_val = max(ev_max - ev_min, 1e-8)
                # Input
                ev_vals_in = np.array([ev[ci] for ev in env_in_list], dtype=np.float32) if env_in_list else np.zeros(T_in, dtype=np.float32)
                ev_norm_in = (np.clip(ev_vals_in, ev_min, ev_max) - ev_min) / ev_range_val
                ev_field_in = ev_norm_in[:, None, None, None] * np.ones((1, H, W, L), dtype=np.float32)
                # Output
                ev_vals_out = np.array([ev[ci] for ev in env_out_list], dtype=np.float32) if env_out_list else np.zeros(T_out, dtype=np.float32)
                ev_norm_out = (np.clip(ev_vals_out, ev_min, ev_max) - ev_min) / ev_range_val
                ev_field_out = ev_norm_out[:, None, None, None] * np.ones((1, H, W, L), dtype=np.float32)
                extra_channels.append((ev_field_in, ev_field_out, ch_name))
                extra_channel_names.append(ch_name)
                extra_norm_params[ch_name] = {
                    "method": "minmax", "min": ev_min, "max": ev_max,
                }

        # Prepend extra input channels (before normalization of radar channels)
        extra_input_list = []
        extra_output_list = []
        for ein, eout, _ in extra_channels:
            extra_input_list.append(ein[..., None])   # (T_in, H, W, L, 1)
            extra_output_list.append(eout[..., None])  # (T_out, H, W, L, 1)
        if extra_input_list:
            extra_input = np.concatenate(extra_input_list, axis=-1)  # (T_in, H, W, L, N_extra)
        else:
            extra_input = None

        # Target: AzShear channel → Gaussian filter → ζ = 2 × AzShear
        try:
            target_ch = self.channel_variables.index(self.target_variable)
        except ValueError:
            target_ch = min(2, radar_array.shape[-1] - 1)
        # 对每个时间步的 AzShear 层应用高斯滤波以降低雷达反演噪声
        sigma = self.gaussian_sigma
        target_azshear_filtered = np.zeros_like(target_full[..., target_ch])
        for t in range(target_full.shape[0]):
            for l in range(target_full.shape[3]):
                target_azshear_filtered[t, :, :, l] = gaussian_filter(
                    target_full[t, :, :, l, target_ch], sigma=sigma
                )
        target_vorticity = target_azshear_filtered * 2.0  # ζ = 2 × AzShear

        # 标签独立 Min-Max 归一化（训练在归一化空间，评估反归一化）
        target_min = float(target_vorticity.min())
        target_max = float(target_vorticity.max())
        target_range = max(target_max - target_min, 1e-8)
        target_normalized = (target_vorticity - target_min) / target_range

        # Normalize (all channels use Min-Max per 方法.docx)
        norm_params = {}
        input_normalized = np.zeros_like(input_data)
        for c, ch_name in enumerate(self.channel_variables):
            ch_data = input_data[..., c]
            ch_norm_cfg = self.norm_config.get(ch_name, {})
            method = ch_norm_cfg.get("method", "minmax")

            if method == "minmax":
                vmin = ch_norm_cfg.get("min", float(np.nanmin(ch_data)))
                vmax = ch_norm_cfg.get("max", float(np.nanmax(ch_data)))
                ch_norm = np.clip(ch_data, vmin, vmax)
                ch_norm = (ch_norm - vmin) / max(vmax - vmin, 1e-8)
                norm_params[ch_name] = {"method": "minmax", "min": vmin, "max": vmax}
            else:
                # 降级为 minmax（所有通道统一）
                vmin = float(np.nanmin(ch_data))
                vmax = float(np.nanmax(ch_data))
                ch_norm = np.clip(ch_data, vmin, vmax)
                ch_norm = (ch_norm - vmin) / max(vmax - vmin, 1e-8)
                norm_params[ch_name] = {"method": "minmax", "min": vmin, "max": vmax}

            input_normalized[..., c] = ch_norm.astype(np.float32)

        # P1-P2 extra channels：跳过 NaN 检测（它们是广播常量）
        if extra_input is not None:
            input_normalized = np.concatenate([input_normalized, extra_input], axis=-1)

        norm_params["channel_order"] = extra_channel_names
        norm_params["target"] = {"method": "minmax", "min": target_min, "max": target_max}

        sample_dict = {
            "input": input_normalized.astype(np.float32),
            "target": target_normalized.astype(np.float32),
            "target_raw": target_vorticity.astype(np.float32),
            "storm_uv": np.array(storm_uv_list, dtype=np.float32),
            "storm_lon_lat": np.array(storm_ll_list, dtype=np.float32),
            "ef_label": int(ef_label),
            "channel_names": np.array(extra_channel_names, dtype=object),
            "norm_params": np.array([norm_params], dtype=object),
        }

        # 附带未来帧的额外通道（供 decoder 物理模块使用）
        if extra_output_list:
            extra_output = np.concatenate(extra_output_list, axis=-1)
            sample_dict["extra_output"] = extra_output.astype(np.float32)

        return sample_dict

    def _find_all_windows(self, rows: list[dict], nc_files: dict[int, Path]
                           ) -> tuple[list[tuple[list[dict], list[Path]]], str | None]:
        """Find all valid sliding time windows matching radar files.

        Uses a sliding window with stride=history_steps (1 hour) so a 6-hour
        storm yields ~5 samples instead of 1.  Each window must have
        total_steps consecutive radar-matched CSV rows within a valid time span.

        Returns:
            (windows, skip_reason) — windows may be empty; skip_reason is None
            if windows found, otherwise a human-readable string.
        """
        n = len(rows)
        if n < self.total_steps:
            return [], f"only {n} CSV rows, need ≥{self.total_steps}"

        nc_ts_list = sorted(nc_files.keys())

        def find_nearest_nc(t: datetime) -> Path | None:
            target_ts = int(t.timestamp())
            idx = np.searchsorted(nc_ts_list, target_ts)
            best = None
            best_diff = self.radar_time_tolerance.total_seconds()
            for i in [idx - 1, idx, idx + 1]:
                if 0 <= i < len(nc_ts_list):
                    diff = abs(nc_ts_list[i] - target_ts)
                    if diff <= best_diff:
                        best_diff = diff
                        best = nc_files[nc_ts_list[i]]
            return best

        # Filter rows with radar matches
        matched_indices = []
        for i, row in enumerate(rows):
            if find_nearest_nc(row["time"]) is not None:
                matched_indices.append(i)

        if len(matched_indices) < self.total_steps:
            return [], (
                f"only {len(matched_indices)}/{n} CSV rows have radar matches, "
                f"need ≥{self.total_steps}"
            )

        # Sliding window over matched indices with configurable stride
        stride = self.window_stride
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
            # Diagnose why: time-gap issue
            first_idx = matched_indices[0]
            last_idx = matched_indices[-1]
            actual_span = rows[last_idx]["time"] - rows[first_idx]["time"]
            return [], (
                f"no contiguous window of {self.total_steps} radar-matched rows "
                f"({len(matched_indices)} matched over {n} CSV rows, "
                f"span {actual_span}, need ≤{expected_span + timedelta(minutes=5)})"
            )

        return windows, None

    def _extract_radar_crop(self, nc_file: Path, center_lon: float,
                            center_lat: float, half_deg: float,
                            n_levels: int) -> np.ndarray | None:
        """Reconstruct cropped region directly from sparse NEXRAD .nc file.

        Computes crop indices first, then only reconstructs the needed sub-region
        (avoids building full 480×480 grid for 93% less allocation per channel).

        Returns:
            [H, W, L, C] array, or None
        """
        if self.lon_arr is None or self.lat_arr is None:
            return None

        lon_mask = (self.lon_arr >= center_lon - half_deg) & (self.lon_arr <= center_lon + half_deg)
        lat_mask = (self.lat_arr >= center_lat - half_deg) & (self.lat_arr <= center_lat + half_deg)
        lon_idx = np.where(lon_mask)[0]
        lat_idx = np.where(lat_mask)[0]

        if len(lon_idx) < 4 or len(lat_idx) < 4:
            return None

        lon_sub = np.round(np.linspace(lon_idx[0], lon_idx[-1], self.grid_size)).astype(int)
        lat_sub = np.round(np.linspace(lat_idx[0], lat_idx[-1], self.grid_size)).astype(int)

        dense_crop = _reconstruct_crop(nc_file, self.channel_variables,
                                       lat_sub, lon_sub, n_levels,
                                       vertical_stride=self.vertical_stride)
        if not dense_crop:
            return None

        # NaN threshold from config (default 0.3)
        nan_threshold = self.nan_max_ratio
        channels = []
        for vname in self.channel_variables:
            if vname not in dense_crop:
                continue
            data = dense_crop[vname]  # (L, H_crop, W_crop)
            if _check_nan_excessive(data, nan_threshold):
                return None  # Drop sample per 方法.docx
            data = _fill_nan(data)  # Fill remaining sparse NaN after threshold check
            data = np.transpose(data, (1, 2, 0))  # → (H, W, L)
            channels.append(data.astype(np.float32))

        if len(channels) < len(self.channel_variables):
            return None

        return np.stack(channels, axis=-1)
