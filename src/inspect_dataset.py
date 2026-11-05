#!/usr/bin/env python3
"""
inspect_dataset.py

Reproducible audit helper for the PROTECT-90 / hv_double_line_90kv dataset.

Features:
- CSV metadata inspection with parsed-dtype auditing
- ZIP inventory without full extraction
- Optional ZIP integrity check via zipfile.testzip() (off by default)
- Selective PKL waveform inspection
- Strict timestamp and waveform value validation
- Missing-value and infinite-value checks
- Metadata-to-filename matching
- Channel summary

Usage:
    python inspect_dataset.py --csv hv_double_line_90kv_labels.csv
    python inspect_dataset.py --csv hv_double_line_90kv_labels.csv --json-out metadata_audit_summary.json
    python inspect_dataset.py --zip hv_double_line_90kv_preprocessed_data.zip
    python inspect_dataset.py --zip hv_double_line_90kv_preprocessed_data.zip --testzip
    python inspect_dataset.py --pkl-dir pkl_out
    python inspect_dataset.py --pkl pkl_out/0_sample.pkl
    python inspect_dataset.py --csv hv_double_line_90kv_labels.csv --zip hv_double_line_90kv_preprocessed_data.zip --pkl-dir pkl_out

This script does not load the full archive into RAM and does not assume undocumented
column names. It fails loudly when expected structure is not found, and it reports
conversion failures explicitly when CSV fields do not parse as expected types.
"""

import argparse
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SAMPLES = 6400
EXPECTED_FS_HZ = 6400
EXPECTED_DURATION_S = 1.0
EXPECTED_INTERVAL_S = EXPECTED_DURATION_S / EXPECTED_SAMPLES


def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    # Load as strings first so we can audit raw CSV formatting explicitly,
    # then derive typed views via pd.to_numeric / bool coercion.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])
    return df


def _parse_int_series(s: pd.Series) -> dict:
    coerced = pd.to_numeric(s, errors="coerce", downcast="integer")
    out = {
        "nonnull_total": int(s.notna().sum()),
        "parseable_as_float": int(pd.to_numeric(s, errors="coerce").notna().sum()),
        "parseable_as_int": int(coerced.notna().sum()),
        "invalid_conversion_count": int(s.shape[0] - pd.to_numeric(s, errors="coerce").notna().sum()),
        "unique_parsed": int(coerced.dropna().nunique()) if coerced.notna().sum() else 0,
        "parsed_min": None if coerced.notna().sum() == 0 else int(coerced.min()),
        "parsed_max": None if coerced.notna().sum() == 0 else int(coerced.max()),
        "parsed_unique_values": sorted([int(x) for x in coerced.dropna().unique()]),
    }
    return out


def _parse_float_series(s: pd.Series) -> dict:
    coerced = pd.to_numeric(s, errors="coerce")
    out = {
        "nonnull_total": int(s.notna().sum()),
        "parseable_count": int(coerced.notna().sum()),
        "invalid_conversion_count": int(s.shape[0] - coerced.notna().sum()),
        "nunique_parsed": int(coerced.dropna().nunique()) if coerced.notna().sum() else 0,
        "parsed_min": None if coerced.notna().sum() == 0 else float(coerced.min()),
        "parsed_max": None if coerced.notna().sum() == 0 else float(coerced.max()),
        "parsed_unique_values_sample": [float(x) for x in coerced.dropna().unique()[:8]],
    }
    return out


def _parse_bool_series(s: pd.Series) -> dict:
    s = s.astype(str).str.strip()
    true_set = {"true", "1", "yes"}
    false_set = {"false", "0", "no"}
    lowered = s.str.lower()
    n_true = int(lowered.isin(true_set).sum())
    n_false = int(lowered.isin(false_set).sum())
    n_other = int(s.shape[0] - n_true - n_false)
    parsed = pd.Series([None] * len(s), dtype=object)
    mask_true = lowered.isin(true_set)
    mask_false = lowered.isin(false_set)
    parsed = parsed.where(~mask_true, True)
    parsed = parsed.where(~mask_false, False)
    out = {
        "nonnull_total": int(s.notna().sum()),
        "parsed_true_count": n_true,
        "parsed_false_count": n_false,
        "unparsed_other_count": n_other,
        "parsed_unique_values": sorted({str(v) for v in s.unique()}),
    }
    return out


def inspect_csv(df: pd.DataFrame) -> dict:
    missing = {col: int(df[col].isna().sum()) for col in df.columns}

    dup_sample_id = None
    unique_sample_id = None
    if "sample_id" in df.columns:
        dup_sample_id = int(df["sample_id"].duplicated().sum())
        unique_sample_id = int(df["sample_id"].nunique())

    parsed_types = {}
    numeric_cols = {}
    bool_cols = {}
    int_cols = {}

    for col in df.columns:
        s = df[col]
        info: dict = {
            "csv_storage_dtype": str(df[col].dtype),
            "csv_unique_raw_values": [str(x) for x in s.unique()[:12]],
            "csv_nunique_raw": int(s.nunique()),
            "missing_count": int(s.isna().sum()),
        }

        stripped = s.astype(str).str.strip()
        is_empty = stripped.eq("").sum()
        info["empty_string_count"] = int(is_empty)

        # Heuristic typed parsing.
        if col.lower().endswith("_on") or col.lower() in {
            "line_1_2_b_on", "line_2_3_b_on", "ext_grid_3_on",
        } or set(stripped.str.lower().unique()) <= {"true", "false", "1", "0"}:
            bool_info = _parse_bool_series(s)
            bool_cols[col] = bool_info
            info["interpreted_as"] = "boolean_attempt"
            info.update(bool_info)
            parsed_types[col] = "boolean_attempt"
            continue

        float_info = _parse_float_series(s)
        if float_info["parseable_count"] == s.shape[0] and float_info["invalid_conversion_count"] == 0:
            info["interpreted_as"] = "float"
            info.update(float_info)
            numeric_cols[col] = float_info
            parsed_types[col] = "float"
            continue

        int_info = _parse_int_series(s)
        if int_info["parseable_as_int"] == s.shape[0] and int_info["invalid_conversion_count"] == 0 and float_info["parseable_count"] == s.shape[0]:
            info["interpreted_as"] = "int"
            info.update(int_info)
            int_cols[col] = int_info
            parsed_types[col] = "int"
            continue

        # Fallback: report what parses and what does not.
        info["interpreted_as"] = "mixed_or_text"
        info.update(float_info)
        parsed_types[col] = "mixed_or_text"

    out = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "csv_storage_dtype": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "parsed_as": parsed_types,
        "numeric_cols": numeric_cols,
        "int_cols": int_cols,
        "bool_cols": bool_cols,
        "missing": missing,
        "empty_string_counts": {col: int(df[col].astype(str).str.strip().eq("").sum()) for col in df.columns},
        "duplicated_sample_id": dup_sample_id,
        "unique_sample_id": unique_sample_id,
        "nunique_raw": {col: int(df[col].nunique()) for col in df.columns},
    }
    return out


def zip_inventory(zip_path: str) -> dict:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    # Standard ZIP inventory: header-only metadata, no decompression.
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        infos = zf.infolist()
        total = sum(info.file_size for info in infos)
        compressed = sum(info.compress_size for info in infos)
        return {
            "archive_path": zip_path,
            "member_count": len(names),
            "testzip": None,
            "total_uncompressed_bytes": total,
            "total_compressed_bytes": compressed,
            "members": [
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compress_type": info.compress_type,
                }
                for info in infos
            ],
        }


def zip_inventory_with_testzip(zip_path: str) -> dict:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        bad = zf.testzip()
        infos = zf.infolist()
        total = sum(info.file_size for info in infos)
        compressed = sum(info.compress_size for info in infos)
        return {
            "archive_path": zip_path,
            "member_count": len(names),
            "testzip": None if bad is None else bad,
            "total_uncompressed_bytes": total,
            "total_compressed_bytes": compressed,
            "members": [
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compress_type": info.compress_type,
                }
                for info in infos
            ],
        }


def discover_pkl_files(pkl_dir: str) -> list:
    p = Path(pkl_dir)
    if not p.is_dir():
        raise NotADirectoryError(pkl_dir)
    # Support both naming conventions observed in the bundle.
    patterns = ["*_sample_hv_double_line_90kv.pkl", "*_sample.pkl"]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(p.glob(pattern)):
            found[path.name] = path
    return list(found.values())


def parse_waveform_columns(columns):
    pattern = re.compile(r"^(.+?)_(cur|vol)_L([123])_(A|V)$")
    locations: dict[str, dict[str, set[str]]] = {}
    parsed = []
    for col in columns:
        m = pattern.match(col)
        if not m:
            parsed.append({"column": col, "parsed": False})
            continue
        location, meas, phase, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        parsed.append(
            {
                "column": col,
                "parsed": True,
                "location": location,
                "meas": meas,
                "phase": phase,
                "unit": unit,
            }
        )
        locations.setdefault(location, {"cur": set(), "vol": set()})
        locations[location][meas].add(phase)
    return parsed, locations


def inspect_pkl(pkl_path: str, csv_df: pd.DataFrame = None) -> dict:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    columns = df.columns.tolist()
    parsed, locations = parse_waveform_columns(columns)

    time_col = "time_s"
    if time_col not in df.columns:
        raise ValueError(f"Expected time column {time_col!r} not found in columns")

    time_series = df[time_col].astype(float)
    time_diff = time_series.diff().dropna()

    t_stats = {
        "start": float(time_series.iloc[0]),
        "end": float(time_series.iloc[-1]),
        "n": int(len(time_series)),
        "n_unique": int(time_series.nunique()),
        "n_duplicate": int(len(time_series) - time_series.nunique()),
        "strictly_increasing": bool(time_series.is_monotonic_increasing and time_diff.gt(0).all()),
        "is_monotonic_increasing": bool(time_series.is_monotonic_increasing),
        "dt_min": float(time_diff.min()),
        "dt_max": float(time_diff.max()),
        "dt_mean": float(time_diff.mean()),
        "dt_std": float(time_diff.std(ddof=0)) if len(time_diff) else 0.0,
        "implied_fs_hz": float(1.0 / time_diff.mean()) if len(time_diff) else None,
        "expected_samples": EXPECTED_SAMPLES,
        "expected_interval_s": EXPECTED_INTERVAL_S,
    }

    # Strict timestamp validation.
    ts_valid = True
    ts_valid = ts_valid and (t_stats["n"] == t_stats["expected_samples"])
    ts_valid = ts_valid and t_stats["strictly_increasing"]
    ts_valid = ts_valid and (t_stats["n_duplicate"] == 0)
    ts_valid = ts_valid and (t_stats["n_unique"] == t_stats["n"])

    max_dev = max(abs(float(time_diff.min()) - EXPECTED_INTERVAL_S),
                  abs(float(time_diff.max()) - EXPECTED_INTERVAL_S))
    t_stats["max_abs_interval_deviation_s"] = float(max_dev)
    t_stats["max_abs_interval_deviation_samples"] = float(max_dev / EXPECTED_INTERVAL_S)
    t_stats["timestamp_validation_passed"] = bool(ts_valid)

    # Finite numeric waveform value check.
    df_numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    missing_numeric = {col: int(df_numeric[col].isna().sum()) for col in columns}
    finite_fail = {col: int((~np.isfinite(df_numeric[col].to_numpy())).sum()) for col in columns}
    non_numeric = {col: int((df_numeric[col].isna() & df[col].notna()).sum()) for col in columns}

    waveform_finite_valid = True
    waveform_finite_valid = waveform_finite_valid and all(v == 0 for v in missing_numeric.values())
    waveform_finite_valid = waveform_finite_valid and all(v == 0 for v in finite_fail.values())

    cur_cols = [c for c in columns if c.endswith("_cur_L1_A") or c.endswith("_cur_L2_A") or c.endswith("_cur_L3_A")]
    vol_cols = [c for c in columns if c.endswith("_vol_L1_V") or c.endswith("_vol_L2_V") or c.endswith("_vol_L3_V")]

    def range_of(cols):
        if not cols:
            return None
        s = pd.concat([df_numeric[c] for c in cols])
        return {"min": float(s.min()), "max": float(s.max())}

    current_range = range_of(cur_cols)
    voltage_range = range_of(vol_cols)

    n_channels_per_location = {}
    for loc, v in locations.items():
        n_channels_per_location[loc] = {
            "cur": sorted(list(v["cur"])),
            "vol": sorted(list(v["vol"])),
            "cur_count": len(v["cur"]),
            "vol_count": len(v["vol"]),
            "total_channels": len(v["cur"]) + len(v["vol"]),
        }

    record = {
        "pkl_path": pkl_path,
        "shape": list(df.shape),
        "columns": columns,
        "waveform_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "time": t_stats,
        "timestamp_validation_passed": bool(ts_valid),
        "missing_numeric": missing_numeric,
        "nonfinite_counts": finite_fail,
        "non_numeric_coercion_failures": non_numeric,
        "current_channels": cur_cols,
        "voltage_channels": vol_cols,
        "current_range": current_range,
        "voltage_range": voltage_range,
        "parsed_columns": parsed,
        "locations": {loc: {"cur": sorted(list(v["cur"])), "vol": sorted(list(v["vol"]))} for loc, v in locations.items()},
        "n_locations": len(locations),
        "n_channels_per_location": n_channels_per_location,
    }

    if csv_df is not None and "sample_id" in csv_df.columns:
        stem = Path(pkl_path).stem
        m = re.match(r"(\d+)_sample", stem)
        if m:
            sample_id = int(m.group(1))
            row = csv_df.loc[csv_df["sample_id"].astype(int) == sample_id]
            if len(row) == 1:
                r = row.iloc[0]
                record["metadata"] = {
                    "sample_id": sample_id,
                    "sc_type": str(r.get("sc_type")),
                    "phase_select": str(r.get("phase_select")),
                    "fault_target": str(r.get("fault_target")),
                    "sc_location": str(r.get("sc_location")),
                    "fault_resistance": str(r.get("fault_resistance")),
                    "t_evnt_start": str(r.get("t_evnt_start")),
                    "t_evnt_end": str(r.get("t_evnt_end")),
                }
            else:
                record["metadata"] = {"sample_id": sample_id, "note": "no matching metadata row found"}
        else:
            record["metadata"] = {"note": "could not parse sample_id from filename"}

    record["waveform_validation_passed"] = bool(ts_valid and waveform_finite_valid)
    return record


def inspect_pkl_dir(pkl_dir: str, csv_df: pd.DataFrame = None, limit: int = None) -> list:
    files = discover_pkl_files(pkl_dir)
    if limit is not None:
        files = files[:limit]
    results = []
    for fn in files:
        try:
            results.append(inspect_pkl(str(fn), csv_df=csv_df))
        except Exception as e:
            results.append({"pkl_path": str(fn), "error": str(e)})
    return results


def summarize_metadata(csv_path: str):
    df = load_csv(csv_path)
    print("CSV path:", csv_path)
    print("Shape:", df.shape)
    print("Columns:")
    for col in df.columns:
        print("  -", col)

    parsed = inspect_csv(df)
    print("\nParsed column types:")
    for col in df.columns:
        info = parsed["parsed_as"]
        print(f"  {col}: storage={df[col].dtype} interpreted_as={info.get(col)}")

    print("\nMissing values per column:")
    for col in df.columns:
        print(f"  {col}: {int(df[col].isna().sum())}")
    print("\nEmpty string counts per column:")
    for col in df.columns:
        print(f"  {col}: {int(df[col].astype(str).str.strip().eq('').sum())}")

    print("\nDuplicate sample_id rows:", parsed["duplicated_sample_id"])
    print("Unique sample_id count:", parsed["unique_sample_id"])

    if "sc_type" in df.columns:
        counts = df["sc_type"].value_counts().to_dict()
        print("\nsc_type counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    if "phase_select" in df.columns:
        counts = df["phase_select"].value_counts().to_dict()
        print("\nphase_select counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    if "fault_target" in df.columns:
        counts = df["fault_target"].value_counts().to_dict()
        print("\nfault_target counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    if "t_evnt_start" in df.columns and "t_evnt_end" in df.columns:
        start = pd.to_numeric(df["t_evnt_start"], errors="coerce")
        end = pd.to_numeric(df["t_evnt_end"], errors="coerce")
        dur = end - start
        dt_sample = EXPECTED_INTERVAL_S
        print("\nt_evnt_start range:", float(start.min()), "to", float(start.max()))
        print("t_evnt_end range:", float(end.min()), "to", float(end.max()))
        print("t_evnt_end - t_evnt_start min/max/mean:", float(dur.min()), float(dur.max()), float(dur.mean()))
        print("All events within 0..1 s:", bool((start >= 0).all() and (end <= 1.0).all()))
        print("Any t_evnt_start > t_evnt_end:", bool((dur < 0).any()))
        print("Episodes with duration < 1 sampling interval (%.6f s): %d (%.4f%%)" % (dt_sample, int((dur < dt_sample).sum()), 100 * (dur < dt_sample).sum() / len(df)))
        print("Episodes with duration < 2 sampling intervals (%.6f s): %d (%.4f%%)" % (2*dt_sample, int((dur < 2*dt_sample).sum()), 100 * (dur < 2*dt_sample).sum() / len(df)))


def main():
    parser = argparse.ArgumentParser(description="Inspect PROTECT-90 dataset artifacts")
    parser.add_argument("--csv", help="Path to hv_double_line_90kv_labels.csv")
    parser.add_argument("--zip", help="Path to hv_double_line_90kv_preprocessed_data.zip")
    parser.add_argument("--testzip", action="store_true", help="Run zipfile.testzip() during ZIP inventory (may read the whole archive)")
    parser.add_argument("--pkl-dir", help="Directory containing *_sample.pkl or *_sample_hv_double_line_90kv.pkl files")
    parser.add_argument("--pkl", help="Single waveform PKL file to inspect")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PKL files inspected in pkl-dir mode")
    parser.add_argument("--json-out", default=None, help="Optional path to write a JSON audit summary")
    args = parser.parse_args()

    if not any([args.csv, args.zip, args.pkl, args.pkl_dir]):
        parser.print_help()
        sys.exit(0)

    csv_df = None
    if args.csv:
        csv_df = load_csv(args.csv)

    summary = {}

    if args.csv:
        print("\n=== CSV inspection ===")
        inspect_csv_result = inspect_csv(csv_df)
        summary["csv"] = inspect_csv_result
        summarize_metadata(args.csv)

    if args.zip:
        print("\n=== ZIP inspection ===")
        try:
            if args.testzip:
                inv = zip_inventory_with_testzip(args.zip)
                print("NOTE: --testzip was used. zipfile.testzip() may read/compress the entire archive.")
            else:
                inv = zip_inventory(args.zip)
                print("NOTE: --testzip was not used. No integrity decompression was performed.")
            summary["zip"] = inv
            print("Archive:", inv["archive_path"])
            print("Member count:", inv["member_count"])
            print("Zip testzip result:", inv["testzip"])
            print("Total uncompressed bytes:", inv["total_uncompressed_bytes"])
            print("Total compressed bytes:", inv["total_compressed_bytes"])
            print("First 20 members:")
            for m in inv["members"][:20]:
                print("  ", m["name"], "uncompressed:", m["file_size"], "compressed:", m["compress_size"], "type:", m["compress_type"])
            if len(inv["members"]) > 20:
                print(f"  ... and {len(inv['members']) - 20} more members")
            pkl_members = [m for m in inv["members"] if m["name"].endswith(".pkl")]
            print("\nPKL members:", len(pkl_members))
            if pkl_members:
                print("First 5 PKL members:")
                for m in pkl_members[:5]:
                    print("  ", m["name"], "uncompressed:", m["file_size"], "compressed:", m["compress_size"])
        except Exception as e:
            print("ZIP inspection failed:", e)
            summary["zip_error"] = str(e)

    if args.pkl:
        print("\n=== Single PKL inspection ===")
        try:
            rec = inspect_pkl(args.pkl, csv_df=csv_df)
            summary["pkl_samples"] = [rec]
            print_pkl_summary(rec)
        except Exception as e:
            print("PKL inspection failed:", e)
            summary["pkl_error"] = str(e)

    if args.pkl_dir:
        print("\n=== PKL directory inspection ===")
        try:
            recs = inspect_pkl_dir(args.pkl_dir, csv_df=csv_df, limit=args.limit)
            summary["pkl_samples"] = recs
            for rec in recs:
                if "error" in rec:
                    print("FAILED:", rec["pkl_path"], rec["error"])
                    continue
                print_pkl_summary(rec)
        except Exception as e:
            print("PKL directory inspection failed:", e)
            summary["pkl_dir_error"] = str(e)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("\nWrote JSON summary to:", args.json_out)


def print_pkl_summary(rec: dict):
    print("PKL:", rec["pkl_path"])
    print("Shape:", rec["shape"])
    print("Time column:", "time_s" if "time_s" in rec["columns"] else "MISSING")
    print("Time stats:")
    for k, v in rec["time"].items():
        print(f"  {k}: {v}")
    print("Timestamp validation passed:", rec["timestamp_validation_passed"])
    print("Waveform validation passed:", rec.get("waveform_validation_passed"))
    if rec.get("missing_numeric"):
        print("Missing numeric values (nonzero only):", {k: v for k, v in rec["missing_numeric"].items() if v})
    if rec["nonfinite_counts"]:
        print("Nonfinite values (nonzero only):", {k: v for k, v in rec["nonfinite_counts"].items() if v})
    if rec["non_numeric_coercion_failures"]:
        print("Non-numeric coercion failures (nonzero only):", {k: v for k, v in rec["non_numeric_coercion_failures"].items() if v})
    print("Number of locations:", rec["n_locations"])
    print("Channels per location:")
    for loc, counts in rec["n_channels_per_location"].items():
        print(f"  {loc}: cur={counts['cur_count']} {counts['cur']}, vol={counts['vol_count']} {counts['vol']}")
    print("Current range:", rec["current_range"])
    print("Voltage range:", rec["voltage_range"])
    if "metadata" in rec:
        print("Metadata match:", rec["metadata"])


if __name__ == "__main__":
    main()
