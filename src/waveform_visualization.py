#!/usr/bin/env python3
"""
waveform_visualization.py
 PROTECT-90 / VIDYUT-KAVACHA — Fault waveform visualization from ZIP archive.

Purpose
-------
Extract representative waveform files from the local ZIP archive and produce
publication-quality plots of three-phase voltage and current waveforms for
each fault class. Marks t_evnt_start (fault inception) and t_evnt_end (fault
clearing) on every plot.

Design notes
------------
* Reads the ZIP through Python's `zipfile` where possible; falls back to the
  `pkl_out/` directory for files that were previously extracted.
* Uses a stratified sample: one representative episode per (sc_type, duration
  bucket) to keep the plot set small and readable.
* All plots are written to `plots/` inside the project root.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("waveform_viz")

# ---------------------------------------------------------------------------
# Fault-class metadata
# ---------------------------------------------------------------------------

FAULT_CLASSES = {
    0: "LLL",
    1: "LL",
    2: "SLG",
    3: "LLG",
}

SC_TYPE_COLORS = {
    0: "#1f77b4",  # LLL   – blue
    1: "#ff7f0e",  # LL    – orange
    2: "#2ca02c",  # SLG   – green
    3: "#d62728",  # LLG   – red
}

MEASUREMENT_LOCATIONS = [
    "Bus_1_Line_01_02A",
    "Bus_1_Line_01_02B",
    "Bus_2_Line_01_02A",
    "Bus_2_Line_01_02B",
    "Bus_2_Line_02_03A",
    "Bus_2_Line_02_03B",
    "Bus_3_Line_02_03A",
    "Bus_3_Line_02_03B",
]

VOLTAGE_CHANNELS = [
    "Bus_1_Line_01_02A_vol_L1_V",
    "Bus_1_Line_01_02A_vol_L2_V",
    "Bus_1_Line_01_02A_vol_L3_V",
]

CURRENT_CHANNELS = [
    "Bus_1_Line_01_02A_cur_L1_A",
    "Bus_1_Line_01_02A_cur_L2_A",
    "Bus_1_Line_01_02A_cur_L3_A",
]


# ---------------------------------------------------------------------------
# ZIP / filesystem helpers
# ---------------------------------------------------------------------------

def open_zip(path: Path) -> Optional[zipfile.ZipFile]:
    """Try to open the waveform ZIP; return None if it cannot be read."""
    try:
        return zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile:
        LOG.warning("Cannot open ZIP %s (BadZipFile). Will rely on pkl_out/.",
                    path)
        return None
    except FileNotFoundError:
        LOG.warning("ZIP not found at %s. Will rely on pkl_out/.", path)
        return None


def resolve_sample_path(
    sample_id: int,
    zip_file: Optional[zipfile.ZipFile],
    pkl_dir: Optional[Path],
) -> Optional[Path]:
    """
    Return a writable local path for *sample_id*.

    The helper does **not** extract anything here; it tells the caller where
    to write an extracted pickle so later code can read from disk.
    """
    # Prefer the ZIP member name used by the archive.
    zip_names = [
        f"{sample_id}_sample_hv_double_line_90kv.pkl",
        f"{sample_id}_sample.pkl",
    ]
    if zip_file is not None:
        namelist = set(zip_file.namelist())
        for candidate in zip_names:
            if candidate in namelist:
                return Path(candidate)
            # The archive stores files under a directory prefix.
            prefixed = f"hv_double_line_90kv_preprocessed_data/{candidate}"
            if prefixed in namelist:
                return Path(prefixed)
    if pkl_dir is not None:
        for candidate in zip_names:
            candidate_path = pkl_dir / candidate
            if candidate_path.is_file():
                return candidate_path
    return None


def load_sample(
    sample_id: int,
    zip_file: Optional[zipfile.ZipFile],
    pkl_dir: Optional[Path],
) -> Optional[pd.DataFrame]:
    """Load one episode DataFrame from ZIP or pkl_out/."""
    # 1) Try the ZIP first.
    if zip_file is not None:
        for name in [
            f"{sample_id}_sample_hv_double_line_90kv.pkl",
            f"{sample_id}_sample.pkl",
        ]:
            try:
                with zip_file.open(name) as fh:
                    return pickle.load(fh)
            except KeyError:
                continue
            except Exception as exc:  # noqa: BLE001
                LOG.debug("ZIP read failed for %s (%s)", name, exc)
        # The archive stores files under a directory prefix, retry with that.
        for name in [
            f"hv_double_line_90kv_preprocessed_data/"
            f"{sample_id}_sample_hv_double_line_90kv.pkl",
            f"hv_double_line_90kv_preprocessed_data/{sample_id}_sample.pkl",
        ]:
            try:
                with zip_file.open(name) as fh:
                    return pickle.load(fh)
            except KeyError:
                continue
            except Exception as exc:  # noqa: BLE001
                LOG.debug("ZIP read failed for %s (%s)", name, exc)

    # 2) Fall back to the local pkl_out/ directory.
    if pkl_dir is not None:
        for name in [
            f"{sample_id}_sample.pkl",
            f"{sample_id}_sample_hv_double_line_90kv.pkl",
        ]:
            candidate = pkl_dir / name
            if candidate.is_file():
                with open(candidate, "rb") as fh:
                    return pickle.load(fh)

    return None


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def pick_representative_samples(
    metadata: pd.DataFrame,
    *,
    samples_per_class: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select a small stratified set of episodes.

    One episode per fault class is chosen: the episode whose fault duration is
    closest to the class median. This gives a “typical” waveform per class
    without favouring extreme short/long durations.
    """
    rng = np.random.default_rng(seed)
    picks: List[pd.Series] = []

    for st, name in FAULT_CLASSES.items():
        subset = metadata[metadata["sc_type"] == st].copy()
        if subset.empty:
            LOG.warning("No metadata rows for sc_type=%s (%s)", st, name)
            continue

        duration_samples = (subset["t_evnt_end"] - subset["t_evnt_start"]) * 6400
        median_duration = duration_samples.median()
        idx = (duration_samples - median_duration).abs().idxmin()
        picks.append(subset.loc[idx])

    return pd.DataFrame(picks)


# ---------------------------------------------------------------------------
# Channel helpers
# ---------------------------------------------------------------------------

def channels_for_location(
    df: pd.DataFrame,
    location: str,
) -> Dict[str, List[str]]:
    """Return { 'vol': [...], 'cur': [...] } for *location*."""
    vol: List[str] = []
    cur: List[str] = []
    for col in df.columns:
        if not col.startswith(location):
            continue
        if "_vol_" in col:
            vol.append(col)
        elif "_cur_" in col:
            cur.append(col)
    vol.sort()
    cur.sort()
    return {"vol": vol, "cur": cur}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_three_phase(
    ax: plt.Axes,
    df: pd.DataFrame,
    channels: List[str],
    label: str,
    color: Optional[str] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    *,
    ylabel: str = "",
    plot_vline: bool = True,
) -> None:
    """Plot three synchronized channels on *ax*."""
    time = df["time_s"].values
    for i, ch in enumerate(channels):
        y = df[ch].values
        ls = "-" if i == 0 else "--"
        lw = 1.1 if i == 0 else 0.9
        alpha = 1.0 if i == 0 else 0.85
        ax.plot(
            time,
            y,
            ls,
            linewidth=lw,
            color=color,
            alpha=alpha,
            label=f"{label} / {ch.split('_')[-2]}" if i == 0 else None,
        )

    if plot_vline and t_start is not None:
        ax.axvline(t_start, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
    if plot_vline and t_end is not None:
        ax.axvline(t_end, color="k", linestyle="--", linewidth=1.0, alpha=0.7)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def annotate_fault_bounds(
    ax: plt.Axes,
    t_start: float,
    t_end: float,
    y_pos: float,
    text_offset: float = 0.0,
) -> None:
    """Add a compact fault-window annotation to *ax*."""
    ax.annotate(
        "fault",
        xy=(t_start, y_pos),
        xytext=(t_start + text_offset, y_pos),
        fontsize=7,
        color="k",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "clear",
        xy=(t_end, y_pos),
        xytext=(t_end + text_offset, y_pos),
        fontsize=7,
        color="k",
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Per-episode figure
# ---------------------------------------------------------------------------

def make_episode_figure(
    sample_id: int,
    df: pd.DataFrame,
    metadata_row: pd.Series,
    location: str,
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """Create a 2-panel figure (V and I) for one location and one episode."""
    row = metadata_row
    t_start = float(row["t_evnt_start"])
    t_end = float(row["t_evnt_end"])
    sc_type = int(row["sc_type"])
    fault_name = FAULT_CLASSES[sc_type]
    color = SC_TYPE_COLORS[sc_type]

    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )

    ch = channels_for_location(df, location)
    plot_three_phase(
        ax_v,
        df,
        ch["vol"],
        label=f"{fault_name} V",
        color=color,
        t_start=t_start,
        t_end=t_end,
        ylabel="Voltage [V]",
    )
    plot_three_phase(
        ax_i,
        df,
        ch["cur"],
        label=f"{fault_name} I",
        color=color,
        t_start=t_start,
        t_end=t_end,
        ylabel="Current [A]",
    )

    ax_i.set_xlabel("Time [s]", fontsize=10)
    fig.suptitle(
        f"Sample {sample_id} · {fault_name} · "
        f"faulted line {row['fault_target']} · "
        f"{location}",
        fontsize=11,
    )

    # Legend for the fault window
    ax_i.plot([], [], "k:", label="t_evnt_start (inception)")
    ax_i.plot([], [], "k--", label="t_evnt_end (clearing)")
    ax_i.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        handlelength=2.0,
    )

    out_path = out_dir / f"waveform_sample{sample_id:04d}_{location.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        fig.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Location comparison figure
# ---------------------------------------------------------------------------

def make_location_comparison_figure(
    sample_id: int,
    df: pd.DataFrame,
    metadata_row: pd.Series,
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """
    One figure comparing all eight measurement locations for a single episode.

    Two columns: voltage on the left, current on the right. Each column has
    eight small subplots (one per location).
    """
    row = metadata_row
    t_start = float(row["t_evnt_start"])
    t_end = float(row["t_evnt_end"])
    sc_type = int(row["sc_type"])
    fault_name = FAULT_CLASSES[sc_type]

    fig, axes = plt.subplots(
        8, 2, figsize=(14, 16), sharex=True, constrained_layout=True
    )

    for row_idx, location in enumerate(MEASUREMENT_LOCATIONS):
        ch = channels_for_location(df, location)
        # Voltage
        ax_v = axes[row_idx, 0]
        plot_three_phase(
            ax_v, df, ch["vol"], label="", t_start=t_start, t_end=t_end,
            ylabel="V [V]", plot_vline=(row_idx == 0),
        )
        ax_v.set_ylabel(location.replace("_", " ") + "\nV", fontsize=7)
        ax_v.tick_params(labelsize=7)

        # Current
        ax_i = axes[row_idx, 1]
        plot_three_phase(
            ax_i, df, ch["cur"], label="", t_start=t_start, t_end=t_end,
            ylabel="I [A]", plot_vline=(row_idx == 0),
        )
        ax_i.set_ylabel(location.replace("_", " ") + "\nI", fontsize=7)
        ax_i.tick_params(labelsize=7)

    axes[0, 0].set_title(f"{fault_name} — Voltage at 8 locations", fontsize=10)
    axes[0, 1].set_title(f"{fault_name} — Current at 8 locations", fontsize=10)
    axes[-1, 0].set_xlabel("Time [s]", fontsize=9)
    axes[-1, 1].set_xlabel("Time [s]", fontsize=9)

    # Shared legend
    ax_legend = axes[0, 0]
    ax_legend.plot([], [], "k:", label="inception")
    ax_legend.plot([], [], "k--", label="clearing")
    ax_legend.legend(
        loc="upper right", fontsize=8, framealpha=0.9,
    )

    out_path = out_dir / f"waveform_locations_sample{sample_id:04d}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        fig.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize representative PROTECT-90 fault waveforms."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("hv_double_line_90kv_labels.csv"),
        help="Path to metadata CSV (default: hv_double_line_90kv_labels.csv).",
    )
    p.add_argument(
        "--zip",
        type=Path,
        default=Path("hv_double_line_90kv_preprocessed_data.zip"),
        help="Path to waveform ZIP (default: hv_double_line_90kv_preprocessed_data.zip).",
    )
    p.add_argument(
        "--pkl-dir",
        type=Path,
        default=Path("pkl_out"),
        help="Directory with pre-extracted pickle files (default: pkl_out).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated figures (default: plots/).",
    )
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Number of representative episodes per class (default: 1).",
    )
    p.add_argument(
        "--locations",
        nargs="+",
        default=["Bus_1_Line_01_02A"],
        help="Measurement locations to plot per episode (default: Bus_1_Line_01_02A).",
    )
    p.add_argument(
        "--compare-locations",
        action="store_true",
        help="Also produce an 8-location comparison figure for each episode.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for any stochastic selection (default: 42).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(
            logging.DEBUG if args.verbose >= 2
            else logging.INFO if args.verbose >= 1
            else logging.WARNING
        ),
        format="%(levelname)s: %(message)s",
    )

    csv_path = args.csv
    zip_path = args.zip
    pkl_dir = args.pkl_dir
    out_dir = args.output_dir
    locations = args.locations
    do_compare = args.compare_locations
    show = args.show
    seed = args.seed

    if not csv_path.is_file():
        LOG.error("Metadata CSV not found: %s", csv_path)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(csv_path)
    LOG.info("Loaded metadata: %d rows, %d columns", len(metadata), len(metadata.columns))

    zip_file = open_zip(zip_path) if zip_path.is_file() else None
    if zip_file is not None:
        LOG.info("ZIP opened: %s (%d members)", zip_path, len(zip_file.namelist()))
    else:
        LOG.info("Will rely on pkl_out/ for waveform files.")

    samples = pick_representative_samples(
        metadata,
        samples_per_class=args.samples_per_class,
        seed=seed,
    )
    LOG.info("Selected %d representative episodes", len(samples))

    generated = 0
    missing = 0

    for _, row in samples.iterrows():
        sample_id = int(row["sample_id"])
        df = load_sample(sample_id, zip_file, pkl_dir)

        if df is None:
            LOG.warning(
                "Cannot load waveform for sample_id=%d (class %s). "
                "Skipping plots for this episode.",
                sample_id,
                FAULT_CLASSES.get(int(row["sc_type"]), "?"),
            )
            missing += 1
            continue

        # Basic sanity checks on the waveform
        if df.shape != (6400, 49):
            LOG.warning(
                "Sample %d has unexpected shape %s; expected (6400, 49).",
                sample_id,
                df.shape,
            )
        if not np.isfinite(df.select_dtypes(include=np.number).values).all():
            LOG.warning("Sample %d contains NaN/Inf values.", sample_id)

        for location in locations:
            if location not in MEASUREMENT_LOCATIONS:
                LOG.warning("Unknown location '%s'; skipping.", location)
                continue
            try:
                make_episode_figure(
                    sample_id,
                    df,
                    row,
                    location,
                    out_dir,
                    show=show,
                )
                generated += 1
            except Exception as exc:  # noqa: BLE001
                LOG.error("Failed to plot sample %d at %s: %s",
                          sample_id, location, exc)

        if do_compare:
            try:
                make_location_comparison_figure(
                    sample_id,
                    df,
                    row,
                    out_dir,
                    show=show,
                )
                generated += 1
            except Exception as exc:  # noqa: BLE001
                LOG.error("Failed to plot location comparison for sample %d: %s",
                          sample_id, exc)

    LOG.info(
        "Done. Generated %d figures; %d episodes could not be loaded.",
        generated,
        missing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
