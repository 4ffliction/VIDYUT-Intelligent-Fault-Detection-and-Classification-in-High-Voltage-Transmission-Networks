#!/usr/bin/env python3
"""
plot_fault_waveforms.py
 PROTECT-90 / VIDYUT-KAVACHA — Focused fault waveform plots for the report.

Purpose
-------
Generate the specific figures referenced in `preprocessing_investigation.md`:

* Per-class representative three-phase voltage and current waveforms.
* Pre-fault / fault-active / post-fault behaviour.
* Comparison of signal behaviour across the eight measurement locations.
* Short-duration event waveforms.

All plots are produced from the actual dataset (ZIP or pkl_out/), never
fabricated.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waveform_visualization import (
    FAULT_CLASSES,
    MEASUREMENT_LOCATIONS,
    SC_TYPE_COLORS,
    channels_for_location,
    load_sample,
    open_zip,
    pick_representative_samples,
)

LOG = logging.getLogger("plot_fault_waveforms")

PHASE_LABELS = ["L1", "L2", "L3"]
PHASE_COLORS_VOLTAGE = {
    "L1": "#1f77b4",
    "L2": "#ff7f0e",
    "L3": "#2ca02c",
}
PHASE_COLORS_CURRENT = {
    "L1": "#d62728",
    "L2": "#9467bd",
    "L3": "#8c564b",
}


def _phase_from_channel(channel: str) -> str:
    """Extract phase label (L1/L2/L3) from a channel name."""
    parts = channel.split("_")
    for part in parts:
        if part in PHASE_LABELS:
            return part
    raise ValueError(f"Cannot determine phase from {channel}")


def plot_single_location_detail(
    sample_id: int,
    df: pd.DataFrame,
    metadata_row: pd.Series,
    location: str,
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """
    Detailed 4-panel plot for one location:
    top-left: voltage; top-right: current; bottom: zoomed fault window.
    """
    row = metadata_row
    t_start = float(row["t_evnt_start"])
    t_end = float(row["t_evnt_end"])
    sc_type = int(row["sc_type"])
    fault_name = FAULT_CLASSES[sc_type]
    color = SC_TYPE_COLORS[sc_type]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), constrained_layout=True,
    )
    time = df["time_s"].values

    ch = channels_for_location(df, location)

    # Voltage — full episode
    ax_v = axes[0, 0]
    legend_entries = []
    for i, col in enumerate(ch["vol"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax_v.plot(time, df[col].values, ls, color=PHASE_COLORS_VOLTAGE[phase],
                  linewidth=1.0, alpha=0.85, label=f"V {phase}")
    ax_v.axvline(t_start, color="k", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_v.axvline(t_end, color="k", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_v.set_ylabel("Voltage [V]", fontsize=9)
    ax_v.legend(loc="upper right", fontsize=8, ncol=3)
    ax_v.set_title(f"{fault_name} · {location} · Voltage", fontsize=10)

    # Current — full episode
    ax_i = axes[0, 1]
    for i, col in enumerate(ch["cur"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax_i.plot(time, df[col].values, ls, color=PHASE_COLORS_CURRENT[phase],
                  linewidth=1.0, alpha=0.85, label=f"I {phase}")
    ax_i.axvline(t_start, color="k", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_i.axvline(t_end, color="k", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_i.set_ylabel("Current [A]", fontsize=9)
    ax_i.legend(loc="upper right", fontsize=8, ncol=3)
    ax_i.set_title(f"{fault_name} · {location} · Current", fontsize=10)

    # Fault window — voltage zoom
    ax_vz = axes[1, 0]
    t_window_start = max(0, t_start - 0.05)
    t_window_end = min(1.0, t_end + 0.1)
    mask = (time >= t_window_start) & (time <= t_window_end)
    t_zoom = time[mask]
    for i, col in enumerate(ch["vol"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax_vz.plot(t_zoom, df.loc[mask, col].values, ls,
                   color=PHASE_COLORS_VOLTAGE[phase], linewidth=1.2, alpha=0.9,
                   label=f"V {phase}")
    ax_vz.axvline(t_start, color="k", linestyle=":", linewidth=1.5)
    ax_vz.axvline(t_end, color="k", linestyle="--", linewidth=1.5)
    ax_vz.set_xlabel("Time [s]", fontsize=9)
    ax_vz.set_ylabel("Voltage [V]", fontsize=9)
    ax_vz.legend(loc="upper right", fontsize=8)
    ax_vz.set_title("Fault window (voltage zoom)", fontsize=10)

    # Fault window — current zoom
    ax_iz = axes[1, 1]
    for i, col in enumerate(ch["cur"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax_iz.plot(t_zoom, df.loc[mask, col].values, ls,
                   color=PHASE_COLORS_CURRENT[phase], linewidth=1.2, alpha=0.9,
                   label=f"I {phase}")
    ax_iz.axvline(t_start, color="k", linestyle=":", linewidth=1.5)
    ax_iz.axvline(t_end, color="k", linestyle="--", linewidth=1.5)
    ax_iz.set_xlabel("Time [s]", fontsize=9)
    ax_iz.set_ylabel("Current [A]", fontsize=9)
    ax_iz.legend(loc="upper right", fontsize=8)
    ax_iz.set_title("Fault window (current zoom)", fontsize=10)

    fig.suptitle(
        f"Sample {sample_id} · {fault_name} · "
        f"faulted line {row['fault_target']} · "
        f"fault {t_start*1000:.1f}ms–{t_end*1000:.1f}ms",
        fontsize=12,
    )

    out_path = out_dir / f"detail_{sample_id:04d}_{location.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        fig.show()
    plt.close(fig)
    return out_path


def plot_short_duration_event(
    sample_id: int,
    df: pd.DataFrame,
    metadata_row: pd.Series,
    location: str,
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """Plot a short-duration event with the full episode and zoomed fault window."""
    row = metadata_row
    t_start = float(row["t_evnt_start"])
    t_end = float(row["t_evnt_end"])
    sc_type = int(row["sc_type"])
    fault_name = FAULT_CLASSES[sc_type]
    duration_s = t_end - t_start

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=False, constrained_layout=True,
    )
    time = df["time_s"].values
    ch = channels_for_location(df, location)

    # Full episode — voltage
    for i, col in enumerate(ch["vol"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax1.plot(time, df[col].values, ls, color=PHASE_COLORS_VOLTAGE[phase],
                  linewidth=0.8, alpha=0.8, label=f"V {phase}")
    ax1.axvline(t_start, color="k", linestyle=":", linewidth=1.5)
    ax1.axvline(t_end, color="k", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("Voltage [V]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(
        f"Short-duration event: sample {sample_id} · {fault_name} · "
        f"duration {duration_s*1000:.3f} ms ({duration_s*6400:.2f} samples)",
        fontsize=11,
    )

    # Zoomed fault window — current (high resolution)
    t_zoom_start = max(0, t_start - 0.02)
    t_zoom_end = min(1.0, t_end + 0.02)
    mask = (time >= t_zoom_start) & (time <= t_zoom_end)
    t_zoom = time[mask]
    for i, col in enumerate(ch["cur"]):
        phase = _phase_from_channel(col)
        ls = "-" if i == 0 else ("--" if i == 1 else "-.")
        ax2.plot(t_zoom, df.loc[mask, col].values, ls,
                  color=PHASE_COLORS_CURRENT[phase], linewidth=1.2, alpha=0.9,
                  label=f"I {phase}")
    ax2.axvline(t_start, color="k", linestyle=":", linewidth=2.0)
    ax2.axvline(t_end, color="k", linestyle="--", linewidth=2.0)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(
        f"Zoom: {t_zoom_start*1000:.1f}ms – {t_zoom_end*1000:.1f}ms "
        f"({(t_zoom_end-t_zoom_start)*6400:.0f} samples)",
        fontsize=10,
    )

    out_path = out_dir / f"short_{sample_id:04d}_{location.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        fig.show()
    plt.close(fig)
    return out_path


def plot_class_comparison(
    samples: Dict[int, pd.DataFrame],
    metadata_rows: Dict[int, pd.Series],
    location: str,
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """
    One figure comparing the same location across all four fault classes.

    4 rows (one per class) × 2 columns (V, I). Each row shows full episode
    with fault window marked.
    """
    fig, axes = plt.subplots(
        4, 2, figsize=(14, 14), sharex=True, constrained_layout=True,
    )
    time = None

    for row_idx, sc_type in enumerate([0, 1, 2, 3]):
        fault_name = FAULT_CLASSES[sc_type]
        color = SC_TYPE_COLORS[sc_type]

        if sc_type not in samples:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, f"No sample for {fault_name}",
                    transform=axes[row_idx, col_idx].transAxes,
                    ha="center", va="center", fontsize=10,
                )
            continue

        df = samples[sc_type]
        row = metadata_rows[sc_type]
        t_start = float(row["t_evnt_start"])
        t_end = float(row["t_evnt_end"])
        if time is None:
            time = df["time_s"].values

        ch = channels_for_location(df, location)

        # Voltage
        ax_v = axes[row_idx, 0]
        for i, col in enumerate(ch["vol"]):
            phase = _phase_from_channel(col)
            ls = "-" if i == 0 else ("--" if i == 1 else "-.")
            ax_v.plot(time, df[col].values, ls,
                      color=PHASE_COLORS_VOLTAGE[phase], linewidth=0.9,
                      alpha=0.85, label=f"V {phase}")
        ax_v.axvline(t_start, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_v.axvline(t_end, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax_v.set_ylabel(f"{fault_name}\nVoltage [V]", fontsize=8)
        ax_v.tick_params(labelsize=7)
        if row_idx == 0:
            ax_v.set_title("Voltage", fontsize=10)

        # Current
        ax_i = axes[row_idx, 1]
        for i, col in enumerate(ch["cur"]):
            phase = _phase_from_channel(col)
            ls = "-" if i == 0 else ("--" if i == 1 else "-.")
            ax_i.plot(time, df[col].values, ls,
                      color=PHASE_COLORS_CURRENT[phase], linewidth=0.9,
                      alpha=0.85, label=f"I {phase}")
        ax_i.axvline(t_start, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_i.axvline(t_end, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax_i.set_ylabel(f"{fault_name}\nCurrent [A]", fontsize=8)
        ax_i.tick_params(labelsize=7)
        if row_idx == 0:
            ax_i.set_title("Current", fontsize=10)

    axes[-1, 0].set_xlabel("Time [s]", fontsize=9)
    axes[-1, 1].set_xlabel("Time [s]", fontsize=9)

    # Shared legend
    ax_legend = axes[0, 0]
    for phase in PHASE_LABELS:
        ax_legend.plot([], [], color=PHASE_COLORS_VOLTAGE[phase],
                       linewidth=2, label=f"V {phase}")
    ax_legend.plot([], [], "k:", label="inception")
    ax_legend.plot([], [], "k--", label="clearing")
    ax_legend.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)

    fig.suptitle(
        f"Fault-class comparison · {location} · "
        f"one representative episode per class",
        fontsize=13,
        y=1.01,
    )

    out_path = out_dir / f"class_comparison_{location.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    plt.close(fig)
    return out_path


def plot_location_discriminability(
    samples_by_location: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """
    Compare RMS voltage and current during fault across locations for one
    representative episode to assess which locations show clearer fault signatures.
    """
    # Use first available sample
    loc = list(samples_by_location.keys())[0]
    df, row = samples_by_location[loc]
    t_start = float(row["t_evnt_start"])
    t_end = float(row["t_evnt_end"])
    fault_name = FAULT_CLASSES[int(row["sc_type"])]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), constrained_layout=True,
    )

    locations_data = []
    for location in MEASUREMENT_LOCATIONS:
        ch = channels_for_location(df, location)
        # RMS during fault window
        time = df["time_s"].values
        mask = (time >= t_start) & (time <= t_end)
        if mask.sum() < 2:
            v_rms = 0.0
            i_rms = 0.0
        else:
            v_vals = df.loc[mask, ch["vol"]].values if ch["vol"] else np.zeros(1)
            i_vals = df.loc[mask, ch["cur"]].values if ch["cur"] else np.zeros(1)
            v_rms = float(np.sqrt(np.mean(v_vals**2)))
            i_rms = float(np.sqrt(np.mean(i_vals**2)))
        locations_data.append((location, v_rms, i_rms, len(ch["vol"]), len(ch["cur"])))

    # Voltage RMS by location
    locs = [d[0] for d in locations_data]
    v_rms = [d[1] for d in locations_data]
    colors = [SC_TYPE_COLORS[int(row["sc_type"])]] * len(locs)
    bars1 = ax1.barh(range(len(locs)), v_rms, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(locs)))
    ax1.set_yticklabels([l.replace("_", "\n") for l in locs], fontsize=7)
    ax1.set_xlabel("RMS Voltage during fault [V]")
    ax1.set_title(f"{fault_name} · Voltage RMS at 8 locations", fontsize=10)
    ax1.axvline(np.mean(v_rms), color="k", linestyle="--", alpha=0.5, label=f"mean={np.mean(v_rms):.1f}")
    ax1.legend(fontsize=8)

    # Current RMS by location
    i_rms = [d[2] for d in locations_data]
    bars2 = ax2.barh(range(len(locs)), i_rms, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(locs)))
    ax2.set_yticklabels([l.replace("_", "\n") for l in locs], fontsize=7)
    ax2.set_xlabel("RMS Current during fault [A]")
    ax2.set_title(f"{fault_name} · Current RMS at 8 locations", fontsize=10)
    ax2.axvline(np.mean(i_rms), color="k", linestyle="--", alpha=0.5, label=f"mean={np.mean(i_rms):.2f}")
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"Location discriminability · Sample {samples_by_location[loc][1]['sample_id']} · "
        f"{fault_name} · faulted line {row['fault_target']}",
        fontsize=12,
    )

    out_path = out_dir / f"location_discriminability_{loc.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate focused fault waveform plots for the report."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("hv_double_line_90kv_labels.csv"),
    )
    p.add_argument(
        "--zip",
        type=Path,
        default=Path("hv_double_line_90kv_preprocessed_data.zip"),
    )
    p.add_argument(
        "--pkl-dir",
        type=Path,
        default=Path("pkl_out"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
    )
    p.add_argument(
        "--location",
        default="Bus_1_Line_01_02A",
        help="Primary measurement location for detailed plots.",
    )
    p.add_argument(
        "--include-short",
        action="store_true",
        help="Also plot the shortest-duration event per class.",
    )
    p.add_argument(
        "--compare-classes",
        action="store_true",
        help="Produce a 4-class comparison plot for the chosen location.",
    )
    p.add_argument(
        "--compare-locations",
        action="store_true",
        help="Produce location discriminability analysis.",
    )
    p.add_argument(
        "--show",
        action="store_true",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
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
    location = args.location
    include_short = args.include_short
    do_compare_classes = args.compare_classes
    do_compare_locations = args.compare_locations
    show = args.show
    seed = args.seed

    if not csv_path.is_file():
        LOG.error("Metadata CSV not found: %s", csv_path)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(csv_path)
    zip_file = open_zip(zip_path) if zip_path.is_file() else None

    # Representative samples (one per class, median duration)
    reps = pick_representative_samples(metadata, samples_per_class=1, seed=seed)

    samples: Dict[int, pd.DataFrame] = {}
    metadata_rows: Dict[int, pd.Series] = {}

    for _, row in reps.iterrows():
        sample_id = int(row["sample_id"])
        df = load_sample(sample_id, zip_file, pkl_dir)
        if df is None:
            LOG.warning("Cannot load sample %d; skipping.", sample_id)
            continue
        samples[int(row["sc_type"])] = df
        metadata_rows[int(row["sc_type"])] = row

    generated = 0

    # 1. Detailed plots per class at selected location
    for sc_type, df in samples.items():
        row = metadata_rows[sc_type]
        try:
            plot_single_location_detail(
                int(row["sample_id"]), df, row, location, out_dir, show=show,
            )
            generated += 1
        except Exception as exc:  # noqa: BLE001
            LOG.error("Failed detail plot for class %s: %s", sc_type, exc)

    # 2. Class comparison
    if do_compare_classes and len(samples) >= 2:
        try:
            plot_class_comparison(samples, metadata_rows, location, out_dir, show=show)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            LOG.error("Failed class comparison plot: %s", exc)

    # 3. Short duration events
    if include_short:
        short_samples: Dict[int, Tuple[pd.DataFrame, pd.Series]] = {}
        for sc_type in [0, 1, 2, 3]:
            subset = metadata[metadata["sc_type"] == sc_type]
            if subset.empty:
                continue
            # Find shortest duration event that we can load
            durations = (subset["t_evnt_end"] - subset["t_evnt_start"]) * 6400
            shortest_idx = durations.idxmin()
            row = subset.loc[shortest_idx]
            sample_id = int(row["sample_id"])
            df = load_sample(sample_id, zip_file, pkl_dir)
            if df is not None:
                short_samples[sc_type] = (df, row)
                try:
                    plot_short_duration_event(
                        sample_id, df, row, location, out_dir, show=show,
                    )
                    generated += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.error("Failed short-duration plot for class %s: %s",
                              sc_type, exc)

    # 4. Location discriminability
    if do_compare_locations and samples:
        samples_by_location: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
        # Use first available class's sample
        first_sc_type = list(samples.keys())[0]
        df = samples[first_sc_type]
        row = metadata_rows[first_sc_type]
        for loc in MEASUREMENT_LOCATIONS:
            samples_by_location[loc] = (df, row)
        try:
            plot_location_discriminability(samples_by_location, out_dir, show=show)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            LOG.error("Failed location discriminability plot: %s", exc)

    LOG.info("Done. Generated %d figures in %s.", generated, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
