"""Matplotlib figures for evaluation and EDA. One hue per job, thin marks, no dual axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d7d6d1")
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def plot_risk_coverage(rc: dict, path: Path, label: str = "PillGuard") -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.plot(rc["coverage"], rc["risk"], color=BLUE, linewidth=2, label=label)
    ax.axhline(rc["full_coverage_risk"], color=GRAY, linewidth=1.2, linestyle="--", label="no reject")
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    _style(ax, f"Risk vs coverage (AURC {rc['aurc']:.3f})", "coverage (share of pills decided)", "risk (error rate on decided)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_reliability(bins: list[dict], path: Path, ece: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"]]
    accs = [b["acc"] for b in bins if b["n"]]
    width = (bins[0]["hi"] - bins[0]["lo"]) * 0.9 if bins else 0.06
    ax.bar(xs, accs, width=width, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    t = "Reliability of p(in prescription)" + (f"  ECE {ece:.3f}" if ece is not None else "")
    _style(ax, t, "predicted probability", "observed share in prescription")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_hist(values, path: Path, title: str, xlabel: str, bins=30, color: str = BLUE, log: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.6)
    if log:
        ax.set_yscale("log")
    _style(ax, title, xlabel, "count")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_sorted_bars(values: list[float], path: Path, title: str, xlabel: str, ylabel: str,
                     highlight_idx: set[int] | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    order = sorted(range(len(values)), key=lambda i: -values[i])
    colors = [ORANGE if highlight_idx and i in highlight_idx else BLUE for i in order]
    ax.bar(range(len(values)), [values[i] for i in order], color=colors, width=0.85, linewidth=0)
    ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    ax.set_xticks([])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_curves(series: dict[str, tuple[list, list]], path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for (name, (x, y)), color in zip(series.items(), [BLUE, ORANGE, AQUA, GRAY]):
        ax.plot(x, y, color=color, linewidth=2, label=name)
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
