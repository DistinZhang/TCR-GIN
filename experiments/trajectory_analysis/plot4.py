#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/trajectory_analysis/plot4.py

Generate the stacked-bar summary figure for trajectory-property evaluation.

This script visualizes three cumulative metrics across algorithms:
1. MAE  : monotonicity_accuracy_mae_mean
2. M    : M_freq_mean
3. S    : S_freq_mean

For each metric, the script:
- groups results by algorithm and dataset,
- stacks contributions from multiple datasets,
- sorts algorithms by cumulative value,
- highlights TCR-GIN in the x-axis labels,
- exports the figure in PDF / SVG / PNG formats.

Input
-----
The script supports two input modes:

1. External file input
   Provide a CSV / TSV / TXT table through `--input_file`.

2. Embedded demo data
   If no input file is provided, the script uses the built-in example table
   bundled in this file. This keeps the script directly runnable for testing
   and repository demonstration purposes.

Expected columns
----------------
The input table should contain at least:
- test_dataset
- algorithm
- M_freq_mean
- S_freq_mean
- monotonicity_accuracy_mae_mean

Optional columns are ignored.

Output
------
The script saves:
- <output_name>.pdf
- <output_name>.svg
- <output_name>.png

into the specified output directory.

Usage
-----
Use embedded data:
    python experiments/trajectory_analysis/plot4.py

Use an external file:
    python experiments/trajectory_analysis/plot4.py \
        --input_file experiments/trajectory_analysis/results/summary.csv

Custom output:
    python experiments/trajectory_analysis/plot4.py \
        --input_file path/to/summary.csv \
        --output_dir experiments/trajectory_analysis/results \
        --output_name plot4

Notes
-----
- Fonts are configured for editable vector output in Illustrator-compatible
  workflows by setting pdf.fonttype = 42 and ps.fonttype = 42.
- The layout is optimized for narrow publication-style figures.
- Dataset display names and colors are controlled centrally in the config
  section below.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

# =============================================================================
# Section 0. Embedded demo data
# =============================================================================

EMBEDDED_DATA = """test_dataset	algorithm	M_freq_mean	M_freq_mean_std	S_freq_mean	S_freq_mean_std	monotonicity_accuracy_mae_mean	monotonicity_accuracy_mae_mean_std
BA100	TCR-GIN	0	0	0.006554	0.010205	0.019345	0.001555
BA100	CI1	0.00128		0.00775		0.015894	
BA100	CI2	0.196412		0.262477		0.072626	
BA100	CI3	0.27244		0.380696		0.179694	
BA100	CoreGDM	0.003089		0.006893		0.005085	
BA100	CoreHD	0.004749		0.023334		0.014079	
BA100	DomiRank	0.100728		0.175515		0.082798	
BA100	EGND	0.140721		0.207187		0.066702	
BA100	EIs1	0.031827		0.04893		0.02441	
BA100	EIs2	0.000392		0.202464		0.495009	
BA100	FINDER	0.014524		0.085274		0.035316	
BA100	GDM	0.000961		0.003073		0.013666	
BA100	GDMR	0		0.002725		0.007771	
BA100	GND	0.133668		0.212172		0.073532	
BA100	GNDR	0.086668		0.165022		0.043443	
BA100	MS	0.016129		0.069504		0.024016	
BA100	MSR	0.011047		0.058104		0.033865	
BA100	BCR	0.041539		0.044683		0.013918	
BA100	BC	0.148511		0.271661		0.156025	
BA100	DCR	0.010229		0.03025		0.057856	
BA100	DC	0.108653		0.21233		0.124878	
BA100	NEL	0.171388		0.285889		0.202632	
BA100	NELR	0.029278		0.101593		0.064661	
BA100	NEM	0.156501		0.283119		0.152662	
BA100	NEMR	0.032815		0.092599		0.055832	
BA100	NES	0.141727		0.332188		0.157133	
BA100	NESR	0.032406		0.089447		0.051942	
BA100	VE	0.14553		0.227885		0.172912	
BA100	VER	0.010293		0.083103		0.060071	
LFR100	TCR-GIN	0	0	0.005398	0.001304	0.013577	0.000506
LFR100	CI1	0.011235		0.04742		0.015774	
LFR100	CI2	0.211811		0.313642		0.098948	
LFR100	CI3	0.255104		0.359605		0.145118	
LFR100	CoreGDM	0.002376		0.010573		0.006683	
LFR100	CoreHD	0.024695		0.076655		0.011517	
LFR100	DomiRank	0.080529		0.198764		0.102238	
LFR100	EGND	0.087864		0.220094		0.087703	
LFR100	EIs1	0.077286		0.119924		0.069522	
LFR100	EIs2	0		0.14342		0.37875	
LFR100	FINDER	0.014755		0.082167		0.037675	
LFR100	GDM	0.001205		0.023165		0.026054	
LFR100	GDMR	0.001714		0.024793		0.009877	
LFR100	GND	0.091012		0.215532		0.089912	
LFR100	GNDR	0.101403		0.181006		0.057759	
LFR100	MS	0.029511		0.078678		0.026509	
LFR100	MSR	0.020017		0.057046		0.02629	
LFR100	BCR	0.018129		0.045134		0.014512	
LFR100	BC	0.109417		0.25041		0.176283	
LFR100	DCR	0.0132		0.039976		0.051434	
LFR100	DC	0.087501		0.265952		0.180016	
LFR100	NEL	0.135113		0.304105		0.246836	
LFR100	NELR	0.031328		0.099738		0.047199	
LFR100	NEM	0.128545		0.303441		0.197059	
LFR100	NEMR	0.026498		0.078989		0.042571	
LFR100	NES	0.078243		0.292097		0.216133	
LFR100	NESR	0.019686		0.075653		0.044336	
LFR100	VE	0.094166		0.23675		0.214528	
LFR100	VER	0.026267		0.084497		0.038364	
ER2000	TCR-GIN	0	0	0.020397	0.013351	0.015223	0.003268
ER2000	CI1	0.1097		0.157138		0.041116	
ER2000	CI2	0.121677		0.308737		0.158154	
ER2000	CI3	0.1035		0.313274		0.192055	
ER2000	CoreGDM	0		0.006692		0.003921	
ER2000	CoreHD	0.014568		0.016375		0.005562	
ER2000	DomiRank	0.072657		0.275155		0.136632	
ER2000	EGND	0.000847		0.077362		0.051154	
ER2000	EIs1	0.02087		0.261028		0.143274	
ER2000	EIs2	0		0.408987		0.454723	
ER2000	FINDER	0		0		0.009894	
ER2000	GDM	0		0.001888		0.017484	
ER2000	GDMR	0		0		0.007162	
ER2000	GND	0.109969		0.168161		0.111115	
ER2000	GNDR	0.020204		0.082096		0.042276	
ER2000	MS	0		0.002028		0.010849	
ER2000	MSR	0		0.002151		0.011913	
ER2000	BCR	0		0.00068		0.007466	
ER2000	BC	0.055688		0.341706		0.241514	
ER2000	DCR	0.000476		0.037251		0.065421	
ER2000	DC	0.063627		0.420328		0.20224	
ER2000	NEL	0.082841		0.387348		0.316888	
ER2000	NELR	0		0.007669		0.047232	
ER2000	NEM	0.093579		0.393405		0.417789	
ER2000	NEMR	0.007789		0.082212		0.056719	
ER2000	NES	0.086322		0.469297		0.276863	
ER2000	NESR	0.008094		0.023574		0.039912	
ER2000	VE	0.042868		0.368766		0.283651	
ER2000	VER	0		0.009848		0.040679	
transport	TCR-GIN	0	0	0.004686	0.002452	0.009923	0.000579
transport	CI1	0.001408		0.01719		0.014783	
transport	CI2	0.054936		0.247017		0.064659	
transport	CI3	0.069832		0.299092		0.104365	
transport	CoreGDM	0.000469		0.010489		0.007504	
transport	CoreHD	0.000939		0.022886		0.009657	
transport	DomiRank	0.003587		0.343547		0.103195	
transport	EGND	0.003079		0.08766		0.027357	
transport	EIs1	0.012347		0.150538		0.049495	
transport	EIs2	0		0.634596		0.422356	
transport	FINDER	0.000939		0.023657		0.013809	
transport	GDM	0.000595		0.026464		0.024874	
transport	GDMR	0		0.003069		0.008737	
transport	GND	0.002789		0.097622		0.039044	
transport	GNDR	0.002224		0.066525		0.025211	
transport	MS	0		0.040603		0.018271	
transport	MSR	0		0.059341		0.026233	
transport	BCR	0		0.006687		0.00981	
transport	BC	0.004343		0.439735		0.163394	
transport	DCR	0.000469		0.22653		0.067789	
transport	DC	0.001034		0.478174		0.235958	
transport	NEL	0.00561		0.498626		0.269074	
transport	NELR	0.001545		0.203112		0.056745	
transport	NEM	0.007276		0.496988		0.265377	
transport	NEMR	0.001076		0.189566		0.055548	
transport	NES	0.043038		0.347075		0.16425	
transport	NESR	0.003694		0.183141		0.033984	
transport	VE	0.011939		0.442914		0.182484	
transport	VER	0		0.192027		0.029017	
power	TCR-GIN	0.015992	0.008534	0.028314	0.01516	0.028856	0.01049
power	CI1	0.01456		0.047425		0.027466	
power	CI2	0.021225		0.193567		0.157437	
power	CI3	0.02471		0.18567		0.164799	
power	CoreGDM	0.054769		0.078506		0.035932	
power	CoreHD	0.011563		0.044094		0.026984	
power	DomiRank	0		0.028537		0.06389	
power	EGND	0.001775		0.014805		0.015958	
power	EIs1	0.014054		0.081205		0.075032	
power	EIs2	0.043745		0.332898		0.34544	
power	FINDER	0.001046		0.19241		0.13487	
power	GDM	0		0		0.004576	
power	GDMR	0.003395		0.004321		0.001313	
power	GND	0.055129		0.156653		0.106679	
power	GNDR	0.038163		0.09002		0.032635	
power	MS	0.000309		0.002778		0.038579	
power	MSR	0.000309		0.002778		0.023067	
power	BCR	0.010341		0.198476		0.179151	
power	BC	0		0.241305		0.239106	
power	DCR	0.000303		0.018557		0.025106	
power	DC	0.000306		0.060241		0.071379	
power	NEL	0.003033		0.260589		0.216364	
power	NELR	0		0.077191		0.040626	
power	NEM	0.01171		0.256739		0.218445	
power	NEMR	0.002842		0.074921		0.042196	
power	NES	0.020303		0.159169		0.158443	
power	NESR	0.017455		0.087237		0.072046	
power	VE	0.005172		0.238831		0.226999	
power	VER	0.010811		0.084935		0.049235	
route	TCR-GIN	0	0	0	0	0.016098	0.003037
route	CI1	0.004839		0.076868		0.104857	
route	CI2	0.004737		0.063753		0.123242	
route	CI3	0.005345		0.070326		0.138227	
route	CoreGDM	0		0		0.000659	
route	CoreHD	0.002785		0.002785		0.002086	
route	DomiRank	0.002653		0.002653		0.027492	
route	EGND	0		0		0.002118	
route	EIs1	0.008576		0.052339		0.152208	
route	EIs2	0		0.279745		0.590933	
route	FINDER	0		0		0.003344	
route	GDM	0		0		0.002581	
route	GDMR	0		0		0.000904	
route	GND	0.015663		0.154814		0.15002	
route	GNDR	0.012005		0.071641		0.056429	
route	MS	0		0		0.005906	
route	MSR	0		0		0.004129	
route	BCR	0		0		0.001257	
route	BC	0		0.042603		0.058482	
route	DCR	0		0		0.013353	
route	DC	0.000595		0.010962		0.045102	
route	NEL	0.009354		0.078339		0.095045	
route	NELR	0.006425		0.020119		0.021319	
route	NEM	0.014073		0.079615		0.146916	
route	NEMR	0.00296		0.015934		0.033106	
route	NES	0.011278		0.102075		0.108125	
route	NESR	0.006262		0.027798		0.023491	
route	VE	0.001815		0.095065		0.071684	
route	VER	0.001815		0.001815		0.010885	
WS2000	TCR-GIN	0	0	0.002759	0.001195	0.011723	0.001967
WS2000	CI1	0.06335		0.090511		0.022629	
WS2000	CI2	0.093564		0.242011		0.118914	
WS2000	CI3	0.101968		0.212136		0.142893	
WS2000	CoreGDM	0.052407		0.125888		0.025998	
WS2000	CoreHD	0.037812		0.052275		0.015254	
WS2000	DomiRank	0.064438		0.193953		0.19959	
WS2000	EGND	0.047039		0.151318		0.113882	
WS2000	EIs1	0.096236		0.193504		0.06895	
WS2000	EIs2	0.051813		0.147354		0.149499	
WS2000	FINDER	0.012805		0.148031		0.115557	
WS2000	GDM	0		0		0.053438	
WS2000	GDMR	0.000749		0.001498		0.020476	
WS2000	GND	0.228409		0.280553		0.233129	
WS2000	GNDR	0.032329		0.120217		0.053069	
WS2000	MS	0.000375		0.000375		0.008138	
WS2000	MSR	0.000375		0.000375		0.004982	
WS2000	BCR	0.067805		0.177908		0.034849	
WS2000	BC	0.042004		0.163692		0.270037	
WS2000	DCR	0.000337		0.016064		0.068443	
WS2000	DC	0.0696		0.160263		0.277974	
WS2000	NEL	0.104037		0.222624		0.292607	
WS2000	NELR	0		0.001261		0.042638	
WS2000	NEM	0.105594		0.195883		0.316034	
WS2000	NEMR	0.002247		0.013566		0.044883	
WS2000	NES	0.077053		0.161257		0.294926	
WS2000	NESR	0		0.000835		0.041064	
WS2000	VE	0.042452		0.196838		0.261208	
WS2000	VER	0		0.001834		0.040618
"""

# =============================================================================
# Section 1. Global style
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

rcParams["axes.titlesize"] = 12
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 8
rcParams["axes.linewidth"] = 0.5

# =============================================================================
# Section 2. Config
# =============================================================================

NAME_MAPPING = {
    "CI1": r"CI $\ell-1$",
    "CI2": r"CI $\ell-2$",
    "CI3": r"CI $\ell-3$",
    "EIs1": r"EI $\sigma_1$",
    "EIs2": r"EI $\sigma_2$",
    "TCR-GIN (Ours)": "TCR-GIN",
}

METRICS_MAP = {
    "MAE": "monotonicity_accuracy_mae_mean",
    "M": "M_freq_mean",
    "S": "S_freq_mean",
}

SUBPLOT_LABELS = ["a", "b", "c"]
SUBPLOT_TITLES = ["MAE", r"$M$", r"$S$"]

LEGEND_ORDER_KEYS = ["BA100", "LFR100", "ER2000", "WS2000", "transport", "power", "route"]
LEGEND_DISPLAY_NAMES = {
    "BA100": "BA100",
    "LFR100": "LFR100",
    "ER2000": "ER2000",
    "WS2000": "WS2000",
    "transport": "Transport",
    "power": "Power",
    "route": "Route",
}

COLORS_LIST = [
    "#4DBBD5",
    "#00A087",
    "#8491B4",
    "#91D1C2",
    "#7E6148",
    "#B09C85",
    "#c93735",
]
DATASET_COLORS = {ds: COLORS_LIST[i] for i, ds in enumerate(LEGEND_ORDER_KEYS)}

# =============================================================================
# Section 3. I/O helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trajectory-analysis stacked bar figure.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input CSV/TSV/TXT file. If omitted, embedded demo data will be used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: <script_dir>/results",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="fig4",
        help="Base filename for exported figures.",
    )
    return parser.parse_args()


def load_data(input_file: str | None) -> pd.DataFrame:
    if input_file is None:
        df = pd.read_csv(io.StringIO(EMBEDDED_DATA), sep="\t")
        return df.fillna(0)

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(input_path, sep="\t")
    else:
        try:
            df = pd.read_csv(input_path)
            if "test_dataset" not in df.columns:
                df = pd.read_csv(input_path, sep="\t")
        except Exception:
            df = pd.read_csv(input_path, sep="\t")

    return df.fillna(0)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["algorithm"] = df["algorithm"].replace(NAME_MAPPING)
    return df


# =============================================================================
# Section 4. Plotting
# =============================================================================

def style_xtick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels():
        if "TCR-GIN" in label.get_text():
            label.set_fontweight("bold")
            label.set_fontsize(8)
            label.set_color("red")
        else:
            label.set_fontweight("normal")
            label.set_color("black")


def plot_metric_panel(ax: plt.Axes, df: pd.DataFrame, metric_key: str, panel_idx: int) -> None:
    col_name = METRICS_MAP[metric_key]
    valid_plot_keys = [k for k in LEGEND_ORDER_KEYS if k in df["test_dataset"].unique()]

    pivot_df = df.pivot(index="algorithm", columns="test_dataset", values=col_name).fillna(0)
    pivot_df["total"] = pivot_df[valid_plot_keys].sum(axis=1)
    pivot_df = pivot_df.sort_values("total", ascending=True)
    plot_data = pivot_df.drop(columns=["total"])

    algorithms = plot_data.index.tolist()
    x = np.arange(len(algorithms))
    bottom = np.zeros(len(plot_data))

    for ds in LEGEND_ORDER_KEYS:
        if ds in plot_data.columns:
            values = plot_data[ds].values
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=LEGEND_DISPLAY_NAMES[ds],
                color=DATASET_COLORS[ds],
                edgecolor="white",
                linewidth=0.15,
                width=0.85,
            )
            bottom += values

    ax.text(
        0.0, 1.03, SUBPLOT_LABELS[panel_idx],
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    ax.text(
        0.5, 0.97, SUBPLOT_TITLES[panel_idx],
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=90, ha="center", fontsize=8, fontweight="normal")
    style_xtick_labels(ax)

    ax.tick_params(axis="x", which="major", width=0.4, length=2)
    ax.tick_params(axis="y", which="major", width=0.4, length=2)

    ax.set_xlim(-0.5, len(algorithms) - 0.5)
    ax.set_ylabel("Cumulative Value", fontsize=10)

    if metric_key == "M":
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)


def add_legend(ax: plt.Axes) -> None:
    handles = [
        mpatches.Patch(color=DATASET_COLORS[k], label=LEGEND_DISPLAY_NAMES[k])
        for k in LEGEND_ORDER_KEYS
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        fontsize=8,
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.3,
        labelspacing=0.3,
        edgecolor="gray",
        facecolor=(1, 1, 1, 0.6),
    )


def create_figure(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(4.13, 8.0), dpi=600)

    for i, metric_key in enumerate(["MAE", "M", "S"]):
        plot_metric_panel(axes[i], df, metric_key, i)

    add_legend(axes[0])
    plt.tight_layout(h_pad=0.5)
    return fig


# =============================================================================
# Section 5. Saving
# =============================================================================

def save_figure(fig: plt.Figure, output_dir: Path, output_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / output_name

    fig.savefig(
        base_path.with_suffix(".pdf"),
        format="pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        base_path.with_suffix(".svg"),
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        base_path.with_suffix(".png"),
        format="png",
        dpi=600,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    return base_path


# =============================================================================
# Section 6. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_DIR / "results"
    )

    df = load_data(args.input_file)
    df = prepare_dataframe(df)

    fig = create_figure(df)
    base_path = save_figure(fig, output_dir, args.output_name)
    plt.close(fig)

    print(f"Plots saved to: {base_path}.[pdf/svg/png]")


if __name__ == "__main__":
    main()
