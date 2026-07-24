#!/usr/bin/env python3
"""Generate transcript_grounding_viz.jpg — three stacked horizontal bar charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_GAP  = "#6b6a67"
C_BLUE = "#2a78d6"
C_AQUA = "#1baf7a"
C_GRN  = "#008300"
C_RED  = "#e34948"
C_ORG  = "#eb6834"

BG   = "#f0f2f5"
SURF = "#ffffff"
T1   = "#111820"
T2   = "#4a5668"
T3   = "#8a97a8"
GRID = "#dde3ea"

# ── Data ──────────────────────────────────────────────────────────────────────
COV_LABELS = ["MedGemma-27B","GPT-5 Pro","HealthScribe","OpenBioLLM-8B","Med42-70B","JSL Text2SOAP","Asclepius-13B"]
COV_NI  = [0.312,0.483,0.567,0.624,0.694,0.784,0.787]
COV_AI  = [0.550,0.354,0.270,0.213,0.148,0.053,0.050]
COV_HU  = [0.026,0.034,0.039,0.125,0.106,0.119,0.140]
COV_BO  = [0.112,0.129,0.124,0.038,0.052,0.044,0.023]
COV_TOT = [1290, 1890, 1890, 1890, 1854, 1890, 1890]

HAL_LABELS = ["HealthScribe","Med42-70B","GPT-5 Pro","JSL Text2SOAP","Asclepius-13B","OpenBioLLM-8B"]
HAL_IT  = [0.886,0.837,0.799,0.523,0.421,0.103]
HAL_FA  = [0.115,0.163,0.201,0.477,0.579,0.897]
HAL_TOT = [2000, 129,  2474, 1009, 76,   78  ]

MIS_LABELS = ["GPT-5 Pro","Med42-70B","HealthScribe","JSL Text2SOAP","OpenBioLLM-8B","Asclepius-13B"]
MIS_IT  = [0.389,0.471,0.528,0.548,0.601,0.651]
MIS_NT  = [0.611,0.529,0.472,0.452,0.399,0.349]
MIS_TOT = [550,  138,  781,  946,  622,  212 ]

# ── Layout ────────────────────────────────────────────────────────────────────
BAR_H = 0.55

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 20), facecolor=BG)
fig.subplots_adjust(left=0.20, right=0.93, top=0.88, bottom=0.03, hspace=0.72)

# Page header
fig.text(0.01, 0.975,
    "AI Clinical Note Documentation — Transcript Grounding Analysis",
    fontsize=12.5, fontweight="bold", color=T1, va="bottom")
fig.text(0.01, 0.966,
    "45 rehabilitation encounters  ·  7 models  ·  1,890 atomic transcript facts  ·  LLM judge: GPT-4o",
    fontsize=8.5, color=T3, va="bottom")

# ── Helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax, title, desc, acc):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9, colors=T2)
    ax.tick_params(axis='y', pad=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.tick_params(axis="x", labelcolor=T3, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    # Accent bar on left edge
    p = mpatches.FancyBboxPatch((-0.014, -0.04), 0.007, 1.08,
        transform=ax.transAxes, boxstyle="square,pad=0",
        facecolor=acc, clip_on=False, zorder=10)
    ax.add_patch(p)
    # Title and description above the axes
    ax.text(0.0, 1.115, title,
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        color=T1, va="bottom")
    ax.text(0.0, 1.038, desc,
        transform=ax.transAxes, fontsize=8.5, color=T3, va="bottom")

def bar_label(ax, x0, x1, y, pct, dark=False):
    w = x1 - x0
    if w < 0.065:
        return
    ax.text((x0 + x1) / 2, y, f"{round(pct * 100)}%",
        ha="center", va="center", fontsize=8.5, fontweight="500",
        color=T1 if dark else "white", zorder=4)

def count_ann(ax, y, n):
    ax.text(1.01, y, f"n={n:,}", va="center", ha="left",
        fontsize=8.5, color=T3,
        transform=ax.get_yaxis_transform())

# ── Chart 1 ───────────────────────────────────────────────────────────────────
ys = np.arange(len(COV_LABELS))
ax1.barh(ys, COV_NI, BAR_H, color=C_GAP,  zorder=2)
l2 = np.array(COV_NI)
ax1.barh(ys, COV_AI, BAR_H, left=l2, color=C_BLUE, zorder=2)
l3 = l2 + np.array(COV_AI)
ax1.barh(ys, COV_HU, BAR_H, left=l3, color=C_AQUA, zorder=2)
l4 = l3 + np.array(COV_HU)
ax1.barh(ys, COV_BO, BAR_H, left=l4, color=C_GRN, zorder=2)

for i, y in enumerate(ys):
    bar_label(ax1, 0,      COV_NI[i],          y, COV_NI[i])
    bar_label(ax1, l2[i],  l2[i]+COV_AI[i],    y, COV_AI[i])
    bar_label(ax1, l3[i],  l3[i]+COV_HU[i],    y, COV_HU[i], dark=True)
    bar_label(ax1, l4[i],  l4[i]+COV_BO[i],    y, COV_BO[i])
    count_ann(ax1, y, COV_TOT[i])

ax1.set_yticks(ys); ax1.set_yticklabels(COV_LABELS)
ax1.invert_yaxis()
style_ax(ax1,
    "Transcript Fact Coverage by Model",
    "Share of ~1,890 transcript facts captured in AI note, human note, both, or neither — sorted by lowest gap first.",
    C_BLUE)
ax1.legend(handles=[
    mpatches.Patch(color=C_GAP,  label="Neither Note (cell 1.3)"),
    mpatches.Patch(color=C_BLUE, label="AI Note Only"),
    mpatches.Patch(color=C_AQUA, label="Human Note Only"),
    mpatches.Patch(color=C_GRN,  label="Both Notes"),
], bbox_to_anchor=(0, -0.13), loc="upper left", ncol=4,
   fontsize=8.5, frameon=False, labelcolor=T2)

# ── Chart 2 ───────────────────────────────────────────────────────────────────
ys2 = np.arange(len(HAL_LABELS))
ax2.barh(ys2, HAL_IT, BAR_H, color=C_BLUE, zorder=2)
l2b = np.array(HAL_IT)
ax2.barh(ys2, HAL_FA, BAR_H, left=l2b, color=C_RED, zorder=2)

for i, y in enumerate(ys2):
    bar_label(ax2, 0,      HAL_IT[i],           y, HAL_IT[i])
    bar_label(ax2, l2b[i], l2b[i]+HAL_FA[i],    y, HAL_FA[i])
    count_ann(ax2, y, HAL_TOT[i])

ax2.set_yticks(ys2); ax2.set_yticklabels(HAL_LABELS)
ax2.invert_yaxis()
style_ax(ax2,
    "AI Hallucination Analysis",
    "Of AI facts absent from the human note — grounded in the transcript (omitted but true) vs. fabricated. Sorted by lowest fabrication rate.",
    C_RED)
ax2.legend(handles=[
    mpatches.Patch(color=C_BLUE, label="In Transcript (omitted but true)"),
    mpatches.Patch(color=C_RED,  label="Fabricated (not in transcript)"),
], bbox_to_anchor=(0, -0.13), loc="upper left", ncol=2,
   fontsize=8.5, frameon=False, labelcolor=T2)

# ── Chart 3 ───────────────────────────────────────────────────────────────────
ys3 = np.arange(len(MIS_LABELS))
ax3.barh(ys3, MIS_IT, BAR_H, color=C_BLUE, zorder=2)
l3b = np.array(MIS_IT)
ax3.barh(ys3, MIS_NT, BAR_H, left=l3b, color=C_ORG, zorder=2)

for i, y in enumerate(ys3):
    bar_label(ax3, 0,      MIS_IT[i],           y, MIS_IT[i])
    bar_label(ax3, l3b[i], l3b[i]+MIS_NT[i],    y, MIS_NT[i])
    count_ann(ax3, y, MIS_TOT[i])

ax3.set_yticks(ys3); ax3.set_yticklabels(MIS_LABELS)
ax3.invert_yaxis()
style_ax(ax3,
    "Missed Fact Grounding",
    "Of facts in the human note that AI missed — recoverable from transcript vs. requiring beyond-session clinical knowledge. Highest unrecoverable share first.",
    C_ORG)
ax3.legend(handles=[
    mpatches.Patch(color=C_BLUE, label="In Transcript (1.2 — recoverable)"),
    mpatches.Patch(color=C_ORG,  label="Not in Transcript (2.2 — beyond-session)"),
], bbox_to_anchor=(0, -0.13), loc="upper left", ncol=2,
   fontsize=8.5, frameon=False, labelcolor=T2)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "metrics_test/gap_analysis/transcript_grounding_viz.jpg"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG, format="jpeg")
print(f"Saved -> {out}")
plt.close()
