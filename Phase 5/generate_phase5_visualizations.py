r"""
Phase 5: IEEE Publication-Grade Visualizations & Empirical Paper Figures
Paper: Architectural Diversity as a Defense Mechanism Against Semantic Payload Propagation in Multi-Agent LLM Pipelines
Authors: Abdullah Mohammed Khalid Hashim, Moataz Ahmed (KFUPM)

Outputs PNG, PDF, and SVG figures to Phase 5/figures/{png,pdf,svg}/.
"""
import os
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==============================================================================
# IEEE PUBLICATION TYPOGRAPHY & RC PARAMETERS
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

PHASE5_DIR = Path(__file__).resolve().parent
FIG_DIR = PHASE5_DIR / "figures"

# Academic IEEE Color Palette (Colorblind-safe & high B&W contrast)
COLOR_CTRL = "#B91C1C"   # Deep Academic Crimson (Homogeneous Control)
COLOR_LMQGG = "#D97706"  # Muted Ochre / Amber (Heterogeneous Test 1: lmqgg)
COLOR_LGMGQ = "#15803D"  # Deep Forest Green (Heterogeneous Test 2: lgmgq)

NAVY = "#1E3A8A"
SLATE_DARK = "#0F172A"
SLATE_BORDER = "#334155"
SLATE_BG = "#F8FAFC"

def save_ieee_fig(fig, filename: str):
    for sub, ext in [("png", "png"), ("pdf", "pdf"), ("svg", "svg")]:
        out_folder = FIG_DIR / sub
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"{filename}.{ext}"
        if ext == "svg":
            fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.08)
        else:
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)

    print(f"Saved IEEE figure: {filename} (.png, .pdf, .svg)")

def draw_box(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor="#1E293B", lw=1.1, radius=0.06, zorder=2):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#1E293B", lw=1.3, style="->", mutation=11, zorder=4):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=mutation, shrinkA=0, shrinkB=0),
        zorder=zorder
    )

# ==============================================================================
# FIGURE 1: HIGH-LEVEL 5-HOP PIPELINE & EVALUATED CONFIGURATIONS (IEEE STYLE)
# ==============================================================================
def plot_fig1_pipeline_schematic():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.axis('off')
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)

    # --- TOP SECTION: 5-HOP SEQUENTIAL PIPELINE ARCHITECTURE ---
    draw_box(ax, 0.2, 3.35, 13.1, 2.65, facecolor=SLATE_BG, edgecolor=SLATE_BORDER, lw=1.2, radius=0.08, zorder=1)
    ax.text(0.45, 5.72, "(a) Sequential Five-Hop Multi-Agent Pipeline Topology and Hybrid Evaluation Points",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color=NAVY)

    # Untrusted Input Node
    draw_box(ax, 0.45, 3.65, 1.85, 1.75, facecolor="#FEF2F2", edgecolor=COLOR_CTRL, lw=1.2)
    ax.text(1.375, 5.10, "Untrusted Input", ha='center', va='center', fontsize=9.2, fontweight='bold', color=COLOR_CTRL)
    ax.text(1.375, 4.50, "User Task $U$\n+ External Data\n(Embedded IPI)", ha='center', va='center', fontsize=8.3, color=SLATE_DARK)
    ax.text(1.375, 3.88, "$N = 102$ Cases", ha='center', va='center', fontsize=8.0, style='italic', color=COLOR_CTRL)

    draw_arrow(ax, 2.32, 4.52, 2.68, 4.52, color=COLOR_CTRL, lw=1.5)

    # 5 Hops
    hops = [
        ("Hop 1: Ingestor", "Summarize raw\ntool data ($O_1$)", "Judge: DeepSeek-R1", 2.70),
        ("Hop 2: Planner", "Draft sequential\naction plan ($O_2$)", "Judge: DeepSeek-R1", 4.82),
        ("Hop 3: Writer", "Map plan to tool\narguments ($O_3$)", "Judge: DeepSeek-R1", 6.94),
        ("Hop 4: Reviewer", "Audit schema &\nconstraints ($O_4$)", "Judge: DeepSeek-R1", 9.06),
        ("Hop 5: Executor", "Propose JSON\ntool calls ($E_5$)", "JSON Proposal Oracle", 11.18),
    ]
    for h_title, h_desc, h_judge, hx in hops:
        is_exec = "Executor" in h_title
        ec = COLOR_CTRL if is_exec else NAVY
        fc = "#FFF1F2" if is_exec else "#FFFFFF"
        draw_box(ax, hx, 3.65, 1.82, 1.75, facecolor=fc, edgecolor=ec, lw=1.15)
        ax.text(hx + 0.91, 5.12, h_title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=ec)
        ax.text(hx + 0.91, 4.55, h_desc, ha='center', va='center', fontsize=8.2, color=SLATE_DARK)

        # Judge sub-pill
        j_fc = "#FEE2E2" if is_exec else "#EFF6FF"
        draw_box(ax, hx + 0.12, 3.78, 1.58, 0.36, facecolor=j_fc, edgecolor=ec, lw=0.7, radius=0.04)
        ax.text(hx + 0.91, 3.96, h_judge, ha='center', va='center', fontsize=7.4, style='italic', color=SLATE_DARK)

        if hx < 11.18:
            draw_arrow(ax, hx + 1.84, 4.52, hx + 2.10, 4.52, color=SLATE_BORDER, lw=1.3)

    # --- BOTTOM SECTION: 3 EVALUATED ARCHITECTURAL CONFIGURATIONS ---
    draw_box(ax, 0.2, 0.2, 13.1, 2.95, facecolor=SLATE_BG, edgecolor=SLATE_BORDER, lw=1.2, radius=0.08, zorder=1)
    ax.text(0.45, 2.88, "(b) Evaluated Model Compositions and Sequential Permutations (102 Paired Runs per Configuration)",
            ha='left', va='center', fontsize=10.5, fontweight='bold', color=NAVY)

    rows = [
        ("1. Homogeneous Control:",
         ["Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B"],
         "ASR: 14.71% (15/102)", COLOR_CTRL, 2.22),
        ("2. Hetero. Perm. 1 (lmqgg):",
         ["Llama-3.1:8B", "Mistral-7B", "Qwen-2.5:7B", "Gemma-2:9B", "GLM-4:9B"],
         "ASR: 12.75% (p = 0.82)", COLOR_LMQGG, 1.42),
        ("3. Hetero. Perm. 2 (lgmgq):",
         ["Llama-3.1:8B", "Gemma-2:9B *", "Mistral-7B", "GLM-4:9B", "Qwen-2.5:7B"],
         "ASR: 3.92% (p = 0.007)", COLOR_LGMGQ, 0.62),
    ]

    for r_label, models, asr_str, col, ry in rows:
        ax.text(0.45, ry, r_label, ha='left', va='center', fontsize=9.0, fontweight='bold', color=col)
        start_x = 3.25
        pw = 1.32
        gap = 0.24
        for idx, m in enumerate(models):
            mx = start_x + idx * (pw + gap)
            is_star = "*" in m
            fc = "#DCFCE7" if is_star else "#FFFFFF"
            lw = 1.4 if is_star else 1.0
            draw_box(ax, mx, ry - 0.24, pw, 0.48, facecolor=fc, edgecolor=col, lw=lw, radius=0.04)
            ax.text(mx + pw / 2, ry, m, ha='center', va='center', fontsize=8.2, fontweight='bold', color=SLATE_DARK)
            if idx < 4:
                draw_arrow(ax, mx + pw + 0.02, ry, mx + pw + gap - 0.02, ry, color=col, lw=1.1, mutation=8)

        # Right Outcome Badge
        draw_box(ax, 11.05, ry - 0.24, 2.0, 0.48, facecolor=col, edgecolor=col, lw=1.0, radius=0.04)
        ax.text(12.05, ry, asr_str, ha='center', va='center', fontsize=8.3, fontweight='bold', color="#FFFFFF")

    save_ieee_fig(fig, "fig1_pipeline_schematic")
    plt.close(fig)

# ==============================================================================
# FIGURE 2: MULTI-HOP PAYLOAD SURVIVAL CURVE (IEEE STYLE)
# ==============================================================================
def plot_fig2_survival_curve():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    hops = ["Hop 1\n(Ingestor)", "Hop 2\n(Planner)", "Hop 3\n(Writer)", "Hop 4\n(Reviewer)", "Hop 5\n(Executor)"]
    x = np.arange(len(hops))

    ctrl = [50.98, 12.75, 10.78, 14.71, 14.71]
    lmqgg = [54.90, 20.59, 17.65, 18.63, 12.75]
    lgmgq = [53.92, 11.76, 8.82, 12.75, 3.92]

    ax.plot(x, ctrl, marker='s', markersize=6.5, linewidth=2.0, color=COLOR_CTRL,
            linestyle='--', label='Homogeneous Control (5× Llama-3.1) — ASR: 14.71%')
    ax.plot(x, lmqgg, marker='^', markersize=6.5, linewidth=2.0, color=COLOR_LMQGG,
            linestyle='-.', label='Heterogeneous Perm. 1 (lmqgg) — ASR: 12.75%')
    ax.plot(x, lgmgq, marker='o', markersize=7.0, linewidth=2.4, color=COLOR_LGMGQ,
            linestyle='-', label='Heterogeneous Perm. 2 (lgmgq) — ASR: 3.92% (p = 0.0074)')

    ax.fill_between(x, ctrl, lgmgq, color=COLOR_LGMGQ, alpha=0.08, label='Defense Gain (lgmgq vs. Control)')

    # Clean callout BELOW Hop 2 (crosses zero lines)
    ax.annotate("Gemma-2 Planner drop\n(11.76% survival)",
                xy=(1.0, 10.9), xytext=(1.0, 2.8),
                arrowprops=dict(arrowstyle="->", color=COLOR_LGMGQ, lw=1.3),
                ha='center', va='center', fontsize=8.5, fontweight='bold', color=COLOR_LGMGQ,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor=COLOR_LGMGQ, lw=0.9))

    # Right-hand terminal labels (well-spaced, no border clipping)
    ax.text(4.12, 15.6, "14.71%", fontsize=8.8, fontweight='bold', color=COLOR_CTRL, va='center')
    ax.text(4.12, 11.8, "12.75%", fontsize=8.8, fontweight='bold', color=COLOR_LMQGG, va='center')
    ax.text(4.12, 3.92, "3.92% (-73.3%)", fontsize=8.8, fontweight='bold', color=COLOR_LGMGQ, va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(hops)
    ax.set_ylabel("Detected Adversarial Intent / Terminal ASR (%)", fontweight='bold')
    ax.set_xlabel("Sequential Pipeline Stage", fontweight='bold', labelpad=6)
    ax.set_xlim(-0.25, 4.95)
    ax.set_ylim(0, 66)
    ax.grid(True, linestyle=':', color='#CBD5E1', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    save_ieee_fig(fig, "fig2_survival_curve")
    plt.close(fig)

# ==============================================================================
# FIGURE 3: ATTENUATION DEPTH (D_att) DISTRIBUTION (IEEE STYLE WITH HATCHING)
# ==============================================================================
def plot_fig3_datt_distribution():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    labels = ["Hop 1\n(Ingestor)", "Hop 2\n(Planner)", "Hop 3\n(Writer)", "Hop 4\n(Reviewer)", "Hop 5\n(Executor)", "Breached\n($D_{\\mathrm{att}} = 6$)"]
    x = np.arange(len(labels))
    width = 0.25

    ctrl_counts = [43, 40, 3, 0, 1, 15]
    lmqgg_counts = [41, 44, 3, 0, 1, 13]
    lgmgq_counts = [47, 45, 5, 1, 0, 4]

    r1 = ax.bar(x - width, ctrl_counts, width, label='Control (5× Llama-3.1)', color=COLOR_CTRL, alpha=0.88, edgecolor='black', lw=0.7, hatch='//')
    r2 = ax.bar(x, lmqgg_counts, width, label='Test 1 (lmqgg)', color=COLOR_LMQGG, alpha=0.88, edgecolor='black', lw=0.7, hatch='..')
    r3 = ax.bar(x + width, lgmgq_counts, width, label='Test 2 (lgmgq)', color=COLOR_LGMGQ, alpha=0.92, edgecolor='black', lw=0.7)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2.5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Clean callout pointing from the right side of the Breached green bar (zero overlap with 15 or 13)
    ax.annotate("-73.3% Breaches\n(15 vs. 4 in lgmgq)",
                xy=(5 + width + 0.08, 4.2), xytext=(4.85, 25),
                arrowprops=dict(arrowstyle="->", color=COLOR_LGMGQ, lw=1.3, connectionstyle="arc3,rad=-0.15"),
                ha='center', va='center', fontsize=8.5, fontweight='bold', color=COLOR_LGMGQ,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor=COLOR_LGMGQ, lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Tasks Attenuated ($N = 102$)", fontweight='bold')
    ax.set_xlabel("Attenuation Depth ($D_{\\mathrm{att}}$)", fontweight='bold', labelpad=6)
    ax.set_xlim(-0.55, 5.75)
    ax.set_ylim(0, 56)
    ax.grid(axis='y', linestyle=':', color='#CBD5E1', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    save_ieee_fig(fig, "fig3_datt_distribution")
    plt.close(fig)

# ==============================================================================
# FIGURE 4: RESILIENCE ACROSS ADVERSARIAL FRAMINGS (IEEE STYLE)
# ==============================================================================
def plot_fig4_framing_resilience():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    framings = ["Base Framing\n(Direct Note, $n=34$)", "Authoritative Framing\n(Policy/NDA, $n=34$)", "Urgency Framing\n(Recall/Alert, $n=34$)"]
    x = np.arange(len(framings))
    width = 0.24

    ctrl_asr = [14.71, 14.71, 14.71]
    lmqgg_asr = [14.71, 8.82, 14.71]
    lgmgq_asr = [0.00, 0.00, 11.76]

    r1 = ax.bar(x - width, ctrl_asr, width, label='Control (5× Llama-3.1)', color=COLOR_CTRL, alpha=0.88, edgecolor='black', lw=0.7, hatch='//')
    r2 = ax.bar(x, lmqgg_asr, width, label='Test 1 (lmqgg)', color=COLOR_LMQGG, alpha=0.88, edgecolor='black', lw=0.7, hatch='..')
    r3 = ax.bar(x + width, lgmgq_asr, width, label='Test 2 (lgmgq)', color=COLOR_LGMGQ, alpha=0.92, edgecolor='black', lw=0.7)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Clean 0.0% label directly on the baseline for Base and Authoritative in lgmgq
    ax.text(0 + width, 0.4, "0.0%\n(0/34)", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=COLOR_LGMGQ)
    ax.text(1 + width, 0.4, "0.0%\n(0/34)", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=COLOR_LGMGQ)

    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylabel("Attack Success Rate (%)", fontweight='bold')
    ax.set_xlabel("Adversarial Psychological Framing", fontweight='bold', labelpad=6)
    ax.set_ylim(0, 21.5)
    ax.grid(axis='y', linestyle=':', color='#CBD5E1', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    save_ieee_fig(fig, "fig4_framing_resilience")
    plt.close(fig)

# ==============================================================================
# FIGURE 5: THREAT VECTOR BREAKDOWN (DH vs. DS, IEEE STYLE)
# ==============================================================================
def plot_fig5_threat_vectors():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    threats = ["Direct Harm (DH)\nState Mutations ($n = 51$)", "Data Stealing (DS)\nExfiltration ($n = 51$)"]
    x = np.arange(len(threats))
    width = 0.22

    ctrl_threat = [25.49, 3.92]
    lmqgg_threat = [23.53, 1.96]
    lgmgq_threat = [5.88, 1.96]

    r1 = ax.bar(x - width, ctrl_threat, width, label='Control (5× Llama-3.1)', color=COLOR_CTRL, alpha=0.88, edgecolor='black', lw=0.7, hatch='//')
    r2 = ax.bar(x, lmqgg_threat, width, label='Test 1 (lmqgg)', color=COLOR_LMQGG, alpha=0.88, edgecolor='black', lw=0.7, hatch='..')
    r3 = ax.bar(x + width, lgmgq_threat, width, label='Test 2 (lgmgq)', color=COLOR_LGMGQ, alpha=0.92, edgecolor='black', lw=0.7)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Clean non-overlapping callout pointing to the right edge of the DH green bar
    ax.annotate("-76.9% Relative\nReduction in DH",
                xy=(0 + width + 0.11, 4.0), xytext=(0.48, 16.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_LGMGQ, lw=1.3),
                ha='center', va='center', fontsize=8.3, fontweight='bold', color=COLOR_LGMGQ,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor=COLOR_LGMGQ, lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(threats)
    ax.set_ylabel("Attack Success Rate (%)", fontweight='bold')
    ax.set_xlabel("Adversarial Threat Vector", fontweight='bold', labelpad=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 32.0)
    ax.grid(axis='y', linestyle=':', color='#CBD5E1', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    save_ieee_fig(fig, "fig5_threat_vectors")
    plt.close(fig)

# ==============================================================================
# FIGURE 6: PAIRED MCNEMAR 2x2 CONTINGENCY MATRICES (IEEE SUBFIGURES a & b)
# ==============================================================================
def plot_fig6_mcnemar_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    mat1 = np.array([[4, 11], [9, 78]])
    mat2 = np.array([[2, 13], [2, 85]])
    cmap = plt.cm.Blues

    # Subplot (a): lmqgg
    ax1.imshow(mat1, cmap=cmap, vmin=0, vmax=95)
    ax1.set_title("(a) Control vs. Test 1 (lmqgg)\n$\\chi^2 = 0.0500,\\ p = 0.8238$ (Not Significant)",
                  fontsize=10.2, fontweight='bold', pad=10, color=SLATE_DARK)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Breached (1)", "Safe (0)"], fontweight='bold')
    ax1.set_yticklabels(["Breached (1)", "Safe (0)"], fontweight='bold')
    ax1.set_xlabel("Test 1 (lmqgg) Outcome", fontweight='bold', labelpad=6)
    ax1.set_ylabel("Control (5× Llama) Outcome", fontweight='bold', labelpad=6)

    lbl1 = [["$a = 4$\n(Both Breached)", "$b = 11$\n(Control Only)"],
            ["$c = 9$\n(lmqgg Only)", "$d = 78$\n(Both Safe)"]]
    for i in range(2):
        for j in range(2):
            c = "white" if mat1[i, j] > 45 else SLATE_DARK
            ax1.text(j, i, lbl1[i][j], ha="center", va="center", color=c, fontweight='bold', fontsize=9.5)

    # Subplot (b): lgmgq
    ax2.imshow(mat2, cmap=cmap, vmin=0, vmax=95)
    ax2.set_title("(b) Control vs. Test 2 (lgmgq)\n$\\chi^2 = 6.6667,\\ p = 0.00739$ ($p < 0.01$, Significant)",
                  fontsize=10.2, fontweight='bold', pad=10, color=COLOR_LGMGQ)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Breached (1)", "Safe (0)"], fontweight='bold')
    ax2.set_yticklabels(["Breached (1)", "Safe (0)"], fontweight='bold')
    ax2.set_xlabel("Test 2 (lgmgq) Outcome", fontweight='bold', labelpad=6)
    ax2.set_ylabel("Control (5× Llama) Outcome", fontweight='bold', labelpad=6)

    lbl2 = [["$a = 2$\n(Both Breached)", "$b = 13$\n(Defense Gain)"],
            ["$c = 2$\n(lgmgq Only)", "$d = 85$\n(Both Safe)"]]
    for i in range(2):
        for j in range(2):
            c = "white" if mat2[i, j] > 45 else SLATE_DARK
            ax2.text(j, i, lbl2[i][j], ha="center", va="center", color=c, fontweight='bold', fontsize=9.5)

    plt.subplots_adjust(left=0.08, right=0.96, top=0.84, bottom=0.16, wspace=0.32)
    save_ieee_fig(fig, "fig6_mcnemar_matrices")
    plt.close(fig)

# ==============================================================================
# FIGURE 7 (AND fig_ieee_methodology): HIGH-LEVEL, UNCLUTTERED IEEE METHODOLOGY
# ==============================================================================
def plot_fig7_high_level_methodology():
    """
    Generate the four-phase research methodology block diagram
    (fig7_research_methodology and fig_ieee_methodology).
    """
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    ax.axis('off')
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.6)

    stages = [
        ("Phase 1: Benchmark Synthesis",
         "LLM-in-the-Loop Red-Teaming",
         [
             ("InjecAgent Seeds (17 Parent Tasks)", "34 base scenarios (17 DH, 17 DS)"),
             ("Generator ↔ Discriminator", "Gemini-Flash ↔ Claude Opus"),
             ("Factorial Matrix (N = 102)", "Base, Auth, Urgency (DH & DS)")
         ],
         0.25, NAVY, "#EFF6FF"),

        ("Phase 2: Controlled Setup",
         "Model Pool & Tool Isolation",
         [
             ("5 Open-Weight Families", "Llama, Mistral, Qwen, Gemma, GLM"),
             ("Unified 40-Tool Catalog", "17 Benign, 10 Attack, 13 Decoy"),
             ("Deterministic Local Runtime", "Ollama (T = 0.0, ctx = 8,192)")
         ],
         3.60, SLATE_BORDER, "#F8FAFC"),

        ("Phase 3: Pipeline Execution",
         "5-Hop Sequential Routing",
         [
             ("5 Sequential Agent Roles", "Ingestor→Planner→Writer→Rev→Exec"),
             ("Homogeneous Baseline", "Control: 5× Llama-3.1:8B"),
             ("Heterogeneous Permutations", "Test 1 (lmqgg) vs. Test 2 (lgmgq)")
         ],
         6.95, "#92400E", "#FFFBEB"),

        ("Phase 4: Hybrid Evaluation",
         "Two-Tier Oracle & Validation",
         [
             ("Hops 1–4 Semantic Judge", "DeepSeek-R1:14B (Survival $S_k$)"),
             ("Hop 5 Execution Oracle", "Deterministic Parser (Breach $E_5$)"),
             ("Hypothesis Tests (RQ1–RQ4)", "Attenuation $D_{\\mathrm{att}}$ & McNemar $\\chi^2$")
         ],
         10.30, COLOR_LGMGQ, "#F0FDF4"),
    ]

    w_col = 3.05
    for title, subtitle, items, sx, border_col, header_bg in stages:
        # Outer Stage Container
        draw_box(ax, sx, 0.25, w_col, 4.10, facecolor="#FFFFFF", edgecolor=border_col, lw=1.3, radius=0.08, zorder=1)
        # Stage Header Banner
        draw_box(ax, sx, 3.60, w_col, 0.75, facecolor=border_col, edgecolor=border_col, lw=1.3, radius=0.06, zorder=2)
        ax.text(sx + w_col / 2, 4.08, title, ha='center', va='center', fontsize=9.8, fontweight='bold', color="#FFFFFF", zorder=3)
        ax.text(sx + w_col / 2, 3.78, subtitle, ha='center', va='center', fontsize=8.2, style='italic', color="#E2E8F0", zorder=3)

        # 3 Clean Sub-blocks per Stage
        y_positions = [2.62, 1.55, 0.48]
        for idx, (b_head, b_sub) in enumerate(items):
            by = y_positions[idx]
            draw_box(ax, sx + 0.16, by, w_col - 0.32, 0.78, facecolor=header_bg, edgecolor=border_col, lw=0.9, radius=0.05, zorder=2)
            ax.text(sx + w_col / 2, by + 0.50, b_head, ha='center', va='center', fontsize=8.8, fontweight='bold', color=SLATE_DARK, zorder=3)
            ax.text(sx + w_col / 2, by + 0.22, b_sub, ha='center', va='center', fontsize=7.9, color="#334155", zorder=3)

            # Vertical mini-arrow between sub-blocks inside the stage
            if idx < 2:
                draw_arrow(ax, sx + w_col / 2, by, sx + w_col / 2, y_positions[idx + 1] + 0.78, color=border_col, lw=1.0, mutation=9)

        # Horizontal connecting arrow to next stage
        if sx < 10.30:
            draw_arrow(ax, sx + w_col + 0.03, 2.30, sx + 3.35 - 0.03, 2.30, color=SLATE_DARK, lw=1.8, mutation=14)

    save_ieee_fig(fig, "fig7_research_methodology")
    save_ieee_fig(fig, "fig_ieee_methodology")
    plt.close(fig)

def main():
    print("=" * 75)
    print("GENERATING IEEE FIGURES (Phase 5/figures/)")
    print("=" * 75)
    plot_fig1_pipeline_schematic()
    plot_fig2_survival_curve()
    plot_fig3_datt_distribution()
    plot_fig4_framing_resilience()
    plot_fig5_threat_vectors()
    plot_fig6_mcnemar_matrices()
    plot_fig7_high_level_methodology()
    print("=" * 75)
    print("COMPLETE: ALL FIGURES SAVED IN IEEE FORMAT (PNG, PDF, SVG)")
    print("=" * 75)

if __name__ == "__main__":
    main()
