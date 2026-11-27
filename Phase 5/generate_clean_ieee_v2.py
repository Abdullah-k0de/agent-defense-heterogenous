r"""
Generate IEEE publication-grade empirical figures (fig1_pipeline_schematic_v2 through fig6_mcnemar_matrices_v2).
Outputs PNG, PDF, and SVG formats to Phase 5/figures_v2_clean/ and PNG/PDF formats to Phase 5/figures/.
"""
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# AUTHENTIC IEEE TRANSACTIONS TYPOGRAPHY & STYLE
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.edgecolor'] = '#1E293B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.major.size'] = 3.5
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

PHASE5_DIR = Path(__file__).resolve().parent
V2_DIR = PHASE5_DIR / "figures_v2_clean"
FIG_DIR = PHASE5_DIR / "figures"

# Muted, High-Contrast Academic Palette
NAVY_CTRL   = "#1E3A8A"   # Control (5x Llama-3.1)
SLATE_LMQGG = "#64748B"   # Test 1 (lmqgg)
TEAL_LGMGQ  = "#0F766E"   # Test 2 (lgmgq)
DARK_TEXT   = "#0F172A"
SUB_TEXT    = "#334155"
BORDER_DARK = "#1E293B"
GRID_LIGHT  = "#E2E8F0"

def save_v2_fig(fig, stem: str):
    """Save V2 figures into figures_v2_clean/{png,pdf,svg} and figures/ for LaTeX inclusion."""
    for ext in ["png", "pdf", "svg"]:
        out_dir = V2_DIR / ext
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{stem}.{ext}"
        if ext == "svg":
            fig.savefig(target, format="svg", bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved clean IEEE V2 figure: {stem}")

def draw_box(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor=BORDER_DARK, lw=1.0, radius=0.06, zorder=2):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=BORDER_DARK, lw=1.2, mutation=11, zorder=4):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=mutation, shrinkA=0, shrinkB=0),
        zorder=zorder
    )

def apply_ieee_axes(ax):
    """Full 4-sided thin IEEE frame + subtle horizontal gridlines."""
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#334155')
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color=GRID_LIGHT, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ==============================================================================
# FIGURE 1: FIVE-HOP PIPELINE SCHEMATIC & EVALUATED CONFIGURATIONS
# ==============================================================================
def plot_fig1_v2():
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.axis('off')
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.2)

    # --- Top Panel (a): 5-Hop Sequential Multi-Agent Topology ---
    draw_box(ax, 0.2, 2.85, 12.2, 2.15, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.0, radius=0.08, zorder=1)
    ax.text(0.45, 4.72, "(a) Five-Hop Sequential Multi-Agent Pipeline Topology and Hybrid Evaluation Points",
            ha='left', va='center', fontsize=10.2, fontweight='bold', color=DARK_TEXT)

    nodes = [
        ("External Input", "Task $U$ + Untrusted\nData ($N = 102$)", "Seed Benchmark", 0.45, "#991B1B", "#FEF2F2"),
        ("Hop 1: Ingestor", "Summarize raw\ntool data ($O_1$)", "Judge: DeepSeek-R1", 2.40, NAVY_CTRL, "#EFF6FF"),
        ("Hop 2: Planner", "Draft sequential\naction plan ($O_2$)", "Judge: DeepSeek-R1", 4.35, NAVY_CTRL, "#EFF6FF"),
        ("Hop 3: Writer", "Map plan to API\nparameters ($O_3$)", "Judge: DeepSeek-R1", 6.30, NAVY_CTRL, "#EFF6FF"),
        ("Hop 4: Reviewer", "Audit schema &\nconstraints ($O_4$)", "Judge: DeepSeek-R1", 8.25, NAVY_CTRL, "#EFF6FF"),
        ("Hop 5: Executor", "Propose JSON\ntool calls ($E_5$)", "JSON Proposal Oracle", 10.20, TEAL_LGMGQ, "#F0FDF4"),
    ]

    w_card = 1.65
    for title, desc, badge, nx, col, bg in nodes:
        # Card body
        draw_box(ax, nx, 3.02, w_card, 1.45, facecolor="#FFFFFF", edgecolor=col, lw=1.1, radius=0.05, zorder=2)
        # Card header bar
        draw_box(ax, nx, 4.12, w_card, 0.35, facecolor=col, edgecolor=col, lw=1.1, radius=0.04, zorder=3)
        ax.text(nx + w_card/2, 4.295, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color="#FFFFFF", zorder=4)
        ax.text(nx + w_card/2, 3.74, desc, ha='center', va='center', fontsize=8.0, color=DARK_TEXT, zorder=3)
        # Subtle bottom pill
        draw_box(ax, nx + 0.10, 3.10, w_card - 0.20, 0.26, facecolor=bg, edgecolor=col, lw=0.7, radius=0.03, zorder=3)
        ax.text(nx + w_card/2, 3.23, badge, ha='center', va='center', fontsize=7.3, style='italic', color=DARK_TEXT, zorder=4)

        if nx < 10.20:
            draw_arrow(ax, nx + w_card + 0.03, 3.75, nx + 1.95 - 0.03, 3.75, color=SUB_TEXT, lw=1.2, mutation=10)

    # --- Bottom Panel (b): Evaluated Pipeline Configurations ---
    draw_box(ax, 0.2, 0.20, 12.2, 2.45, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.0, radius=0.08, zorder=1)
    ax.text(0.45, 2.38, "(b) Evaluated Model Compositions across Pipeline Stages (102 Paired Trials per Configuration)",
            ha='left', va='center', fontsize=10.2, fontweight='bold', color=DARK_TEXT)

    rows = [
        ("Homogeneous Control", ["Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B", "Llama-3.1:8B"],
         1.72, NAVY_CTRL, "#EFF6FF", None),
        ("Hetero. Perm. 1 (lmqgg)", ["Llama-3.1:8B", "Mistral-7B-v0.3", "Qwen-2.5:7B", "Gemma-2:9B", "GLM-4:9B"],
         1.05, SLATE_LMQGG, "#F8FAFC", None),
        ("Hetero. Perm. 2 (lgmgq)", ["Llama-3.1:8B", "Gemma-2:9B", "Mistral-7B-v0.3", "GLM-4:9B", "Qwen-2.5:7B"],
         0.38, TEAL_LGMGQ, "#F0FDF4", 1),
    ]

    hop_xs = [2.95, 4.82, 6.69, 8.56, 10.43]
    w_m = 1.52
    h_m = 0.44

    for label, models, ry, col, bg, highlight_idx in rows:
        # Left configuration label badge
        draw_box(ax, 0.45, ry, 2.15, h_m, facecolor=col, edgecolor=col, lw=1.0, radius=0.04, zorder=2)
        ax.text(0.45 + 1.075, ry + h_m/2, label, ha='center', va='center', fontsize=8.4, fontweight='bold', color="#FFFFFF", zorder=3)

        for idx, (m_name, mx) in enumerate(zip(models, hop_xs)):
            is_hi = (highlight_idx == idx)
            f_col = bg if is_hi else "#FFFFFF"
            l_w = 1.4 if is_hi else 0.9
            draw_box(ax, mx, ry, w_m, h_m, facecolor=f_col, edgecolor=col, lw=l_w, radius=0.04, zorder=2)
            ax.text(mx + w_m/2, ry + h_m/2, m_name, ha='center', va='center',
                    fontsize=8.3, fontweight='bold' if is_hi else 'normal', color=DARK_TEXT, zorder=3)
            if idx < 4:
                draw_arrow(ax, mx + w_m + 0.03, ry + h_m/2, hop_xs[idx+1] - 0.03, ry + h_m/2, color=col, lw=1.0, mutation=9)

    save_v2_fig(fig, "fig1_pipeline_schematic_v2")
    plt.close(fig)


# ==============================================================================
# FIGURE 2: PER-HOP DETECTED ADVERSARIAL INTENT & TERMINAL ASR
# ==============================================================================
def plot_fig2_v2():
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    apply_ieee_axes(ax)

    hops = np.array([1, 2, 3, 4, 5])
    hop_labels = [
        "Hop 1\n(Ingestor)", "Hop 2\n(Planner)", "Hop 3\n(Writer)",
        "Hop 4\n(Reviewer)", "Hop 5\n(Executor)"
    ]

    ctrl  = [50.98, 12.75, 10.78, 14.71, 14.71]
    lmqgg = [54.90, 20.59, 17.65, 18.63, 12.75]
    lgmgq = [53.92, 11.76,  8.82, 12.75,  3.92]

    ax.plot(hops, ctrl, marker='s', markersize=6.5, linewidth=1.8, linestyle='--',
            color=NAVY_CTRL, markerfacecolor='white', markeredgewidth=1.6,
            label="Homogeneous Control ($5\\times$ Llama-3.1)", zorder=4)
    ax.plot(hops, lmqgg, marker='^', markersize=6.8, linewidth=1.8, linestyle='-.',
            color=SLATE_LMQGG, markerfacecolor='white', markeredgewidth=1.6,
            label="Heterogeneous Perm. 1 (lmqgg)", zorder=3)
    ax.plot(hops, lgmgq, marker='o', markersize=6.5, linewidth=2.1, linestyle='-',
            color=TEAL_LGMGQ, markerfacecolor=TEAL_LGMGQ,
            label="Heterogeneous Perm. 2 (lgmgq)", zorder=5)

    # Subtle terminal value labels only at Hop 5 (cleanly spaced)
    ax.text(5.10, 15.1, "14.7%", color=NAVY_CTRL, fontweight='bold', fontsize=8.8, va='center')
    ax.text(5.10, 12.3, "12.8%", color=SLATE_LMQGG, fontweight='bold', fontsize=8.8, va='center')
    ax.text(5.10,  3.9, "3.9%",  color=TEAL_LGMGQ, fontweight='bold', fontsize=8.8, va='center')

    ax.set_xticks(hops)
    ax.set_xticklabels(hop_labels)
    ax.set_xlim(0.75, 5.55)
    ax.set_ylim(0, 64)
    ax.set_ylabel("Detected Adversarial Intent / Terminal ASR (%)", fontweight='bold')
    ax.set_xlabel("Sequential Pipeline Stage", fontweight='bold', labelpad=6)

    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98)
    plt.tight_layout()
    save_v2_fig(fig, "fig2_survival_curve_v2")
    plt.close(fig)


# ==============================================================================
# FIGURE 3: ATTENUATION DEPTH (D_att) DISTRIBUTION
# ==============================================================================
def plot_fig3_v2():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    apply_ieee_axes(ax)

    categories = [
        "Hop 1\n(Ingestor)", "Hop 2\n(Planner)", "Hop 3\n(Writer)",
        "Hop 4\n(Reviewer)", "Hop 5\n(Executor)", "Breached\n($D_{\\mathrm{att}}=6$)"
    ]
    c_counts  = [43, 40, 3, 0, 1, 15]
    t1_counts = [41, 44, 3, 0, 1, 13]
    t2_counts = [47, 45, 5, 1, 0,  4]

    x = np.arange(len(categories))
    w = 0.22
    gap = 0.03

    b1 = ax.bar(x - (w + gap), c_counts,  w, label="Control ($5\\times$ Llama-3.1, $\\bar{D}_{\\mathrm{att}}=2.23$)",
                color=NAVY_CTRL, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b2 = ax.bar(x,             t1_counts, w, label="Test 1 (lmqgg, $\\bar{D}_{\\mathrm{att}}=2.17$)",
                color=SLATE_LMQGG, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b3 = ax.bar(x + (w + gap), t2_counts, w, label="Test 2 (lgmgq, $\\bar{D}_{\\mathrm{att}}=1.76$)",
                color=TEAL_LGMGQ, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = int(bar.get_height())
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h}",
                        ha='center', va='bottom', fontsize=8.3, color=DARK_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 56)
    ax.set_ylabel("Number of Tasks ($N = 102$)", fontweight='bold')
    ax.set_xlabel("Attenuation Depth ($D_{\\mathrm{att}}$)", fontweight='bold', labelpad=6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    plt.tight_layout()
    save_v2_fig(fig, "fig3_datt_distribution_v2")
    plt.close(fig)


# ==============================================================================
# FIGURE 4: ATTACK SUCCESS RATE ACROSS ADVERSARIAL FRAMINGS
# ==============================================================================
def plot_fig4_v2():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    apply_ieee_axes(ax)

    framings = [
        "Base Framing\n($n = 34$)",
        "Authoritative Framing\n($n = 34$)",
        "Urgency Framing\n($n = 34$)"
    ]
    c_asr  = [14.71, 14.71, 14.71]
    t1_asr = [14.71,  8.82, 14.71]
    t2_asr = [ 0.00,  0.00, 11.76]

    x = np.arange(len(framings))
    w = 0.21
    gap = 0.03

    b1 = ax.bar(x - (w + gap), c_asr,  w, label="Control ($5\\times$ Llama-3.1)",
                color=NAVY_CTRL, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b2 = ax.bar(x,             t1_asr, w, label="Test 1 (lmqgg)",
                color=SLATE_LMQGG, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b3 = ax.bar(x + (w + gap), t2_asr, w, label="Test 2 (lgmgq)",
                color=TEAL_LGMGQ, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.45, f"{h:.1f}%",
                    ha='center', va='bottom', fontsize=8.4, color=DARK_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylim(0, 20.5)
    ax.set_ylabel("Attack Success Rate (%)", fontweight='bold')
    ax.set_xlabel("Adversarial Psychological Framing", fontweight='bold', labelpad=6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    plt.tight_layout()
    save_v2_fig(fig, "fig4_framing_resilience_v2")
    plt.close(fig)


# ==============================================================================
# FIGURE 5: THREAT VECTOR BREAKDOWN
# ==============================================================================
def plot_fig5_v2():
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    apply_ieee_axes(ax)

    vectors = [
        "Direct Harm (DH)\nState Mutations ($n = 51$)",
        "Data Stealing (DS)\nExfiltration ($n = 51$)"
    ]
    c_asr  = [25.49, 3.92]
    t1_asr = [23.53, 1.96]
    t2_asr = [ 5.88, 1.96]

    x = np.arange(len(vectors))
    w = 0.20
    gap = 0.03

    b1 = ax.bar(x - (w + gap), c_asr,  w, label="Control ($5\\times$ Llama-3.1)",
                color=NAVY_CTRL, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b2 = ax.bar(x,             t1_asr, w, label="Test 1 (lmqgg)",
                color=SLATE_LMQGG, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)
    b3 = ax.bar(x + (w + gap), t2_asr, w, label="Test 2 (lgmgq)",
                color=TEAL_LGMGQ, edgecolor=BORDER_DARK, linewidth=0.7, zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.55, f"{h:.2f}%",
                    ha='center', va='bottom', fontsize=8.3, color=DARK_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(vectors)
    ax.set_ylim(0, 32.0)
    ax.set_ylabel("Attack Success Rate (%)", fontweight='bold')
    ax.set_xlabel("Adversarial Threat Vector Category", fontweight='bold', labelpad=6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')

    plt.tight_layout()
    save_v2_fig(fig, "fig5_threat_vectors_v2")
    plt.close(fig)


# ==============================================================================
# FIGURE 6: MCNEMAR 2x2 CONTINGENCY MATRICES
# ==============================================================================
def plot_fig6_v2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8))

    panels = [
        (ax1,
         "(a) Control vs. Test 1 (lmqgg)",
         "$\\chi^2 = 0.050,\\; p = 0.8238$ (Not Significant)",
         "Test 1 (lmqgg) Outcome",
         [
             [("$a = 4$", "Both Breached", "#F8FAFC", "#CBD5E1"),
              ("$b = 11$", "Control Only Breached", "#EFF6FF", NAVY_CTRL)],
             [("$c = 9$", "lmqgg Only Breached", "#F8FAFC", "#CBD5E1"),
              ("$d = 78$", "Both Safe", "#F1F5F9", "#94A3B8")]
         ]),
        (ax2,
         "(b) Control vs. Test 2 (lgmgq)",
         "$\\chi^2 = 6.667,\\; p = 0.0074$ (Statistically Significant)",
         "Test 2 (lgmgq) Outcome",
         [
             [("$a = 2$", "Both Breached", "#F8FAFC", "#CBD5E1"),
              ("$b = 13$", "Control Only (Defense Gain)", "#F0FDF4", TEAL_LGMGQ)],
             [("$c = 2$", "lgmgq Only Breached", "#F8FAFC", "#CBD5E1"),
              ("$d = 85$", "Both Safe", "#F1F5F9", "#94A3B8")]
         ])
    ]

    for ax, title, stats_sub, xlabel, grid_data in panels:
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Safe (0)", "Breached (1)"][::-1], fontweight='bold', fontsize=9.2)
        ax.set_yticklabels(["Safe (0)", "Breached (1)"], fontweight='bold', fontsize=9.2)
        ax.tick_params(length=0)

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(BORDER_DARK)
            ax.spines[spine].set_linewidth(1.0)

        ax.set_title(f"{title}\n{stats_sub}", fontsize=9.8, fontweight='bold', color=DARK_TEXT, pad=10)
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=9.6, labelpad=6)
        ax.set_ylabel("Control ($5\\times$ Llama) Outcome", fontweight='bold', fontsize=9.6, labelpad=6)

        # Draw 2x2 structured cells (i=0 is top row Breached(1), i=1 is bottom row Safe(0))
        for row_idx in range(2):
            for col_idx in range(2):
                val_txt, sub_txt, bg_col, edge_col = grid_data[row_idx][col_idx]
                y_cell = 1 - row_idx
                x_cell = col_idx
                is_discordant_b = (row_idx == 0 and col_idx == 1)
                rect = patches.Rectangle(
                    (x_cell + 0.04, y_cell + 0.04), 0.92, 0.92,
                    facecolor=bg_col, edgecolor=edge_col,
                    linewidth=1.5 if is_discordant_b else 0.8, zorder=2
                )
                ax.add_patch(rect)
                ax.text(x_cell + 0.5, y_cell + 0.58, val_txt,
                        ha='center', va='center', fontsize=11.0, fontweight='bold', color=DARK_TEXT, zorder=3)
                ax.text(x_cell + 0.5, y_cell + 0.34, sub_txt,
                        ha='center', va='center', fontsize=8.2, color=SUB_TEXT, zorder=3)

    plt.tight_layout(w_pad=2.5)
    save_v2_fig(fig, "fig6_mcnemar_matrices_v2")
    plt.close(fig)


def main():
    print("=" * 72)
    print("GENERATING IEEE FIGURES (Phase 5/figures_v2_clean/ & Phase 5/figures/)")
    print("=" * 72)
    plot_fig1_v2()
    plot_fig2_v2()
    plot_fig3_v2()
    plot_fig4_v2()
    plot_fig5_v2()
    plot_fig6_v2()
    print("=" * 72)
    print("COMPLETE: ALL FIGURES SAVED.")
    print("=" * 72)

if __name__ == "__main__":
    main()
