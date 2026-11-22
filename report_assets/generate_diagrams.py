#!/usr/bin/env python3
"""
Generate high-resolution (300 DPI) publication-quality research diagrams:
1. Fig 1: RAG Pipeline with Spurious Feature Injection
2. Fig 2: M2 Baseline Results (Clean vs Spurious Noisy)
3. Fig 3: M4 Three-Condition Comparison (Clean, Noisy Unfiltered, Noisy Filtered)
4. Fig 4: Ablation Study (Threshold vs Performance)
5. Fig 5: M5 Outcome Matrix Heatmap (2x2 Transition Analysis)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: RAG Pipeline Architecture with Spurious Injection
# -------------------------------------------------------------
def generate_fig1():
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Box coordinates and labels
    boxes = [
        (0.5, 3.5, 1.8, 1.2, "User Query\n(q)", "#E3F2FD", "#1565C0"),
        (2.8, 3.5, 1.8, 1.2, "Retriever\n(R)", "#E8EAF6", "#283593"),
        (5.1, 3.5, 2.0, 1.2, "Grounding Context\n{d1, ..., dk}", "#E0F2F1", "#00695C"),
        (7.6, 3.5, 2.2, 1.2, "Language Model\n(Reader G)", "#FFF3E0", "#E65100"),
        (10.3, 3.5, 1.4, 1.2, "Answer\n(y)", "#E8F5E9", "#2E7D32")
    ]

    for x, y, w, h, text, facecolor, edgecolor in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                      facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#212121')

    # Horizontal connecting arrows
    arrow_props = dict(arrowstyle="->", lw=2.5, color="#37474F", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(2.8, 4.1), xytext=(2.3, 4.1), arrowprops=arrow_props)
    ax.annotate("", xy=(5.1, 4.1), xytext=(4.6, 4.1), arrowprops=arrow_props)
    ax.annotate("", xy=(7.6, 4.1), xytext=(7.1, 4.1), arrowprops=arrow_props)
    ax.annotate("", xy=(10.3, 4.1), xytext=(9.8, 4.1), arrowprops=arrow_props)

    # Spurious injection box and downward dashed arrow
    spurious_box = (5.1, 0.8, 2.0, 1.2, "⚠ Spurious\nDistractor", "#FFEBEE", "#C62828")
    sx, sy, sw, sh, stext, sface, sedge = spurious_box
    s_rect = patches.FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.1",
                                    facecolor=sface, edgecolor=sedge, linewidth=2, linestyle='--')
    ax.add_patch(s_rect)
    ax.text(sx + sw / 2, sy + sh / 2, stext, ha='center', va='center',
            fontsize=11, fontweight='bold', color='#B71C1C')

    # Red dashed arrow pointing downward
    down_arrow = dict(arrowstyle="->", lw=2.5, color="#C62828", linestyle="--", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(6.1, 2.0), xytext=(6.1, 3.5), arrowprops=down_arrow)
    ax.text(6.25, 2.7, "Spurious Feature\nInjection", ha='left', va='center',
            fontsize=9.5, fontweight='bold', color="#C62828")

    plt.title("Figure 1: RAG Pipeline with Spurious Feature Injection",
              fontsize=14, fontweight='bold', pad=15, color="#1A237E")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_rag_pipeline.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# Figure 2: M2 Baseline Results Bar Chart
# -------------------------------------------------------------
def generate_fig2():
    labels = ['Clean Condition', 'Noisy Condition']
    em_scores = [68.00, 54.00]
    f1_scores = [73.17, 60.98]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    rects1 = ax.bar(x - width/2, em_scores, width, label='Exact Match (EM)', color='#1976D2', edgecolor='#0D47A1', lw=1.5)
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Token F1 Score', color='#FF9800', edgecolor='#E65100', lw=1.5)

    # Clean EM ceiling reference line
    ax.axhline(y=68.0, color='#D32F2F', linestyle='--', linewidth=2, label='Clean EM Ceiling (68.0%)')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', framealpha=0.9, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Add direct value labels on top of bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Figure 2: M2 Baseline Results — Clean vs Spurious Noisy Condition",
              fontsize=13, fontweight='bold', pad=15, color="#1A237E")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_m2_results.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# Figure 3: M4 Three Conditions Comparison
# -------------------------------------------------------------
def generate_fig3():
    conditions = ['Clean', 'Noisy Unfiltered', 'Noisy Filtered']
    em_scores = [68.00, 54.00, 16.00]
    colors = ['#2E7D32', '#F57C00', '#D32F2F']
    edge_colors = ['#1B5E20', '#E65100', '#B71C1C']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(conditions, em_scores, color=colors, edgecolor=edge_colors, width=0.55, lw=1.5)

    # Reference lines
    ax.axhline(y=68.0, color='#2E7D32', linestyle='--', linewidth=1.8, label='Clean Ceiling (68.0% EM)')
    ax.axhline(y=54.0, color='#F57C00', linestyle='--', linewidth=1.8, label='Noisy Baseline (54.0% EM)')

    ax.set_ylabel('Exact Match Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotation for the filtered bar failure
    ax.annotate("⚠ 44% Collateral Deletion",
                xy=(2, 16.00), xytext=(2, 35.0),
                arrowprops=dict(facecolor='#B71C1C', shrink=0.08, width=2, headwidth=8),
                ha='center', fontsize=10.5, fontweight='bold', color='#B71C1C',
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#B71C1C", lw=1.5))

    plt.title("Figure 3: M4 Three-Condition Results — Filter Catastrophic Failure",
              fontsize=13, fontweight='bold', pad=15, color="#1A237E")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_m4_conditions.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# Figure 4: Ablation Study Line Chart
# -------------------------------------------------------------
def generate_fig4():
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    em = [14.0, 12.0, 16.0, 20.0, 30.0]
    f1 = [14.14, 12.14, 16.14, 21.48, 32.78]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    ax.plot(thresholds, em, marker='o', markersize=8, color='#1976D2', linewidth=2.5, label='Exact Match (EM)')
    ax.plot(thresholds, f1, marker='s', markersize=8, color='#FF9800', linewidth=2.5, label='Token F1 Score')

    # Horizontal baseline reference lines
    ax.axhline(y=54.0, color='#D32F2F', linestyle='--', linewidth=2, label='Noisy Baseline EM (54.0%)')
    ax.axhline(y=68.0, color='#2E7D32', linestyle='--', linewidth=2, label='Clean Baseline EM (68.0%)')

    ax.set_xlabel('Jaccard Overlap Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticks(thresholds)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center left', frameon=True, fontsize=10)

    # Annotation at (0.5, 30.0)
    ax.annotate("Best: 30% EM",
                xy=(0.5, 30.0), xytext=(0.42, 42.0),
                arrowprops=dict(facecolor='#1976D2', shrink=0.08, width=1.8, headwidth=7),
                fontsize=11, fontweight='bold', color='#1565C0',
                bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1565C0", lw=1.2))

    plt.title("Figure 4: Ablation Study — Threshold vs Performance",
              fontsize=13, fontweight='bold', pad=15, color="#1A237E")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_ablation.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------
# Figure 5: M5 Failure 2x2 Outcome Matrix Heatmap
# -------------------------------------------------------------
def generate_fig5():
    # 2x2 counts:
    # Row 0: Noisy Correct -> [Filtered Correct (Both Correct), Filtered Wrong (Filter Hurt)]
    # Row 1: Noisy Wrong   -> [Filtered Correct (Filter Helped), Filtered Wrong (Both Wrong)]
    matrix_counts = np.array([
        [5, 22],
        [3, 20]
    ])

    labels = np.array([
        ["Both Correct\n(N=5, 10%)", "Filter Hurt\n[Collateral Deletion]\n(N=22, 44%)"],
        ["Filter Helped\n[Recovery]\n(N=3, 6%)", "Both Wrong\n[Persistent Error]\n(N=20, 40%)"]
    ])

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    # Diverging colormap: green for helpful/robust, red for damage
    # Custom values for color mapping:
    # (0,0): 5 (neutral/good), (0,1): 22 (hurt/bad), (1,0): 3 (helped/good), (1,1): 20 (both wrong)
    # Let's use a custom colored cell matrix
    color_map = np.array([
        [0.65, 0.05],  # 5=greenish/neutral, 22=deep red
        [0.85, 0.20]   # 3=bright green, 20=red
    ])

    sns.heatmap(color_map, annot=labels, fmt="", cmap="RdYlGn", cbar=False,
                xticklabels=['Filtered Correct', 'Filtered Wrong'],
                yticklabels=['Noisy Correct', 'Noisy Wrong'],
                linewidths=3, linecolor='white', ax=ax,
                annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#111111"})

    ax.set_xlabel("Condition C (Filtered at τ=0.3)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Condition B (Noisy Unfiltered)", fontsize=12, fontweight='bold', labelpad=10)

    plt.suptitle("Figure 5: M5 Outcome Matrix — Filter Effect on 50 Examples",
                 fontsize=13, fontweight='bold', color="#1A237E", y=0.98)
    plt.title("Filter Hurt: 22 | Filter Helped: 3 | Net Effect: Harmful",
              fontsize=11, fontweight='semibold', color="#C62828", pad=12)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig5_outcome_matrix.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("All 5 diagrams successfully generated at 300 DPI!")
