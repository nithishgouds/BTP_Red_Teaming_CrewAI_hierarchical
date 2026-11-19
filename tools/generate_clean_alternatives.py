"""
Script to generate ultra-clean, spacious, publication-grade alternative visualizations:
1. docs/figures/leakage_clean_presentation.png
2. docs/figures/tradeoff_clean_presentation.png

Key Design Goals:
- High readability: Larger fonts (11-14pt), zero overlapping text, zero crowded annotations.
- Generous padding and expanded axis limits for ample breathing room.
- Clear, distinct data points and high-contrast color palette.
- Does NOT overwrite the original first PNGs.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Apply clean academic aesthetic
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.3
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# FIGURE 1: CLEAN LEAKAGE MITIGATION PRESENTATION
# ==============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Panel 1: Pre-Transmission Leakage by Model
models = ['Gemma 2 (2B)', 'Llama 3.2 (3B)', 'Mistral (7B)']
x = np.arange(len(models))
width = 0.24

# Exact empirical values (entity-pooled pre-transmission leakage %)
cfg_a = [38.46, 16.67, 14.29]
cfg_b = [0.00, 0.00, 0.00]
cfg_c = [0.00, 0.00, 0.00]

r1 = ax1.bar(x - width, cfg_a, width, label='Config A: Baseline (No Guardrail)', color='#D9534F', edgecolor='#333333', linewidth=1.2, zorder=3)
r2 = ax1.bar(x, cfg_b, width, label='Config B: + Presidio Guardrail', color='#F0AD4E', edgecolor='#333333', linewidth=1.2, zorder=3)
r3 = ax1.bar(x + width, cfg_c, width, label='Config C: + Guardrail + ChromaDB + Hierarchical', color='#2E7D32', edgecolor='#333333', linewidth=1.2, zorder=3)

ax1.set_ylabel("Pre-Transmission PII Exposure Rate (%)", fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title("(a) PII Leakage Rate by Architecture (N = 45 Live Runs)", fontsize=13, fontweight='bold', pad=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11.5, fontweight='bold')
ax1.set_ylim(0, 56)  # Extra room at top to prevent label clipping
ax1.grid(axis='y', zorder=0)
ax1.legend(loc='upper right', framealpha=0.96, fontsize=9.5, edgecolor='#CCCCCC')

# Clear, spaced value labels
for rect in r1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + 1.2, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#C9302C')

for rect in r2:
    ax1.text(rect.get_x() + rect.get_width()/2, 1.2, '0.0%',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#B26A00')

for rect in r3:
    ax1.text(rect.get_x() + rect.get_width()/2, 1.2, '0.0%',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1B5E20')

# Clean non-intrusive status banner
ax1.text(0.5, 51.5, "Actual Egress Transmission: 0.00% across all 30 guarded runs",
         ha='center', fontsize=9.5, fontweight='bold', color='#155724',
         bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#A5D6A7", lw=1))

# Panel 2: Threat Class Containment
threats = [
    'Delimiter Smuggling (n=3)',
    'Context Hijacking (n=3)',
    'Selective Exfiltration (n=3)',
    'Encoded Payload (n=3)',
    'Role Confusion (n=3)'
]
y = np.arange(len(threats))
bar_h = 0.32

# Peak exposure rate across any model in Config A vs Guarded
peak_exposure = [100.0, 75.0, 12.5, 12.5, 12.5]
guarded_rate  = [0.0,   0.0,  0.0,  0.0,  0.0]

b1 = ax2.barh(y + bar_h/2, peak_exposure, bar_h, label='Config A: Peak Exposure (Worst-Case)', color='#C9302C', edgecolor='#333333', linewidth=1.2, zorder=3)
b2 = ax2.barh(y - bar_h/2, guarded_rate, bar_h, label='Configs B & C: Guarded (100% Contained)', color='#2E7D32', edgecolor='#333333', linewidth=1.2, zorder=3)

ax2.set_xlabel("Peak PII Exposure Rate (%)", fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title("(b) Vulnerability Containment by Threat Class", fontsize=13, fontweight='bold', pad=14)
ax2.set_yticks(y)
ax2.set_yticklabels(threats, fontsize=10.5, fontweight='bold')
ax2.set_xlim(0, 125)  # Generous width so labels don't get clipped
ax2.grid(axis='x', zorder=0)
ax2.legend(loc='lower right', framealpha=0.96, fontsize=9.5, edgecolor='#CCCCCC')

for rect in b1:
    w = rect.get_width()
    ax2.text(w + 2, rect.get_y() + rect.get_height()/2, f'{w:.1f}%',
             ha='left', va='center', fontsize=10, fontweight='bold', color='#C9302C')

for rect in b2:
    ax2.text(2, rect.get_y() + rect.get_height()/2, '0.0%',
             ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1B5E20')

plt.tight_layout(pad=3.0)

clean_leakage_png = os.path.join(output_dir, "leakage_clean_presentation.png")
clean_leakage_pdf = os.path.join(output_dir, "leakage_clean_presentation.pdf")
fig1.savefig(clean_leakage_png, dpi=300, bbox_inches='tight')
fig1.savefig(clean_leakage_pdf, dpi=300, bbox_inches='tight')
plt.close(fig1)

print(f"Generated: {clean_leakage_png}")


# ==============================================================================
# FIGURE 2: CLEAN ARCHITECTURAL TRADE-OFF PRESENTATION
# ==============================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Panel 1: 2D Trade-off Frontier
# Systems data: [Name, Flexibility %, Determinism %, Color, Marker, Size, Text offset (dx, dy)]
pts = [
    ("LangGraph\n(Static State Machine)", 25, 96, "#4A6FA5", "s", 260, (-5, -16), "High Determinism\nLow Flexibility"),
    ("Sequential CrewAI\n(Config A Baseline)", 88, 34, "#D9534F", "X", 300, (-14, 12), "High Flexibility\nLow Determinism"),
    ("Sequential + Guardrail\n(Config B)", 84, 70, "#F0AD4E", "o", 260, (-14, -16), "Inline Redaction\nModerate Determinism"),
    ("Zero-Trust Hierarchical\n(Config C Innovation)", 78, 92, "#2E7D32", "*", 450, (-14, 12), "Zero-Trust Supervisor\nHigh Determinism & Flexibility")
]

# Classical linear boundary connecting extreme endpoints
x_line = np.array([25, 88])
y_line = np.array([96, 34])
ax3.plot(x_line, y_line, color="#888888", linestyle="-.", linewidth=2, label="Classical MAS Trade-off Line (Slope: -0.98)")

# Shade the desirable high-performance zone
ax3.fill_between([65, 95], 80, 102, color="#E8F5E9", alpha=0.5, zorder=1)
ax3.text(80, 82, "Security-Hardened Envelope\n[D ≥ 80%, F ≥ 65%]", color="#1B5E20", fontsize=9, fontweight='bold', ha='center')

for name, flex, det, col, m, sz, (dx, dy), desc in pts:
    ax3.scatter(flex, det, color=col, marker=m, s=sz, edgecolors='black', linewidths=1.3, zorder=5, label=name.split('\n')[0])
    ax3.annotate(
        f"{name}\n({desc})",
        xy=(flex, det),
        xytext=(flex + dx, det + dy),
        fontsize=8.5, fontweight='bold', color=col,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.2, alpha=0.95),
        zorder=6
    )

ax3.set_xlim(10, 105)
ax3.set_ylim(15, 110)
ax3.set_xlabel("Workflow & Reasoning Flexibility (%)", fontsize=12, fontweight='bold', labelpad=10)
ax3.set_ylabel("Execution Determinism & State Invariance (%)", fontsize=12, fontweight='bold', labelpad=10)
ax3.set_title("(a) Architectural Pareto Frontier: Determinism vs. Flexibility", fontsize=13, fontweight='bold', pad=14)
ax3.grid(True, zorder=0)
ax3.legend(loc='lower left', framealpha=0.96, fontsize=9, edgecolor='#CCCCCC')

# Panel 2: Multi-Attribute Profile
attrs = [
    "Determinism\nScore",
    "Reasoning\nFlexibility",
    "Auditability\nDepth",
    "Injection\nResistance",
    "Latency\nEfficiency*"
]
x_attr = np.arange(len(attrs))
bar_w = 0.25

lang_scores = [96, 25, 60, 75, 95]
seq_scores  = [34, 88, 10, 15, 78]
hier_scores = [92, 78, 100, 100, 44]

ab1 = ax4.bar(x_attr - bar_w, lang_scores, bar_w, label='LangGraph (Static Graph)', color='#4A6FA5', edgecolor='#333333', linewidth=1.2, zorder=3)
ab2 = ax4.bar(x_attr, seq_scores, bar_w, label='Sequential CrewAI (Config A)', color='#D9534F', edgecolor='#333333', linewidth=1.2, zorder=3)
ab3 = ax4.bar(x_attr + bar_w, hier_scores, bar_w, label='Zero-Trust Hierarchical (Config C)', color='#2E7D32', edgecolor='#333333', linewidth=1.2, zorder=3)

ax4.set_ylabel("Normalized Evaluation Score (0 - 100)", fontsize=12, fontweight='bold', labelpad=10)
ax4.set_title("(b) Multi-Attribute Architectural Evaluation Profile", fontsize=13, fontweight='bold', pad=14)
ax4.set_xticks(x_attr)
ax4.set_xticklabels(attrs, fontsize=10.5, fontweight='bold')
ax4.set_ylim(0, 125)
ax4.grid(axis='y', zorder=0)
ax4.legend(loc='upper right', framealpha=0.96, fontsize=9.5, edgecolor='#CCCCCC')

# Value labels on bars
def label_bars(rects):
    for r in rects:
        h = r.get_height()
        ax4.text(r.get_x() + r.get_width()/2, h + 2, f'{h}',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

label_bars(ab1)
label_bars(ab2)
label_bars(ab3)

# Clear latency footnote
ax4.text(0.5, -0.16, "*Note: Latency Efficiency scores reflect log-scaled inverse mean runtimes: LangGraph ~12.0s (95), Config A 45.2s (78), Config C 175.7s (44).",
         transform=ax4.transAxes, ha='center', fontsize=8.5, style='italic', color='#555555')

plt.tight_layout(pad=3.0)

clean_tradeoff_png = os.path.join(output_dir, "tradeoff_clean_presentation.png")
clean_tradeoff_pdf = os.path.join(output_dir, "tradeoff_clean_presentation.pdf")
fig2.savefig(clean_tradeoff_png, dpi=300, bbox_inches='tight')
fig2.savefig(clean_tradeoff_pdf, dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"Generated: {clean_tradeoff_png}")
