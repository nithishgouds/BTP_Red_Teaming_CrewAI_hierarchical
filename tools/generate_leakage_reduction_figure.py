"""
Refined Script for Figure 1: Empirical PII Leakage Mitigation across LLMs & Threat Classes.
Incorporates peer-review feedback:
1. Clarified Metric: 'Pre-Transmission PII Leakage Rate (%)' vs 'Actual Egress Transmission (0.0%)'.
2. Explicit Defense Breakdown: Attack Attempts (100%), Guardrail Redaction Rate (100.0%), Leakage (0.0%).
3. Sample Metadata & Denominators: N=45 live runs (15 per model), n=3 per threat class in Config A.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.2), dpi=300)

# ----------------------------------------------------------------------
# Panel (a): Pre-Transmission PII Leakage Rate by Model & Architecture (N=45)
# ----------------------------------------------------------------------
models = ['Gemma 2 (2B)', 'Llama 3.2 (3B)', 'Mistral (7B)']
x = np.arange(len(models))
width = 0.25

# Empirical Entity-Pooled Leakage Rates (%) from full benchmark CSVs
# Config A: Gemma 38.46% (15/39 PII entities exposed), Llama 16.67% (8/48), Mistral 14.29% (7/49)
config_a_leakage = [38.46, 16.67, 14.29]
config_b_leakage = [0.00, 0.00, 0.00]
config_c_leakage = [0.00, 0.00, 0.00]

rects1 = ax1.bar(x - width, config_a_leakage, width, 
                 label='Config A: Baseline (No Guardrail, Sequential)', 
                 color='#D9534F', edgecolor='black', linewidth=1)
rects2 = ax1.bar(x, config_b_leakage, width, 
                 label='Config B: + Inline Presidio Guardrail', 
                 color='#F0AD4E', edgecolor='black', linewidth=1)
rects3 = ax1.bar(x + width, config_c_leakage, width, 
                 label='Config C: + Guardrail + ChromaDB + Zero-Trust Supervisor', 
                 color='#2E7D32', edgecolor='black', linewidth=1)

ax1.set_ylabel("Pre-Transmission PII Leakage Rate (%)\n[Exposed Entities / Total Source PII]", fontsize=10.5, fontweight='bold', labelpad=8)
ax1.set_title("(a) PII Leakage Rate Across Architectures (N = 45 Live Runs)\n[15 runs per model: 5 threat classes × 3 configs]", fontsize=11, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10.5, fontweight='semibold')
ax1.set_ylim(0, 52)
ax1.grid(axis='y')
ax1.legend(loc='upper right', framealpha=0.95, fontsize=8.2)

# Value annotations on bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#D9534F')

for rect in rects2:
    ax1.annotate('0.00%\n(100% blocked)',
                xy=(rect.get_x() + rect.get_width() / 2, 0.5),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#B26A00')

for rect in rects3:
    ax1.annotate('0.00%\n(100% blocked)',
                xy=(rect.get_x() + rect.get_width() / 2, 0.5),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1B5E20')

# Highlight badge for 0% Actual Egress Transmission
ax1.text(0.5, 46.5, "★ Post-Defense Actual Egress Transmission: 0.00% across all 30 guarded runs (Config B & C)",
         ha="center", fontsize=8.5, fontweight='bold', color='#155724',
         bbox=dict(boxstyle="round,pad=0.35", fc="#D4EDDA", ec="#C3E6CB", lw=1))

# ----------------------------------------------------------------------
# Panel (b): Peak vs Mean Vulnerability by Threat Class in Config A (n=3 runs/class)
# ----------------------------------------------------------------------
threats = [
    'Delimiter Smuggling\n(n = 3 runs: 7 PII entities)',
    'Context Hijacking\n(n = 3 runs: 8 PII entities)',
    'Selective Exfiltration\n(n = 3 runs: 8 PII entities)',
    'Encoded Payload\n(n = 3 runs: 8 PII entities)',
    'Role Confusion\n(n = 3 runs: 8 PII entities)'
]
y_pos = np.arange(len(threats))
bar_height = 0.28

# Config A stats (n=3 runs per threat class across Gemma 2, Llama 3.2, Mistral)
config_a_threat_peak = [100.0, 75.0, 12.5, 12.5, 12.5]  # Worst-case model
config_a_threat_mean = [71.43, 33.33, 8.33, 8.33, 4.17]  # Macro-average across 3 models
guarded_threat_rate  = [0.00,  0.00,  0.00, 0.00, 0.00]  # Config B & C (All models)

rects_peak = ax2.barh(y_pos + bar_height, config_a_threat_peak, bar_height, 
                      label='Config A: Peak Model Leakage (Worst-case)', color='#C9302C', edgecolor='black', linewidth=1)
rects_mean = ax2.barh(y_pos, config_a_threat_mean, bar_height, 
                      label='Config A: Mean Leakage (Across 3 LLMs)', color='#F0AD4E', edgecolor='black', linewidth=1)
rects_guard = ax2.barh(y_pos - bar_height, guarded_threat_rate, bar_height, 
                       label='Configs B & C: Guarded (Inline Redaction + Supervisor)', color='#2E7D32', edgecolor='black', linewidth=1)

ax2.set_xlabel("PII Leakage Rate (% of Injected Entities Exposed)", fontsize=10.5, fontweight='bold', labelpad=8)
ax2.set_title("(b) Threat-Class Vulnerability Breakdown\n[Config A vs. Guarded Configs B & C]", fontsize=11, fontweight='bold', pad=10)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(threats, fontsize=9, fontweight='semibold')
ax2.set_xlim(0, 118)
ax2.grid(axis='x')
ax2.legend(loc='lower right', framealpha=0.95, fontsize=8)

for rect in rects_peak:
    w = rect.get_width()
    ax2.annotate(f'Peak: {w:.1f}%',
                xy=(w, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=8, fontweight='bold', color='#C9302C')

for rect in rects_mean:
    w = rect.get_width()
    ax2.annotate(f'Mean: {w:.1f}%',
                xy=(w, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=8, fontweight='bold', color='#B26A00')

for rect in rects_guard:
    ax2.annotate('0.00% (100% contained)',
                xy=(1, rect.get_y() + rect.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                ha='left', va='center', fontsize=7.5, fontweight='bold', color='#1B5E20')

# Caption Box
plt.figtext(
    0.5, 0.015,
    "Empirical Clarification: 'Pre-Transmission Leakage' measures PII present in intermediate agent responses before gateway filtering. In Config A, baseline leakage reached up to 100.0%.\nIn Configs B & C, 100.0% of adversarial attacks were actively executed (Canary present), but the Presidio Guardrail achieved 100.0% redaction, resulting in 0.00% exposure and 0.00% transmitted PII.",
    ha="center", fontsize=8, style="italic",
    bbox=dict(boxstyle="square,pad=0.5", fc="#F8F9FA", ec="#CED4DA", lw=1)
)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
png_path = os.path.join(output_dir, "leakage-reduction-empirical.png")
pdf_path = os.path.join(output_dir, "leakage-reduction-empirical.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Refined Figure 1 successfully generated:\n- {png_path}\n- {pdf_path}")
