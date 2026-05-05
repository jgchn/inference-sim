import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# --- Pareto Frontiers: Throughput vs Latency (mean TTFT) ---

# Baseline: Uniform SLO (300ms for ALL users)
baseline_ttft = np.array([95, 130, 170, 210, 260, 300, 350, 410])
baseline_tput = np.array([320, 480, 620, 740, 850, 920, 980, 1020])

# Treatment: Tiered SLO (premium 300ms, standard 400ms)
treatment_ttft = np.array([90, 120, 150, 185, 220, 265, 300, 340])
treatment_tput = np.array([380, 580, 750, 900, 1020, 1100, 1150, 1180])

# Plot frontiers (hollow markers = estimated)
ax.plot(baseline_ttft, baseline_tput, 'o--', color='#999999',
        markersize=6, markerfacecolor='none', markeredgewidth=1.3,
        linewidth=1.8, label='Uniform SLO (300ms for all)', zorder=3)

ax.plot(treatment_ttft, treatment_tput, 's-', color='#2471a3',
        markersize=6, markerfacecolor='none', markeredgewidth=1.3,
        linewidth=2.2, label='Tiered SLO (premium 300ms, standard 400ms)', zorder=3)

# --- SLO threshold ---
ax.axvline(x=300, color='#666666', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(308, 1200, 'Premium SLO\nlimit (300ms)', fontsize=8.5,
        color='#666666', fontstyle='italic', va='top')

# --- Best configs at SLO boundary (estimated) — filled versions of same shape ---
# Baseline best estimated: 920 tok/s at 300ms (filled circle)
ax.plot(300, 920, 'o', color='#999999', markersize=12,
        markerfacecolor='#999999', markeredgecolor='white',
        markeredgewidth=1.5, zorder=5)

# Treatment best estimated: 1150 tok/s at 300ms (filled square)
ax.plot(300, 1150, 's', color='#2471a3', markersize=12,
        markerfacecolor='#2471a3', markeredgecolor='white',
        markeredgewidth=1.5, zorder=5)

# --- Actual measured (llm-d) — same shape, darker fill, with drift arrows ---
# Baseline actual (filled circle, darker)
actual_base_ttft, actual_base_tput = 310, 900
ax.plot(actual_base_ttft, actual_base_tput, 'o', color='#555555',
        markersize=12, markerfacecolor='#555555', markeredgecolor='white',
        markeredgewidth=1.5, zorder=6)
ax.annotate('', xy=(actual_base_ttft, actual_base_tput),
            xytext=(300, 920),
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1.5))

# Treatment actual (filled square, darker)
actual_treat_ttft, actual_treat_tput = 295, 1080
ax.plot(actual_treat_ttft, actual_treat_tput, 's', color='#1a5276',
        markersize=12, markerfacecolor='#1a5276', markeredgecolor='white',
        markeredgewidth=1.5, zorder=6)
ax.annotate('', xy=(actual_treat_ttft, actual_treat_tput),
            xytext=(300, 1150),
            arrowprops=dict(arrowstyle='->', color='#2471a3', lw=1.5))

# --- Dashed lines showing estimated vs actual throughput levels ---
ax.hlines(y=920, xmin=50, xmax=300, colors='#999999', linestyles='dashed',
          linewidth=0.8, alpha=0.4)
ax.hlines(y=1150, xmin=50, xmax=300, colors='#2471a3', linestyles='dashed',
          linewidth=0.8, alpha=0.4)
ax.hlines(y=actual_base_tput, xmin=50, xmax=actual_base_ttft, colors='#555555',
          linestyles='dotted', linewidth=0.8, alpha=0.5)
ax.hlines(y=actual_treat_tput, xmin=50, xmax=actual_treat_ttft, colors='#1a5276',
          linestyles='dotted', linewidth=0.8, alpha=0.5)

# --- Annotations: estimated vs actual gain ---
# Estimated gain bracket (left side)
est_gain_pct = (1150 - 920) / 920 * 100
ax.annotate('', xy=(65, 1150), xytext=(65, 920),
            arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.8))
ax.text(75, 1035, f'Estimated\n+{est_gain_pct:.0f}%',
        fontsize=9, color='#27ae60', fontweight='bold', va='center')

# Actual gain bracket (right side)
act_gain_pct = (actual_treat_tput - actual_base_tput) / actual_base_tput * 100
ax.annotate('', xy=(420, actual_treat_tput), xytext=(420, actual_base_tput),
            arrowprops=dict(arrowstyle='<->', color='#e67e22', lw=1.8))
ax.text(430, (actual_treat_tput + actual_base_tput) / 2,
        f'Actual\n+{act_gain_pct:.0f}%',
        fontsize=9, color='#e67e22', fontweight='bold', va='center')

# --- Legend ---
# Add legend entries for filled/actual markers
ax.plot([], [], 'o', color='#999999', markersize=8,
        markerfacecolor='#999999', markeredgecolor='white',
        label='Estimated best (at SLO boundary)')
ax.plot([], [], 'o', color='#555555', markersize=8,
        markerfacecolor='#555555', markeredgecolor='white',
        label='Actual (llm-d)')

# --- Shade region between frontiers ---
from numpy import interp
x_fill = np.linspace(90, 300, 50)
y_baseline = interp(x_fill, baseline_ttft, baseline_tput)
y_treatment = interp(x_fill, treatment_ttft, treatment_tput)
ax.fill_between(x_fill, y_baseline, y_treatment, alpha=0.07, color='#2471a3')

# --- Formatting ---
ax.set_xlabel('Mean TTFT p99 (ms)')
ax.set_ylabel('Total Throughput (tok/s)')
ax.set_xlim(50, 470)
ax.set_ylim(200, 1280)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='lower right', frameon=True, fancybox=False,
          edgecolor='#cccccc', fontsize=8.5)

ax.set_title(
    'SLO Tiering Shifts the Pareto Frontier\n'
    'Relaxing standard-tier SLO by 100ms — estimated +25% throughput, actual +20% (validated on llm-d)',
    fontsize=10.5, linespacing=1.4
)

fig.tight_layout()
fig.savefig('chart_v2_fig3_draft.png')
plt.close(fig)
print("Saved chart_v2_fig3_draft.png")
