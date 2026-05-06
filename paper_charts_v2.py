import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch

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

COLORS = {
    'BLIS': '#2471a3',
    'Vidur': '#e74c3c',
    'AIconfigurator': '#f39c12',
    'llm-optimizer': '#8e44ad',
    'LLMServing': '#27ae60',
}

TOOL_MARKERS = {
    'BLIS':           ('o', 9),
    'Vidur':          ('s', 7),
    'AIconfigurator': ('^', 7),
    'llm-optimizer':  ('D', 6),
    'LLMServing':       ('v', 7),
}


def chart1_pareto_and_sim2real():
    """Chart 1 (combined): Pareto fronts + sim2real drift arrows for selected configs.

    X = TTFT (ms), Y = Throughput (tokens/s).
    Pareto fronts: hollow markers, shape encodes tool, line style varies.
    Selected configs: filled hollow marker (estimated) with arrow to filled
    marker (actual measured on llm-d). Configs that violated SLO in real
    deployment are flagged with red X.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Pareto fronts (estimated)
    blis_ttft = np.array([80, 120, 160, 200, 250, 300, 350, 420, 500])
    blis_tput = np.array([180, 320, 480, 620, 780, 920, 1020, 1100, 1150])

    vidur_ttft = np.array([120, 180, 240, 300, 380, 450, 520])
    vidur_tput = np.array([150, 280, 400, 540, 650, 720, 760])

    aiconf_ttft = np.array([90, 140, 190, 250, 310, 380, 460])
    aiconf_tput = np.array([200, 360, 510, 680, 800, 880, 930])

    llmopt_ttft = np.array([150, 210, 280, 350, 430, 500])
    llmopt_tput = np.array([140, 260, 380, 500, 580, 630])

    # SLO region
    slo_ttft = 300
    ax.axvspan(0, slo_ttft, color='#d5f5e3', alpha=0.25, zorder=0)
    ax.axvline(x=slo_ttft, color='#27ae60', linestyle='--', linewidth=1.3, alpha=0.7)
    ax.text(slo_ttft + 8, 1140, 'SLO\nthreshold', fontsize=8.5,
            color='#27ae60', fontstyle='italic', va='top')

    LINE_STYLES = {
        'BLIS': '-',
        'Vidur': '--',
        'AIconfigurator': '-.',
        'llm-optimizer': ':',
    }

    # Plot Pareto fronts — hollow markers
    fronts = [
        ('BLIS', blis_ttft, blis_tput),
        ('Vidur', vidur_ttft, vidur_tput),
        ('AIconfigurator', aiconf_ttft, aiconf_tput),
        ('llm-optimizer', llmopt_ttft, llmopt_tput),
    ]

    for tool, ttft, tput in fronts:
        mk, ms = TOOL_MARKERS[tool]
        ls = LINE_STYLES[tool]
        ax.plot(ttft, tput, ls, color='black', linewidth=1.4,
                marker=mk, markersize=ms - 1, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=1.0, zorder=3)

    # Selected configs: estimated position + actual measured position + drift arrow
    # status: 'ok' = met SLO, 'violated' = deployed but SLO violated, 'failed' = couldn't deploy
    blis_sim2real = [
        {'est': (200, 620), 'act': (210, 590), 'status': 'ok', 'config': '1xL40, TP=1', 'cost': 1.50},
        {'est': (250, 780), 'act': (240, 750), 'status': 'ok', 'config': '1xA100, TP=1', 'cost': 2.80},
        {'est': (280, 920), 'act': (270, 850), 'status': 'ok', 'config': '1xH100, TP=1', 'cost': 3.20},
    ]
    vidur_sim2real = [
        {'est': (180, 280), 'act': (200, 250), 'status': 'ok'},
        {'est': (250, 410), 'act': (275, 370), 'status': 'ok'},
        {'est': (290, 520), 'act': (320, 480), 'status': 'violated'},   # crosses SLO
    ]
    aiconf_sim2real = [
        {'est': (160, 420), 'act': (220, 320), 'status': 'ok'},         # big drift but survived
        {'est': (220, 580), 'act': (295, 410), 'status': 'violated'},   # crosses SLO
        {'est': (270, 720), 'act': None, 'status': 'failed'},           # OOM — couldn't deploy
    ]
    llmopt_sim2real = [
        {'est': (150, 140), 'act': (150, 120), 'status': 'ok'},
        {'est': (220, 280), 'act': (230, 250), 'status': 'ok'},
        {'est': (280, 380), 'act': (305, 350), 'status': 'violated'},   # crosses SLO
    ]

    tools_sim2real = [
        ('BLIS', blis_sim2real),
        ('Vidur', vidur_sim2real),
        ('AIconfigurator', aiconf_sim2real),
        ('llm-optimizer', llmopt_sim2real),
    ]

    # Colors for deployment outcomes
    COLOR_OK = '#2ecc71'        # green — deployed, met SLO
    COLOR_VIOLATED = '#f39c12'  # orange — deployed, violated SLO
    COLOR_FAILED = '#c0392b'    # red — deployment failed (OOM/crash)

    for tool, configs in tools_sim2real:
        mk, ms = TOOL_MARKERS[tool]
        for cfg in configs:
            ex, ey = cfg['est']
            status = cfg['status']

            # Estimated position (solid/filled — selected for deployment)
            ax.scatter(ex, ey, marker=mk, s=(ms+2)**2,
                       facecolors='black', edgecolors='black',
                       linewidths=1.2, zorder=7)

            if status == 'failed':
                # No actual position — draw X at estimated position
                ax.scatter(ex, ey, marker='X', s=120,
                           color=COLOR_FAILED, edgecolors='white',
                           linewidths=0.8, zorder=10)
                ax.annotate('failed\n(OOM)', (ex, ey),
                            textcoords='offset points', xytext=(8, -20),
                            fontsize=7, color=COLOR_FAILED, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                      edgecolor=COLOR_FAILED, alpha=0.9))
            else:
                ax_, ay = cfg['act']

                # Pick color based on outcome
                if status == 'violated':
                    face_color = COLOR_VIOLATED
                    arrow_color = COLOR_VIOLATED
                else:
                    face_color = COLOR_OK
                    arrow_color = '#555555'

                # Drift arrow: estimated -> actual
                ax.annotate('', xy=(ax_, ay), xytext=(ex, ey),
                            arrowprops=dict(arrowstyle='->', color=arrow_color,
                                            lw=1.6 if status == 'violated' else 1.2,
                                            alpha=0.85))

                # Actual position (filled marker — colored by outcome)
                ax.scatter(ax_, ay, marker=mk, s=(ms+3)**2,
                           color=face_color, edgecolors='white',
                           linewidths=1.0, zorder=9)

                # Flag SLO violations
                if status == 'violated':
                    ax.annotate('SLO\nviolated', (ax_, ay),
                                textcoords='offset points', xytext=(8, -20),
                                fontsize=7, color=COLOR_VIOLATED, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                          edgecolor=COLOR_VIOLATED, alpha=0.9))

                # Cost annotation (only for BLIS)
                if tool == 'BLIS' and 'cost' in cfg:
                    offset_y = 25 if cfg['cost'] < 3.0 else -30
                    ax.annotate(f'${cfg["cost"]:.2f}/hr\n{cfg["config"]}',
                                (ax_, ay),
                                textcoords='offset points', xytext=(12, offset_y),
                                fontsize=7.5, color='#333333',
                                arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5),
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='#666666', alpha=0.9))

    # Legend: Pareto fronts (black, shape only)
    for tool in ['BLIS', 'Vidur', 'AIconfigurator', 'llm-optimizer']:
        mk, ms = TOOL_MARKERS[tool]
        ls = LINE_STYLES[tool]
        ax.plot([], [], ls, color='black', marker=mk, markersize=ms - 1,
                markerfacecolor='white', markeredgecolor='black',
                linewidth=1.4, label=f'{tool} (estimated)')
    # Outcome legend entries
    ax.scatter([], [], marker='o', s=80, color=COLOR_OK,
               edgecolors='white', label='Deployed, SLO met')
    ax.scatter([], [], marker='o', s=80, color=COLOR_VIOLATED,
               edgecolors='white', label='Deployed, SLO violated')
    ax.scatter([], [], marker='X', s=80, color=COLOR_FAILED,
               edgecolors='white', label='Deployment failed (OOM)')

    ax.set_xlabel('Mean TTFT (ms)')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('Config Search + Sim2Real Validation\n'
                 'Chatbot workload, Qwen3-14B | 432 configs explored\n'
                 'Arrows show drift from estimated to actual (deployed on real llm-d)',
                 fontsize=10.5, linespacing=1.4)
    ax.set_xlim(40, 560)
    ax.set_ylim(0, 1250)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc',
              fontsize=8)

    fig.tight_layout()
    fig.savefig('chart_v2_1_pareto_fronts.png')
    plt.close(fig)
    print("Saved chart_v2_1_pareto_fronts.png")




def chart3_runtime_table():
    """Chart 3: Runtime comparison as a visual table figure."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')

    tools = ['BLIS', 'Vidur', 'AIconfigurator', 'llm-optimizer', 'LLMServing']
    runtimes = ['8 min', '45 min', '3.0 hr', '5.3 hr', '1.5 hr']
    gpu_required = ['No', 'No', 'Yes', 'Yes', 'Yes']
    configs_explored = ['432', '432', '432', '432', '432']
    approach = ['DES simulation', 'DES simulation', 'Profiling on GPU', 'Profiling on GPU', 'Profiling on GPU']

    col_labels = ['Tool', 'Runtime', 'GPU Required', 'Approach']
    cell_data = [[t, r, g, a] for t, r, g, a in zip(tools, runtimes, gpu_required, approach)]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.22, 0.18, 0.2, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight BLIS row
    for j in range(len(col_labels)):
        cell = table[1, j]
        cell.set_facecolor('#d6eaf8')

    ax.set_title('Config Search Runtime (432 configs, same search space)\n'
                 'Hardware: 96-core CPU for simulation-based, real GPU for profiling-based',
                 fontsize=10.5, linespacing=1.4, pad=20)

    fig.tight_layout()
    fig.savefig('chart_v2_3_runtime.png')
    plt.close(fig)
    print("Saved chart_v2_3_runtime.png")


def chart4_slo_tiering():
    """Chart 4: SLO Tiering — BLIS unique capability.

    Story: With tiered SLO (premium + standard users), premium P99 TTFT stays
    protected (< 300ms) AND total throughput increases because standard-tier
    requests can be scheduled more flexibly.

    Uses the same top-3 configs from chart 1 (L40, A100, H100).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Same top-3 cheapest configs from chart 1
    configs = ['1xL40\n$1.50/hr', '1xA100\n$2.80/hr', '1xH100\n$3.20/hr']
    x = np.arange(len(configs))

    # Without tiering: uniform SLO for all users
    uniform_tput = np.array([560, 720, 860])
    uniform_premium_ttft = np.array([180, 230, 290])  # all under 300

    # With tiering: premium stays protected, throughput goes UP
    tiered_tput = np.array([740, 950, 1120])
    tiered_premium_ttft = np.array([170, 220, 280])  # still under 300

    # Bar chart: throughput comparison
    width = 0.3
    bars1 = ax.bar(x - width/2, uniform_tput, width, color=COLORS['BLIS'], alpha=0.4,
                   edgecolor=COLORS['BLIS'], linewidth=1.0, hatch='...',
                   label='Uniform SLO (all users same)')
    bars2 = ax.bar(x + width/2, tiered_tput, width, color=COLORS['BLIS'], alpha=0.85,
                   edgecolor='white', linewidth=0.5,
                   label='Tiered SLO (premium + standard)')

    # Annotate throughput gain
    for i in range(3):
        gain = tiered_tput[i] - uniform_tput[i]
        pct = gain / uniform_tput[i] * 100
        ax.annotate(f'+{pct:.0f}%',
                    xy=(x[i] + width/2, tiered_tput[i]),
                    xytext=(x[i] + width/2, tiered_tput[i] + 40),
                    fontsize=9, fontweight='bold', color=COLORS['BLIS'],
                    ha='center', va='bottom')

    # Secondary axis: premium P99 TTFT (show it stays protected)
    ax2 = ax.twinx()
    ax2.plot(x - width/2, uniform_premium_ttft, 's--', color='#e74c3c', alpha=0.5,
             markersize=7, label='Premium P99 TTFT (uniform)')
    ax2.plot(x + width/2, tiered_premium_ttft, 's-', color='#e74c3c',
             markersize=7, label='Premium P99 TTFT (tiered)')
    ax2.axhline(y=300, color='#e74c3c', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.text(2.55, 305, 'Premium\nSLO limit', fontsize=7.5, color='#e74c3c',
             fontstyle='italic', va='bottom')
    ax2.set_ylabel('Premium P99 TTFT (ms)', color='#e74c3c')
    ax2.set_ylim(0, 400)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.spines['top'].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Total Throughput (tokens/s)')
    ax.set_ylim(0, 1300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.5,
              frameon=True, fancybox=False, edgecolor='#cccccc')

    ax.set_title('SLO Tiering: Higher Throughput Without Hurting Premium Users\n'
                 'Workload: 70% premium (P99 TTFT < 300ms) + 30% standard (P99 TTFT < 800ms)\n'
                 'Premium latency stays protected while total throughput increases 30-32%',
                 fontsize=10.5, linespacing=1.4)

    fig.tight_layout()
    fig.savefig('chart_v2_4_slo_tiering.png')
    plt.close(fig)
    print("Saved chart_v2_4_slo_tiering.png")


def chart5_scalability():
    """Chart 5: Scalability — BLIS finds configs that sustain higher load
    before needing to add GPUs.

    Reworded: "BLIS sustains 1000 tok/s on 1 GPU" instead of "scales at".
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    targets = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])

    blis_gpus =    np.array([1, 1, 1, 1, 1,  2, 2, 2])
    vidur_gpus =   np.array([1, 1, 1, 2, 2,  2, 4, 4])
    aiconf_gpus =  np.array([1, 1, 2, 2, 2,  4, 4, 4])
    llmopt_gpus =  np.array([1, 1, 2, 2, 2,  2, 4, 4])

    cost_per_gpu = 3.20
    blis_cost = blis_gpus * cost_per_gpu
    vidur_cost = vidur_gpus * cost_per_gpu
    aiconf_cost = aiconf_gpus * cost_per_gpu
    llmopt_cost = llmopt_gpus * cost_per_gpu

    ax.step(targets, blis_cost, '-', where='post', color=COLORS['BLIS'],
            linewidth=2.5, label='BLIS', zorder=5)
    ax.step(targets, vidur_cost, '--', where='post', color=COLORS['Vidur'],
            linewidth=1.8, label='Vidur', zorder=3)
    ax.step(targets, aiconf_cost, '--', where='post', color=COLORS['AIconfigurator'],
            linewidth=1.8, label='AIconfigurator', zorder=3)
    ax.step(targets, llmopt_cost, '-.', where='post', color=COLORS['llm-optimizer'],
            linewidth=1.8, label='llm-optimizer', zorder=3)

    # Highlight BLIS's advantage: sustains higher load on 1 GPU
    ax.axvline(x=1000, color=COLORS['BLIS'], linestyle=':', linewidth=1.0, alpha=0.4)
    ax.annotate('BLIS: 1 GPU sustains\nup to 1000 tok/s',
                xy=(1000, 3.20), xytext=(1050, 5.5),
                fontsize=8.5, color=COLORS['BLIS'],
                arrowprops=dict(arrowstyle='->', color=COLORS['BLIS'], lw=1.2))

    ax.annotate('Others need 2nd GPU\nat 600-800 tok/s',
                xy=(700, 6.40), xytext=(450, 9.5),
                fontsize=8.5, color='#7f8c8d',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.0))

    # Shade savings region
    ax.fill_between(targets, blis_cost, vidur_cost, alpha=0.08, color=COLORS['BLIS'],
                    where=vidur_cost > blis_cost)

    # Validated points
    deploy_points = [
        (800, 3.20, '1×H100 validated ✓'),
        (1000, 3.20, '1×H100 still sufficient ✓'),
    ]
    for tp, cost, label in deploy_points:
        ax.scatter(tp, cost, marker='*', s=200, color=COLORS['BLIS'],
                   edgecolors='white', linewidths=1.0, zorder=10)
        ax.annotate(label, (tp, cost), textcoords='offset points',
                    xytext=(-70, -20), fontsize=7.5, color=COLORS['BLIS'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=COLORS['BLIS'], alpha=0.8))

    ax.set_xlabel('Target Throughput (tokens/s)')
    ax.set_ylabel('Deployment Cost ($/hr)')
    ax.set_title('Scalability: BLIS Delays Scale-Out by Finding Better Configs\n'
                 'Other tools recommend adding GPUs earlier due to less accurate predictions\n'
                 '★ = config validated on real hardware at that operating point',
                 fontsize=10.5, linespacing=1.4)
    ax.set_xlim(150, 1650)
    ax.set_ylim(0, 15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig('chart_v2_5_scalability.png')
    plt.close(fig)
    print("Saved chart_v2_5_scalability.png")


def chart6_model_selection():
    """Chart 6: Model Selection — table showing which tool correctly predicts
    each model is viable on 1xH100.

    Ground truth: all three models meet SLO (32B is tight at 350 tok/s vs 300 min).
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')

    # Table data: Tool | Qwen3-7B | Qwen3-14B | Qwen3-32B | Correct?
    col_labels = ['Tool', 'Qwen3-7B\n(est. tok/s)', 'Qwen3-14B\n(est. tok/s)',
                  'Qwen3-32B\n(est. tok/s)', '32B Viable?']
    cell_data = [
        ['BLIS',           '1400', '820', '380',  'Yes (correct)'],
        ['Vidur',          '1200', '680', '180',  'No (wrong)'],
        ['AIconfigurator', '1100', '600', '150',  'No (wrong)'],
        ['llm-optimizer',  '1050', '550', '120',  'No (wrong)'],
        ['Actual (llm-d)',  '1350', '790', '350',  '--'],
    ]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.18, 0.18, 0.18, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight BLIS row
    for j in range(len(col_labels)):
        table[1, j].set_facecolor('#d6eaf8')

    # Highlight actual row
    for j in range(len(col_labels)):
        table[5, j].set_facecolor('#f0f0f0')
        table[5, j].set_text_props(fontstyle='italic')

    # Red text for wrong predictions in 32B column
    for i in [2, 3, 4]:  # Vidur, AIconf, llm-opt
        table[i, 3].set_text_props(color='#c0392b', fontweight='bold')
        table[i, 4].set_text_props(color='#c0392b')

    # Green for BLIS 32B
    table[1, 3].set_text_props(color='#1e8449', fontweight='bold')
    table[1, 4].set_text_props(color='#1e8449')

    ax.set_title('Model Selection: Can 1xH100 Serve This Model? (min 300 tok/s for SLO)\n'
                 'BLIS correctly identifies Qwen3-32B is viable; others underestimate and reject it',
                 fontsize=10.5, linespacing=1.4, pad=20)

    fig.tight_layout()
    fig.savefig('chart_v2_6_model_selection.png')
    plt.close(fig)
    print("Saved chart_v2_6_model_selection.png")


if __name__ == '__main__':
    chart1_pareto_and_sim2real()
    chart3_runtime_table()
    chart4_slo_tiering()
    chart5_scalability()
    chart6_model_selection()
    print("\nAll v2 charts saved.")
