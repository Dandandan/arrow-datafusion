#!/usr/bin/env python3
"""Generate a visualization for TPC-DS SF1 benchmark results."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Benchmark data
queries = list(range(1, 100))
main_ms = [
    42.22, 133.44, 111.01, 1104.06, 162.65, 1145.26, 319.76, 108.82, 125.71,
    103.50, 710.07, 40.30, 361.81, 893.70, 13.86, 38.02, 206.57, 118.91,
    146.93, 11.70, 16.03, 415.75, 939.03, 357.67, 314.03, 79.64, 316.79,
    149.34, 256.66, 39.82, 171.04, 55.06, 129.34, 99.61, 102.28, 194.44,
    174.42, 80.00, 84.43, 98.13, 11.05, 104.14, 80.22, 8.33, 44.07, 206.07,
    589.20, 263.76, 235.66, 181.45, 165.34, 104.75, 98.23, 135.31, 103.76,
    131.38, 152.75, 256.74, 166.18, 132.09, 166.03, 775.25, 97.95, 660.01,
    223.08, 214.53, 288.50, 256.25, 102.88, 308.17, 125.04, 746.94, 95.08,
    432.55, 254.84, 123.12, 178.05, 315.84, 205.01, 298.42, 24.99, 189.77,
    34.60, 47.93, 142.95, 34.52, 81.27, 92.58, 113.10, 20.15, 60.61, 55.79,
    165.94, 57.39, 111.41, 67.67, 109.64, 139.54, 9004.76,
]
morsel_ms = [
    21.08, 43.58, 25.22, 278.49, 51.92, 828.29, 72.79, 31.98, 117.33,
    39.05, 184.72, 15.96, 83.50, 488.09, 10.90, 20.96, 66.17, 50.96,
    36.40, 10.29, 9.81, 76.58, 762.67, 176.08, 87.84, 31.99, 68.56,
    138.38, 78.55, 25.10, 86.21, 18.07, 43.85, 31.28, 43.49, 48.52,
    38.71, 42.35, 43.15, 47.16, 12.37, 22.79, 18.57, 7.00, 22.79, 51.90,
    148.10, 65.21, 73.40, 68.00, 71.58, 23.54, 26.19, 48.11, 22.81,
    40.95, 60.97, 103.95, 62.00, 44.24, 58.42, 766.45, 25.64, 348.97,
    59.59, 134.21, 194.77, 57.17, 34.51, 106.65, 39.13, 557.26, 28.65,
    127.16, 152.03, 49.55, 54.83, 118.52, 50.57, 100.31, 19.25, 43.50,
    22.53, 19.23, 51.93, 15.82, 45.49, 96.94, 31.71, 14.58, 26.27, 19.16,
    48.46, 22.72, 59.59, 18.63, 36.25, 31.26, 9456.70,
]

speedup = [m / p for m, p in zip(main_ms, morsel_ms)]

# --- Figure with 2 subplots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1.2]})
fig.suptitle('TPC-DS SF1 Benchmark: main vs morsel-driven parquet execution',
             fontsize=16, fontweight='bold', y=0.98)

# ── Top chart: execution times (log scale bar chart) ──
x = np.arange(len(queries))
width = 0.38

bars1 = ax1.bar(x - width/2, main_ms, width, label='main', color='#4C72B0', alpha=0.85)
bars2 = ax1.bar(x + width/2, morsel_ms, width, label='morsel-driven', color='#DD8452', alpha=0.85)

ax1.set_yscale('log')
ax1.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([str(q) for q in queries], fontsize=7, rotation=90)
ax1.set_xlabel('TPC-DS Query', fontsize=12)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3, which='both')
ax1.set_xlim(-0.7, len(queries) - 0.3)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

# Summary annotation
ax1.annotate(
    f'Total: main 29,564 ms  vs  morsel 18,515 ms\n'
    f'95 faster · 2 slower · 2 no change',
    xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
    fontsize=10, fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9))

# ── Bottom chart: speedup factor ──
colors = ['#55a868' if s > 1.05 else '#c44e52' if s < 0.95 else '#8c8c8c' for s in speedup]
ax2.bar(x, speedup, color=colors, width=0.7, alpha=0.85)
ax2.axhline(y=1.0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Speedup (×)', fontsize=12)
ax2.set_xlabel('TPC-DS Query', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([str(q) for q in queries], fontsize=7, rotation=90)
ax2.set_xlim(-0.7, len(queries) - 0.3)
ax2.grid(axis='y', alpha=0.3)

# Color legend for speedup
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#55a868', alpha=0.85, label='Faster'),
    Patch(facecolor='#c44e52', alpha=0.85, label='Slower'),
    Patch(facecolor='#8c8c8c', alpha=0.85, label='No change'),
]
ax2.legend(handles=legend_elements, fontsize=10, loc='upper left')

# Mark peak speedups
top_indices = sorted(range(len(speedup)), key=lambda i: speedup[i], reverse=True)[:3]
for i in top_indices:
    ax2.annotate(f'{speedup[i]:.1f}×', xy=(i, speedup[i]),
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2d6a2d')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/user/arrow-datafusion/tpcds_benchmark.png', dpi=150, bbox_inches='tight')
print("Saved tpcds_benchmark.png")
