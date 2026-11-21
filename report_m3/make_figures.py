"""Figure for the Milestone 3 report. Run from the repository root:
python report_m3/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'report_m3'
mae = pd.read_csv(ROOT/'results/two_stage/mae_by_origin.csv', parse_dates=['origin'])
stage1 = pd.read_csv(ROOT/'results/two_stage/stage1_by_issue.csv', parse_dates=['issue'])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
style = {'six_release_mean': ('Six-release mean', '#9a9a9a', ':'),
         'direct': ('Direct ridge', '#1f4e79', '-'),
         'two_stage': ('+ predicted releases', '#c0504d', '--'),
         'two_stage_batch': ('+ batch-predicted releases', '#e3a33b', '--'),
         'oracle': ('+ actual releases', '#4f8a3c', '-.')}
for col, (label, color, ls) in style.items():
    ax[0].plot(mae.origin, mae[col], ls, color=color, marker='o', ms=3, lw=1.4, label=label)
ax[0].set_ylabel('Adoption MAE (pp)')
ax[0].set_title('(a) Error at each issue date', fontsize=10, loc='left')
ax[0].set_ylim(2.75, 4.6)
ax[0].legend(fontsize=7, frameon=False, ncol=2, loc='upper right')
ax[0].tick_params(axis='x', labelrotation=30, labelsize=7)

done = stage1[stage1.realized_future_count.notna()]
ax[1].bar(done.issue, done.realized_future_count, width=9, color='#c9d6e3', label='Actual releases')
ax[1].plot(stage1.issue, stage1.expected_future_count, color='#c0504d', lw=1.6, label='Poisson forecast')
ax[1].plot(stage1.issue, stage1.expected_batch_count, color='#e3a33b', lw=1.6, ls='--', label='Compound Poisson forecast')
ax[1].set_ylabel('Releases in 24-day window')
ax[1].set_title('(b) Stage 1: forecast vs. actual release counts', fontsize=10, loc='left')
ax[1].set_ylim(0, 15)
ax[1].legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
ax[1].tick_params(axis='x', labelrotation=30, labelsize=7)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/'fig_two_stage.pdf')
fig.savefig(OUT/'fig_two_stage.png', dpi=200)
