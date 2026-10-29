"""Figures and table values for the final Milestone 2 report.

Reads saved outputs of extended_analysis.py run on the official Census workbook
(official_workbook_run/) and writes milestone2_final/figures/ and numbers.json.
Run from the repository root: python milestone2_final/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.dates
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'official_workbook_run'
OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
SIZES = {'A': '1–4', 'B': '5–9', 'C': '10–19', 'D': '20–49', 'E': '50–99'}
SHOW = {'trailing_mean': ('Six-release mean', '#8a8a8a', 'o'),
        'baseline': ('Pooled regression (baseline)', '#1f5a8a', 's'),
        'industry_models': ('Separate industry regressions', '#c0512b', '^')}
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False})

pred = pd.read_csv(RUN / 'predictions.csv', dtype={'Sector': str}, parse_dates=['origin'])
growth = pd.read_csv(RUN / 'industry_growth.csv').sort_values('growth_pp_per_4weeks')
perf = pd.read_csv(RUN / 'performance.csv')
paired = pd.read_csv(RUN / 'paired_comparisons.csv').set_index('model')

# Figure 1: industry growth with block-bootstrap intervals.
fig, ax = plt.subplots(figsize=(6.6, 2.3))
y = np.arange(len(growth))
ax.errorbar(growth.growth_pp_per_4weeks, y,
            xerr=[growth.growth_pp_per_4weeks - growth.ci_low, growth.ci_high - growth.growth_pp_per_4weeks],
            fmt='o', ms=3.5, color='#1f5a8a', ecolor='#7fa3c2', capsize=2, lw=1)
ax.set_yticks(y, growth.industry, fontsize=7.5)
ax.set_xlabel('Change in AI-use rate, percentage points per four weeks (95% interval)')
ax.set_xlim(0, None)
ax.grid(axis='x', color='.92')
fig.tight_layout()
fig.savefig(OUT / 'fig1_industry_growth.pdf')

# Figure 2: (a) MAE by issue date, (b) MAE by size band, each against survey noise.
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={'width_ratios': [1.25, 1]})
by_date = pred.groupby('origin')
for m, (label, color, marker) in SHOW.items():
    mae = by_date.apply(lambda d: np.abs(d[m] - d.target).mean(), include_groups=False)
    a.plot(mae.index, mae.values, marker=marker, ms=3.5, color=color, lw=1.2, label=label)
a.axhline(pred.target_se.median(), ls='--', color='#b22', lw=.9)
a.text(pred.origin.min(), pred.target_se.median() - .12, 'median standard error of target',
       color='#b22', fontsize=7)
last = matplotlib.dates.date2num(pred.origin.max())
a.axvspan(last - 5, last + 5, color='.9', zorder=0)
a.text(last, 4.22, 'target\n202620', ha='center', va='top', fontsize=6.5, color='.35')
a.set_ylabel('MAE (pp)')
a.set_title('(a) By forecast issue date (about 70 forecasts each)', fontsize=8.5, loc='left')
a.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b 2026'))
a.set_ylim(2.4, 4.3)
x = np.arange(len(SIZES))
for i, (m, (label, color, marker)) in enumerate(SHOW.items()):
    mae = pred.groupby('Empsize').apply(lambda d: np.abs(d[m] - d.target).mean(), include_groups=False)
    b.bar(x + (i - 1) * .26, mae.reindex(list(SIZES)).values, .26, color=color, label=label)
se = pred.groupby('Empsize').target_se.median().reindex(list(SIZES))
b.plot(x, se.values, 'D', color='#b22', ms=4, label='Median SE of target')
counts = pred.groupby('Empsize').size().reindex(list(SIZES))
b.set_xticks(x, [f'{SIZES[k]}\n(n={counts[k]})' for k in SIZES], fontsize=7)
b.set_xlabel('Employees')
b.set_title('(b) By employee-size band', fontsize=8.5, loc='left')
fig.tight_layout(rect=(0, .08, 1, 1))
fig.legend(*b.get_legend_handles_labels(), ncol=4, loc='lower center', frameon=False, fontsize=7.2)
fig.savefig(OUT / 'fig2_errors.pdf')

# Numbers quoted in the report, all from the official workbook run.
all_ = perf[perf.evaluation == 'all'].set_index('model')
ext = perf[perf.evaluation == 'extension'].set_index('model')
rows = {}
for m in ['persistence', 'trailing_mean', 'baseline', 'industry_models', 'boosting', 'full_ols']:
    r = {'mae': all_.loc[m, 'mae'], 'rmse': all_.loc[m, 'rmse'], 'bias': all_.loc[m, 'bias'],
         'within_2se': all_.loc[m, 'within_2se'], 'mae_202620': ext.loc[m, 'mae']}
    if m in paired.index:
        r.update(delta_vs_mean=paired.loc[m, 'mean_origin_mae_difference'],
                 ci=[paired.loc[m, 'block_ci_low'], paired.loc[m, 'block_ci_high']],
                 wins=int(paired.loc[m, 'wins']))
    rows[m] = {k: (np.round(v, 3).tolist() if not isinstance(v, int) else v) for k, v in r.items()}
summary = json.loads((RUN / 'summary.json').read_text())
numbers = {'forecasts': len(pred), 'origins': pred.origin.nunique(),
           'first_origin': str(pred.origin.min().date()), 'last_origin': str(pred.origin.max().date()),
           'median_target_se': round(float(pred.target_se.median()), 2),
           'median_target_se_by_size': se.round(2).to_dict(),
           'mae_by_size': {m: pred.groupby('Empsize').apply(lambda d: np.abs(d[m] - d.target).mean(),
                                                             include_groups=False).round(2).to_dict()
                           for m in SHOW},
           'cells': summary['potential_cells'], 'missing_rates': summary['suppressed_or_missing_rates'],
           'groups': summary['groups'], 'industries': summary['industries'],
           'complete_pairs': summary['complete_pairs'], 'models': rows}
(Path(__file__).resolve().parent / 'numbers.json').write_text(json.dumps(numbers, indent=2) + '\n')
print(json.dumps(numbers, indent=2))
