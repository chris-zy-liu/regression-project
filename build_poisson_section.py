"""Generate tables and figures for the group's Poisson-process report section."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'poisson_investigation'
REPORT = ROOT / 'poisson_report'
LABELS = {
    'constant': 'Constant model rate', 'trend': 'Time trend', 'weekday': 'Time + weekday',
    'batch_constant': 'Constant batch rate', 'batch_trend': 'Batches + time',
    'batch_weekday': 'Batches + time + weekday', 'batch_day_of_week': 'Separate weekday indicators',
    'batch_quadratic': 'Quadratic batch trend', 'batch_recent_365': '365-day rate history',
    'batch_recent_504': '504-day rate history', 'batch_recent_sizes': 'Recent batch sizes',
    'batch_provider_sizes': 'Provider-specific sizes', 'nested_selection': 'Earlier-window selection',
}


def main():
    REPORT.mkdir(exist_ok=True)
    summary = pd.read_csv(RESULTS / 'summary.csv').set_index('model')
    rows = []
    for name, label in LABELS.items():
        row = summary.loc[name]
        rows.append(f"{label} & {row.count_mae:.2f} & {row.count_nll:.2f} & "
                    f"{int(row.covered)}/6 & {row.mean_width:.1f} & {row.interval_score:.2f} " + r'\\')
    table = '\n'.join(rows)
    (REPORT / 'table.tex').write_text(table + '\n' + r'\bottomrule' + '\n', encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9})
    names = ['constant', 'weekday', 'batch_weekday', 'batch_provider_sizes', 'nested_selection']
    fig, axis = plt.subplots(figsize=(7.3, 2.7))
    axis.barh([LABELS[name] for name in names], summary.loc[names, 'expected_busy_days'], color='#305b76')
    axis.axvline(16, color='#a65038', linestyle='--', label='16 observed days')
    axis.set_xlabel('Expected days with at least two models, across six windows')
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(RESULTS / f'busy_days.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    selected = pd.read_csv(RESULTS / 'selected.csv')
    batch_gain = 100 * (1 - summary.loc['batch_weekday', 'count_mae'] / summary.loc['weekday', 'count_mae'])
    nested_gain = 100 * (1 - summary.loc['nested_selection', 'count_mae'] / summary.loc['weekday', 'count_mae'])
    print(f'Batch MAE reduction {batch_gain:.1f}%; nested selection reduction {nested_gain:.1f}%.')
    print(selected[['train_end', 'chosen']].to_string(index=False))


if __name__ == '__main__':
    main()
