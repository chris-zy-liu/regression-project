"""Compare the saved extract with a locally supplied official Census workbook."""
import argparse
from pathlib import Path

import pandas as pd

import extended_analysis as analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    args = parser.parse_args()
    official, official_dates, _ = analysis.original.load(args.workbook, last=analysis.MAX_PERIOD)
    extracted, extracted_dates, _, _ = analysis.load_extract(Path('data/btos_ai_extract_2026-10-15.csv'))
    keys = ['Sector', 'Empsize', 'period']
    paired = official.merge(extracted, on=keys, suffixes=('_official', '_extract'),
                            how='inner', validate='one_to_one')
    print(f'Compared {len(paired)} common sector-size-release cells.')
    for name in ['rate', 'expected', 'se']:
        x, y = paired[f'{name}_official'], paired[f'{name}_extract']
        both = x.notna() & y.notna()
        print(f'{name}: {both.sum()} numeric pairs; {(x.notna() != y.notna()).sum()} missingness mismatches; '
              f'maximum absolute difference {(x[both] - y[both]).abs().max():.10g}')
    dates = official_dates[['Publication Date']].join(
        extracted_dates[['Publication Date']], how='inner', lsuffix='_official', rsuffix='_extract')
    different = dates[dates['Publication Date_official'] != dates['Publication Date_extract']]
    print('Publication-date differences:')
    print(different.to_string() if len(different) else 'none')


if __name__ == '__main__':
    main()
