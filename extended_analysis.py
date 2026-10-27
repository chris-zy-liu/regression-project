"""Industry heterogeneity, nonlinear selection and model complexity in BTOS.

Run: python extended_analysis.py --workbook data/sector_size.xlsx
All forecast selection and preprocessing use only outcomes published by the
outer issue date. Original scripts and deliverables remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Milestone2_Code_Group1'))
import analysis as original

SEED = 4031
H = 3
AS_OF = pd.Timestamp('2026-10-26')
HISTORICAL_LAST = 202619
MAX_PERIOD = 202621  # Ends before the next questionnaire collection regime.
OUT = ROOT / 'results'
SECTORS = {'11': 'Forestry, fishing and support', '21': 'Mining', '22': 'Utilities',
           '23': 'Construction', '31': 'Manufacturing', '44': 'Retail trade',
           '42': 'Wholesale trade', '48': 'Transportation and warehousing',
           '51': 'Information', '52': 'Finance and insurance', '53': 'Real estate',
           '54': 'Professional and technical services', '55': 'Management of companies',
           '56': 'Administrative and waste services', '61': 'Educational services',
           '62': 'Health care and social assistance', '71': 'Arts and recreation',
           '72': 'Accommodation and food services', '81': 'Other services'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')


def load_extract(path):
    """Publicly redistributed Census estimates, with explicit source limitations.

    Missing rows in this extract are unavailable published estimates. They are
    not assigned a more specific suppression reason. Collection dates are
    supplied; publication is the following Thursday in this contiguous window.
    """
    raw = pd.read_csv(path, dtype={'geography_code': str})
    raw = raw[(raw.question_wording == 'any of its business functions') &
              (raw.answer == 'Yes') & raw.survey_cycle.between(original.FIRST, MAX_PERIOD)].copy()
    assert not raw.duplicated(['geography_type', 'geography_code', 'question', 'survey_cycle']).any()
    cal = raw[['survey_cycle', 'collection_start', 'collection_end']].drop_duplicates()
    assert not cal.survey_cycle.duplicated().any()
    dates = cal.rename(columns={'survey_cycle': 'period'}).set_index('period')
    dates['Collection Start Date'] = pd.to_datetime(dates.collection_start)
    dates['Collection End Date'] = pd.to_datetime(dates.collection_end)
    dates['Reference End Date'] = dates['Collection Start Date']-pd.Timedelta(days=1)
    dates['Reference Start Date'] = dates['Reference End Date']-pd.Timedelta(days=13)
    dates['Publication Date'] = dates['Collection End Date']+pd.Timedelta(days=4)
    periods = sorted(dates.index)
    source = raw[raw.geography_type == 'sector_size'].copy()
    source[['Sector', 'Empsize']] = source.geography_code.str.split('|', expand=True, regex=False)
    sectors = sorted(source.Sector.unique())
    # Sector categories absent from the published extract can have no surviving
    # estimates. Restore all in-scope categories and multi-sector XX explicitly.
    sectors = sorted(set(sectors) | set(SECTORS) | {'XX'})
    idx = pd.MultiIndex.from_product([sectors, list(original.SIZES), periods], names=['Sector', 'Empsize', 'period'])
    panel = pd.DataFrame(index=idx).reset_index()
    for question, field in [('used AI in last two weeks', 'rate'),
                            ('expects to use AI in next six months', 'expected')]:
        q = source[(source.question == question) & source.Empsize.isin(original.SIZES)]
        vals = q[['Sector', 'Empsize', 'survey_cycle', 'estimate_pct', 'standard_error_pct']].rename(
            columns={'survey_cycle': 'period', 'estimate_pct': field, 'standard_error_pct': field+'_se'})
        panel = panel.merge(vals, on=['Sector', 'Empsize', 'period'], how='left', validate='one_to_one')
    panel = panel.rename(columns={'rate_se': 'se'})
    panel['published'] = panel.period.map(dates['Publication Date'])
    panel['step'] = panel.period.map(dict(zip(periods, range(len(periods)))))
    aggregate_fields = []
    for geo, key in [('national', None), ('sector', 'Sector'), ('size', 'Empsize')]:
        for question, measure in [('used AI in last two weeks', 'rate'),
                                  ('expects to use AI in next six months', 'expected')]:
            sub = raw[(raw.geography_type == geo) & (raw.question == question)]
            # Some publishers label employment-size aggregates differently.
            if sub.empty:
                continue
            field = geo+'_'+measure
            sub = sub.rename(columns={'survey_cycle': 'period', 'estimate_pct': field})
            keys = ['period']
            if key:
                sub = sub.rename(columns={'geography_code': key})
                keys.append(key)
            panel = panel.merge(sub[keys+[field]], on=keys, how='left', validate='many_to_one')
            aggregate_fields.append(field)
    return panel, dates, periods, aggregate_fields


def prepare(workbook, as_of=AS_OF):
    """Keep the revised-question series; never fill a target or suppressed rate."""
    extract = Path(workbook).suffix.lower() == '.csv'
    aggregate_fields = []
    if extract:
        panel, dates, periods, aggregate_fields = load_extract(workbook)
    else:
        panel, dates, periods = original.load(Path(workbook), last=MAX_PERIOD)
    dates = dates.copy()
    for col in dates.columns:
        if 'date' in str(col).lower():
            dates[col] = pd.to_datetime(dates[col], errors='coerce')
    panel['published'] = pd.to_datetime(panel['published'])
    panel = panel[panel.published <= as_of].copy()
    periods = sorted(panel.period.unique())
    panel['step'] = panel.period.map(dict(zip(periods, range(len(periods)))))
    # Retain collection and reference dates verbatim for event alignment audits.
    dates.to_csv(OUT / 'release_calendar.csv')
    pair, groups = original.build_pairs(panel[['Sector', 'Empsize', 'period', 'rate', 'expected', 'se', 'step', 'published']])
    panel = panel.sort_values(['Sector', 'Empsize', 'step']).reset_index(drop=True)
    keys = ['Sector', 'Empsize']
    for lag in [1, 2, 3, 6]:
        source = panel[keys + ['step', 'rate']].copy()
        source['step'] += lag
        source = source.rename(columns={'rate': f'lag{lag}'})
        panel = panel.merge(source, on=keys + ['step'], how='left', validate='one_to_one')
    grouped = panel.groupby(keys, sort=False)['rate']
    panel['mean6'] = grouped.transform(lambda x: x.rolling(6, min_periods=1).mean())
    panel['change1'] = panel.rate - panel.lag1
    panel['expected_gap'] = panel.expected - panel.rate
    panel['new_sample'] = (panel.period >= 202616).astype(float)
    panel['sin6'] = np.sin(2 * np.pi * panel.step / 6)
    panel['cos6'] = np.cos(2 * np.pi * panel.step / 6)
    # Additional non-AI questions: first-answer shares, chosen by content availability
    # at each issue release. Missing values are imputed within each training fold.
    descriptors = []
    estimates = pd.read_excel(workbook, 'Response Estimates') if not extract else pd.DataFrame()
    questions = sorted(pd.to_numeric(estimates['Question ID'], errors='coerce').dropna().astype(int).unique()) if not extract else []
    for q in questions:
        if q in [7, 24]:
            continue
        field = f'q{q}_answer1'
        extra = original.to_long(estimates, q, field, MAX_PERIOD)
        panel = panel.merge(extra, on=keys + ['period'], how='left', validate='one_to_one')
        row = estimates[(estimates['Question ID'] == q) & (estimates['Answer ID'] == 1)].iloc[0]
        descriptors.append({'feature': field, 'question': str(row.get('Question', '')),
                            'answer': str(row.get('Answer', ''))})
    pd.DataFrame(descriptors, columns=['feature', 'question', 'answer']).to_csv(OUT / 'additional_predictors.csv', index=False)
    panel['current_se'] = panel.se
    feature_names = ['lag1', 'lag2', 'lag3', 'lag6', 'mean6', 'change1', 'expected_gap',
                     'new_sample', 'sin6', 'cos6', 'current_se'] + aggregate_fields + [d['feature'] for d in descriptors]
    pair = pair.merge(panel[keys + ['step'] + feature_names], on=keys + ['step'], validate='one_to_one')
    pair['target_published'] = pd.to_datetime(pair.target_published)
    complete = pair.dropna(subset=['rate', 'expected', 'target']).copy().reset_index(drop=True)
    assert not panel.duplicated(keys + ['period']).any()
    assert panel.rate.dropna().between(0, 100).all()
    panel.to_csv(OUT / 'panel.csv', index=False)
    complete.to_csv(OUT / 'forecast_pairs.csv', index=False)
    return panel, complete, groups, dates, periods, feature_names


class Features:
    """Fold-specific imputations and scaling; categorical levels are early-screened."""
    def __init__(self, sectors, sizes, extra):
        self.sectors, self.sizes, self.extra = sectors, sizes, extra

    def raw(self, rows):
        x = pd.DataFrame(index=rows.index)
        x['rate'] = rows.rate / 100
        x['time'] = rows.step / 10
        for s in self.sectors[1:]:
            x[f'industry_{s}'] = (rows.Sector == s).astype(float)
        for s in self.sizes[1:]:
            x[f'size_{s}'] = (rows.Empsize == s).astype(float)
        blocks = {'base': list(x.columns)}
        x['expected'] = rows.expected / 100
        blocks['expected'] = ['expected']
        x['rate_sq'] = x.rate ** 2
        x['time_sq'] = x.time ** 2
        x['rate_time'] = x.rate * x.time
        blocks.update({'rate_sq': ['rate_sq'], 'time_sq': ['time_sq'], 'rate_time': ['rate_time']})
        x['expected_sq'] = x.expected ** 2
        x['rate_expected'] = x.rate * x.expected
        blocks.update({'expected_sq': ['expected_sq'], 'rate_expected': ['rate_expected']})
        blocks['industry_trend'], blocks['size_trend'], blocks['industry_rate'] = [], [], []
        for s in self.sectors[1:]:
            for kind, parent in [('industry_trend', 'time'), ('industry_rate', 'rate')]:
                col = f'{kind}_{s}'
                x[col] = x[f'industry_{s}'] * x[parent]
                blocks[kind].append(col)
        for s in self.sizes[1:]:
            col = f'size_trend_{s}'
            x[col] = x[f'size_{s}'] * x.time
            blocks['size_trend'].append(col)
        for col in self.extra:
            x[col] = rows[col] / 100 if col not in ['new_sample', 'sin6', 'cos6'] else rows[col]
        blocks['history'] = ['lag1', 'lag2', 'lag3', 'lag6', 'mean6', 'change1']
        blocks['survey'] = [c for c in self.extra if c.startswith('q')]
        blocks['cycle'] = ['sin6', 'cos6']
        blocks['sample'] = ['new_sample']
        return x, blocks

    def fit_transform(self, train, test):
        a, self.blocks = self.raw(train)
        b, _ = self.raw(test)
        self.columns = list(a.columns)
        # Explicit zero for an entirely unavailable feature, without peeking at test.
        empty = a.isna().all()
        a.loc[:, empty] = 0.
        b.loc[:, empty] = 0.
        imputer = SimpleImputer(strategy='median', add_indicator=False)
        za, zb = imputer.fit_transform(a), imputer.transform(b)
        indicators = [c for c in self.extra if a[c].isna().any()]
        if indicators:
            za = np.column_stack([za, a[indicators].isna().astype(float)])
            zb = np.column_stack([zb, b[indicators].isna().astype(float)])
            self.columns += [f'{c}_missing' for c in indicators]
        scaler = StandardScaler()
        self.a, self.b = scaler.fit_transform(za), scaler.transform(zb)
        self.imputer, self.scaler = imputer, scaler
        self.indices = {k: [self.columns.index(c) for c in v] for k, v in self.blocks.items()}
        return self.a, self.b


def least_squares(a, y, b):
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    coef, _, rank, _ = linalg.lstsq(a, y, lapack_driver='gelsy')
    return b @ coef, {'p': a.shape[1], 'rank': int(rank),
                     'train_rmse': float(np.sqrt(np.mean((a @ coef - y) ** 2)))}


def inner_splits(train, dates, periods):
    """Only earlier validation targets available at the outer origin are present."""
    usable = []
    for step in sorted(train.step.unique()):
        if step < 7:
            continue
        issue = dates.loc[periods[int(step)], 'Publication Date']
        a = train[train.target_published <= issue]
        b = train[train.step == step]
        if len(a) >= 100 and len(b) >= 20:
            assert a.step.max() <= step - H
            usable.append((a, b))
    # At most four distinct earlier origins, spanning the available history.
    if len(usable) > 4:
        usable = [usable[i] for i in np.unique(np.linspace(0, len(usable)-1, 4).astype(int))]
    return usable


SUBSET_BLOCKS = ['expected', 'rate_sq', 'time_sq', 'rate_time', 'expected_sq',
                 'rate_expected', 'industry_trend', 'size_trend', 'industry_rate']


def subset_candidates(indices, nonlinear=True):
    """Exhaust all allowed blocks with strong hierarchy (base effects mandatory)."""
    names = SUBSET_BLOCKS if nonlinear else ['expected', 'industry_trend', 'size_trend', 'industry_rate']
    candidates = []
    for flags in itertools.product([False, True], repeat=len(names)):
        selected = [n for n, flag in zip(names, flags) if flag]
        if any(n in selected for n in ['expected_sq', 'rate_expected']) and 'expected' not in selected:
            continue
        columns = indices['base'] + [i for n in selected for i in indices[n]]
        candidates.append((tuple(selected), columns))
    return candidates


def subset_choose(folds, sectors, sizes, extra, nonlinear=True):
    """Blocked validation loss; ties favor fewer fitted coefficients."""
    cache = []
    for a, b in folds:
        f = Features(sectors, sizes, extra)
        xa, xb = f.fit_transform(a, b)
        cache.append((xa, xb, a.target.to_numpy(), b.target.to_numpy(), f.indices))
    candidates = subset_candidates(cache[0][-1], nonlinear)
    scores = []
    for names, cols in candidates:
        losses = []
        for xa, xb, ya, yb, _ in cache:
            pred, _ = least_squares(xa[:, cols], ya, xb[:, cols])
            losses.append(np.abs(pred - yb).mean())
        scores.append((float(np.mean(losses)), len(cols), names))
    best = min(scores)
    return list(best[2]), best[0], len(candidates)


def ridge_predict(xa, y, xb, alpha):
    model = Ridge(alpha=alpha, solver='cholesky')
    model.fit(xa, y)
    return model.predict(xb)


def model_predict(kind, xa, y, xb, alpha):
    if kind == 'ridge':
        return ridge_predict(xa, y, xb, alpha)
    if kind == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=20000, tol=1e-5)
    elif kind == 'boosting':
        model = HistGradientBoostingRegressor(max_iter=150, learning_rate=.05,
                   max_leaf_nodes=alpha, min_samples_leaf=25, l2_regularization=10,
                   early_stopping=False, random_state=SEED)
    else:
        raise ValueError(kind)
    model.fit(xa, y)
    return model.predict(xb)


def tune(kind, folds, sectors, sizes, extra):
    choices = {'ridge': [.1, 1, 10, 100, 1000, 10000, 100000],
               'elastic_net': [.01, .1, .5, 1], 'boosting': [3, 7, 15]}[kind]
    loss = []
    for a, b in folds:
        f = Features(sectors, sizes, extra)
        xa, xb = f.fit_transform(a, b)
        loss.append([np.abs(model_predict(kind, xa, a.target, xb, alpha)-b.target).mean()
                     for alpha in choices])
    return choices[int(np.argmin(np.mean(loss, axis=0)))]


def partial_predict(train, test, sectors, sizes, alpha):
    """Ridge-penalized industry/size intercept and industry slope deviations.

    Global intercept/current rate/time coefficients are unpenalized. This is a
    penalized partial-pooling estimator, not a fitted random-effects likelihood.
    """
    def design(r):
        base = np.column_stack([np.ones(len(r)), r.rate / 100, r.step / 10])
        dev = [(r.Sector == s).to_numpy(float) for s in sectors]
        dev += [(r.Empsize == s).to_numpy(float) for s in sizes]
        dev += [(r.Sector == s).to_numpy(float) * r.step.to_numpy()/10 for s in sectors]
        return np.column_stack([base] + dev)
    a, b = design(train), design(test)
    penalty = np.diag([0.] * 3 + [alpha] * (a.shape[1]-3))
    coef = linalg.solve(a.T @ a + penalty, a.T @ train.target, assume_a='sym')
    return b @ coef


def forecast(pair, dates, periods, groups, extra):
    sectors, sizes = sorted({g[0] for g in groups}), sorted({g[1] for g in groups})
    results, selections, tuning, audits = [], [], [], []
    for step in range(10, len(periods)-H):
        issue = dates.loc[periods[step], 'Publication Date']
        train = pair[pair.target_published <= issue].copy()
        test = pair[pair.step == step].copy()
        if test.empty:
            continue
        assert train.step.max() <= step-H
        assert train.target_published.max() <= issue < test.target_published.min()
        folds = inner_splits(train, dates, periods)
        if not folds:
            raise ValueError(f'No legal inner validation at origin {step}')
        f = Features(sectors, sizes, extra)
        xa, xb = f.fit_transform(train, test)
        y = train.target.to_numpy()
        base = f.indices['base']
        test['persistence'] = test.rate
        test['trailing_mean'] = test.mean6
        test['baseline'], diag = least_squares(xa[:, base], y, xb[:, base])
        test['full_ols'], full_diag = least_squares(xa, y, xb)
        for label, nonlinear in [('subset_linear', False), ('subset_nonlinear', True)]:
            chosen, inner_mae, count = subset_choose(folds, sectors, sizes, extra, nonlinear)
            cols = base + [i for n in chosen for i in f.indices[n]]
            test[label], _ = least_squares(xa[:, cols], y, xb[:, cols])
            selections.append({'step': step, 'origin': issue, 'model': label,
                               'blocks': '|'.join(chosen), 'parameters': len(cols)+1,
                               'inner_mae': inner_mae, 'candidates': count})
        for kind in ['ridge', 'elastic_net', 'boosting']:
            alpha = tune(kind, folds, sectors, sizes, extra)
            test[kind] = model_predict(kind, xa, y, xb, alpha)
            tuning.append({'step': step, 'model': kind, 'parameter': alpha})
        partial_alphas = [.1, 1, 10, 100]
        losses = [[np.abs(partial_predict(a, b, sectors, sizes, alpha)-b.target).mean()
                   for alpha in partial_alphas] for a, b in folds]
        alpha = partial_alphas[int(np.argmin(np.mean(losses, axis=0)))]
        test['partial_pooling'] = partial_predict(train, test, sectors, sizes, alpha)
        tuning.append({'step': step, 'model': 'partial_pooling', 'parameter': alpha})
        test['industry_models'] = np.nan
        for sector in sectors:
            a = train[train.Sector == sector]
            ix = test.index[test.Sector == sector]
            if not len(ix):
                continue
            b = test.loc[ix]
            ff = Features([sector], sorted(a.Empsize.unique()), extra)
            aa, bb = ff.fit_transform(a, b)
            cols = ff.indices['base']
            pred, dd = least_squares(aa[:, cols], a.target, bb[:, cols])
            if dd['rank'] < dd['p'] or len(a) <= dd['p']:
                raise ValueError(f'Industry {sector} model is not estimable at step {step}')
            test.loc[ix, 'industry_models'] = pred
        test['origin'] = issue
        test['evaluation'] = np.where(test.target_period <= HISTORICAL_LAST, 'historical', 'extension')
        results.append(test)
        audits.append({'step': step, 'origin': issue, 'train_n': len(train), 'test_n': len(test),
                       'latest_training_target': train.target_published.max(),
                       'inner_origins': len(folds), 'baseline_p': diag['p'], 'baseline_rank': diag['rank'],
                       'full_p': full_diag['p'], 'full_rank': full_diag['rank'],
                       'full_train_rmse': full_diag['train_rmse']})
        print(f'Forecast origin {step}: train={len(train)}, test={len(test)}, nonlinear={selections[-1]["blocks"]}', flush=True)
    result = pd.concat(results, ignore_index=True)
    pd.DataFrame(selections).to_csv(OUT / 'subset_selection.csv', index=False)
    pd.DataFrame(tuning).to_csv(OUT / 'tuning.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'validation_audit.csv', index=False)
    result.to_csv(OUT / 'predictions.csv', index=False)
    return result


MODELS = ['persistence', 'trailing_mean', 'baseline', 'subset_linear', 'subset_nonlinear',
          'full_ols', 'ridge', 'elastic_net', 'boosting', 'partial_pooling', 'industry_models']


def ridge_ablations(pair, result, dates, periods, groups, extra):
    """Test the contribution of cycles, histories, aggregates, and curvature.

    Rechecks the full ridge penalty on a wider fixed grid. Every ablation is
    tuned independently on the same legal inner origins.
    """
    sectors, sizes = sorted({g[0] for g in groups}), sorted({g[1] for g in groups})
    penalties = [.1, 1, 10, 100, 1000, 10000, 100000]
    variants = ['full', 'no_cycle', 'no_history', 'no_aggregates', 'no_curvature']
    records, audits = [], []
    def columns(f, variant):
        excluded = set()
        if variant == 'no_cycle': excluded = {'sin6', 'cos6'}
        if variant == 'no_history': excluded = {'lag1', 'lag2', 'lag3', 'lag6', 'mean6', 'change1'}
        if variant == 'no_aggregates':
            excluded = {c for c in extra if c.startswith(('national_', 'sector_', 'size_'))}
        if variant == 'no_curvature':
            excluded = {'rate_sq', 'time_sq', 'expected_sq', 'rate_time', 'rate_expected'}
        return [i for i, c in enumerate(f.columns) if c not in excluded and c.removesuffix('_missing') not in excluded]
    for step in sorted(result.step.unique()):
        issue = dates.loc[periods[int(step)], 'Publication Date']
        train, test = pair[pair.target_published <= issue], pair[pair.step == step]
        folds = inner_splits(train, dates, periods)
        cached = []
        for a, b in folds:
            f = Features(sectors, sizes, extra); xa, xb = f.fit_transform(a, b)
            cached.append((f, xa, xb, a.target.to_numpy(), b.target.to_numpy()))
        f = Features(sectors, sizes, extra); xa, xb = f.fit_transform(train, test)
        for variant in variants:
            losses = []
            for ff, aa, bb, ya, yb in cached:
                cols = columns(ff, variant)
                losses.append([np.abs(ridge_predict(aa[:, cols], ya, bb[:, cols], alpha)-yb).mean() for alpha in penalties])
            alpha = penalties[int(np.argmin(np.mean(losses, axis=0)))]
            cols = columns(f, variant)
            pred = ridge_predict(xa[:, cols], train.target, xb[:, cols], alpha)
            if variant == 'full':
                result.loc[result.step == step, 'ridge'] = pred
            row = test[['Sector', 'Empsize', 'step', 'target']].copy()
            row['variant'], row['prediction'], row['alpha'] = variant, pred, alpha
            records.append(row)
        _, diag = least_squares(xa, train.target, xb)
        audits.append({'step': step, 'full_p': diag['p'], 'full_rank': diag['rank'],
                       'full_train_rmse': diag['train_rmse']})
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(OUT/'ridge_ablation_predictions.csv', index=False)
    summary = predictions.groupby('variant').apply(lambda d: pd.Series({
        'mae': np.abs(d.prediction-d.target).mean(),
        'rmse': np.sqrt(np.mean((d.prediction-d.target)**2)), 'n': len(d)}), include_groups=False)
    summary.to_csv(OUT/'ridge_ablations.csv')
    audit = pd.read_csv(OUT/'validation_audit.csv').drop(columns=['full_p','full_rank','full_train_rmse'], errors='ignore')
    audit = audit.merge(pd.DataFrame(audits), on='step', validate='one_to_one')
    audit.to_csv(OUT/'validation_audit.csv', index=False)
    tuning = pd.read_csv(OUT/'tuning.csv')
    for step in result.step.unique():
        alpha = predictions[(predictions.step==step) & (predictions.variant=='full')].alpha.iloc[0]
        tuning.loc[(tuning.step==step)&(tuning.model=='ridge'), 'parameter'] = alpha
    tuning.to_csv(OUT/'tuning.csv', index=False)
    result.to_csv(OUT/'predictions.csv', index=False)
    return result


def moving_blocks(n, rng, length=3):
    starts = rng.integers(0, n, size=int(np.ceil(n/length)))
    return np.concatenate([(s + np.arange(length)) % n for s in starts])[:n]


def summarize(result, bootstrap=2000):
    records = []
    for split, d in [('all', result)] + list(result.groupby('evaluation')):
        for model in MODELS:
            err = d[model] - d.target
            records.append({'evaluation': split, 'model': model, 'n': len(d),
                            'origins': d.origin.nunique(), 'mae': err.abs().mean(),
                            'rmse': np.sqrt(np.mean(err**2)), 'bias': err.mean(),
                            'within_2se': (err.abs() <= 2*d.target_se).mean(),
                            'outside_0_100': ((d[model] < 0) | (d[model] > 100)).mean()})
    perf = pd.DataFrame(records)
    perf.to_csv(OUT / 'performance.csv', index=False)
    by_origin = pd.DataFrame({m: (result[m]-result.target).abs().groupby(result.origin).mean() for m in MODELS})
    by_origin.to_csv(OUT / 'mae_by_origin.csv')
    for key in ['Sector', 'Empsize']:
        by = pd.DataFrame({m: (result[m]-result.target).abs().groupby(result[key]).mean() for m in MODELS})
        by.to_csv(OUT / f'mae_by_{key.lower()}.csv')
    rng = np.random.default_rng(SEED)
    comparisons = []
    for model in MODELS[2:]:
        diff = (by_origin[model] - by_origin.trailing_mean).to_numpy()
        samples = np.array([diff[moving_blocks(len(diff), rng)].mean() for _ in range(bootstrap)])
        # Exact paired sign-flip test on origin means; dependent origins limit interpretation.
        signs = np.array(list(itertools.product([-1, 1], repeat=len(diff))))
        p = np.mean(np.abs(signs @ diff / len(diff)) >= abs(diff.mean())-1e-12)
        comparisons.append({'model': model, 'mean_origin_mae_difference': diff.mean(),
                            'block_ci_low': np.quantile(samples, .025), 'block_ci_high': np.quantile(samples, .975),
                            'wins': int((diff < 0).sum()), 'origins': len(diff), 'signflip_p_exploratory': p})
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_comparisons.csv', index=False)
    additional = []
    for model, reference in [('industry_models', 'baseline'), ('elastic_net', 'baseline'),
                              ('subset_nonlinear', 'baseline'), ('subset_nonlinear', 'subset_linear')]:
        diff = (by_origin[model]-by_origin[reference]).to_numpy()
        draws = np.array([diff[moving_blocks(len(diff), rng)].mean() for _ in range(bootstrap)])
        additional.append({'model': model, 'reference': reference,
                           'difference': diff.mean(), 'ci_low': np.quantile(draws, .025),
                           'ci_high': np.quantile(draws, .975), 'wins': int((diff<0).sum())})
    pd.DataFrame(additional).to_csv(OUT/'additional_comparisons.csv', index=False)
    # MAE remains primary. Compare squared error to the published variance without
    # interpreting SE as a hard lower bound on forecast MAE.
    return perf, by_origin


def industry_trends(panel, groups, bootstrap=2000):
    """Fixed group intercepts, sector-specific trends, shared moving-block residuals.

    The complete-rate subset is used descriptively; uncertainty is conditional
    on this linear trend model and published-cell coverage, not design-based.
    """
    d = panel[[k in groups for k in zip(panel.Sector, panel.Empsize)]].dropna(subset=['rate']).copy()
    sectors = sorted(d.Sector.unique())
    g = d.Sector + '_' + d.Empsize
    ids = sorted(g.unique())
    t = d.step.to_numpy() / 2.0  # Four weeks = two releases.
    a = np.column_stack([(g == k).to_numpy(float) for k in ids] +
                        [(d.Sector == s).to_numpy(float)*t for s in sectors])
    coef = linalg.lstsq(a, d.rate.to_numpy(), lapack_driver='gelsy')[0]
    fitted, residual = a @ coef, d.rate.to_numpy()-a @ coef
    nsteps = int(d.step.max())+1
    # Wild block residual multipliers are shared by every group at a release.
    # Block length 6 reflects the rotating six-panel cycle and temporal dependence.
    rng = np.random.default_rng(SEED)
    pinv = linalg.pinv(a)
    draws = []
    for _ in range(bootstrap):
        offset = int(rng.integers(0, 6))
        block = (np.arange(nsteps)+offset)//6
        weights = rng.choice([-1., 1.], block.max()+1)[block]
        beta = pinv @ (fitted + residual*weights[d.step.to_numpy(int)])
        draws.append(beta[-len(sectors):])
    draws = np.array(draws)
    ranks = (-draws).argsort(axis=1).argsort(axis=1)+1
    table = pd.DataFrame({'sector': sectors, 'industry': [SECTORS.get(s, s) for s in sectors],
                          'growth_pp_per_4weeks': coef[-len(sectors):],
                          'ci_low': np.quantile(draws, .025, axis=0),
                          'ci_high': np.quantile(draws, .975, axis=0),
                          'median_rank': np.median(ranks, axis=0),
                          'rank_low': np.quantile(ranks, .025, axis=0),
                          'rank_high': np.quantile(ranks, .975, axis=0),
                          'bootstrap_top3_share': (ranks <= 3).mean(axis=0)})
    table = table.sort_values('growth_pp_per_4weeks', ascending=False)
    # Sensitivity to the annual sample replacement and heteroskedastic survey SEs.
    step_change = (d.period >= 202616).to_numpy(float)
    adjusted = linalg.lstsq(np.column_stack([a, step_change]), d.rate, lapack_driver='gelsy')[0]
    weights = 1/np.maximum(d.se.fillna(d.se.median()).to_numpy(), 1.)
    weighted = linalg.lstsq(a*weights[:, None], d.rate.to_numpy()*weights, lapack_driver='gelsy')[0]
    table['growth_sample_adjusted'] = table.sector.map(dict(zip(sectors, adjusted[len(ids):len(ids)+len(sectors)])))
    table['growth_se_weighted'] = table.sector.map(dict(zip(sectors, weighted[-len(sectors):])))
    table.to_csv(OUT / 'industry_growth.csv', index=False)
    pd.DataFrame(draws, columns=sectors).to_csv(OUT / 'industry_growth_draws.csv', index=False)
    diffs = []
    for i, s in enumerate(sectors):
        for j in range(i+1, len(sectors)):
            delta = draws[:, i]-draws[:, j]
            diffs.append({'sector_a': s, 'sector_b': sectors[j],
                          'difference': coef[-len(sectors)+i]-coef[-len(sectors)+j],
                          'ci_low': np.quantile(delta, .025), 'ci_high': np.quantile(delta, .975)})
    pd.DataFrame(diffs).to_csv(OUT / 'industry_growth_differences.csv', index=False)
    # Coverage and composition are explicit: equal group weight, no population weighting.
    d.groupby('Sector').agg(observations=('rate', 'size'), sizes=('Empsize', 'nunique')).to_csv(OUT/'industry_coverage.csv')
    return table


def nonlinear_curves(pair, groups, extra, dates, periods):
    selections = pd.read_csv(OUT/'subset_selection.csv').fillna({'blocks': ''})
    last = selections[selections.model == 'subset_nonlinear'].iloc[-1]
    step = int(last.step)
    issue = dates.loc[periods[step], 'Publication Date']
    train = pair[pair.target_published <= issue]
    sectors, sizes = sorted({g[0] for g in groups}), sorted({g[1] for g in groups})
    chosen = last.blocks.split('|') if last.blocks else []
    curves = []
    for sector in ['23', '44', '51']:
        observed = train[(train.Sector == sector) & (train.Empsize == 'A')]
        if observed.empty:
            continue
        representative = observed.iloc[-1].copy()
        for col in ['expected']+extra:
            representative[col] = observed[col].median()
        representative['step'] = step
        representative['rate'] = observed.rate.median()
        for variable in ['rate', 'expected']:
            grid = np.linspace(observed[variable].quantile(.05), observed[variable].quantile(.95), 60)
            rows = pd.DataFrame([representative]*len(grid)).reset_index(drop=True)
            rows[variable] = grid
            f = Features(sectors, sizes, extra)
            xa, xb = f.fit_transform(train, rows)
            cols = f.indices['base']+[i for name in chosen for i in f.indices[name]]
            pred, _ = least_squares(xa[:, cols], train.target, xb[:, cols])
            baseline, _ = least_squares(xa[:, f.indices['base']], train.target, xb[:, f.indices['base']])
            curves.append(pd.DataFrame({'sector': sector, 'variable': variable, 'value': grid,
                                         'prediction': pred, 'baseline': baseline,
                                         'current_fixed': representative.rate,
                                         'expected_fixed': representative.expected, 'issue_step': step}))
    result = pd.concat(curves, ignore_index=True)
    result.to_csv(OUT/'conditional_curves.csv', index=False)
    return result


def event_analysis(panel, groups, dates):
    """Align releases to reference windows; separate visual timing from causal effects."""
    events = pd.read_csv(ROOT / 'data/ai_releases.csv', parse_dates=['date'])
    refcols = [c for c in dates.columns if 'reference' in c.lower() and 'end' in c.lower()]
    # Fail explicitly rather than silently substitute publication dates.
    if not refcols:
        raise ValueError(f'No reference-end date column in {list(dates.columns)}')
    ref = dates[refcols[0]]
    d = panel[[k in groups for k in zip(panel.Sector, panel.Empsize)]].copy()
    wide = d.pivot(index='step', columns=['Sector', 'Empsize'], values='rate')
    # Use a constant fully observed group set so changing suppression does not
    # mechanically generate jumps in the event timeline.
    balanced = wide.dropna(axis=1)
    if balanced.empty:
        raise ValueError('No balanced groups for the release-event analysis')
    mean = balanced.mean(axis=1)
    series = pd.DataFrame({'step': mean.index, 'mean_rate': mean.values})
    periodmap = panel.drop_duplicates('step').set_index('step').period
    series['period'] = series.step.map(periodmap)
    series['reference_end'] = series.period.map(ref)
    series['published'] = series.period.map(dates['Publication Date'])
    series.to_csv(OUT / 'event_series.csv', index=False)
    records = []
    for _, event in events.iterrows():
        # Exclude windows straddling the event: whole reference fortnights only.
        pre = series[series.reference_end < event.date].tail(3)
        post = series[series.reference_end-pd.Timedelta(days=13) >= event.date].head(3)
        supported = len(pre) == 3 and len(post) == 3
        record = {'model': event.model, 'date': event.date, 'source': event.source,
                  'balanced_groups': len(balanced.columns), 'pre_releases': len(pre),
                  'post_releases': len(post), 'supported_window': supported,
                  'mean_change_pp': float(post.mean_rate.mean()-pre.mean_rate.mean()) if supported else None}
        records.append(record)
    pd.DataFrame(records).to_csv(OUT / 'release_event_windows.csv', index=False)
    # A pooled event exposure predictor is evaluated as an exploratory association,
    # not used to retroactively improve the original forecasting specifications.
    exposure = np.array([(events.date <= r).sum() for r in series.reference_end], dtype=float)
    x = np.column_stack([np.ones(len(series)), series.step, (series.period >= 202616).astype(float), exposure])
    fit = linalg.lstsq(x, series.mean_rate, lapack_driver='gelsy')[0]
    assoc = {'balanced_groups': len(balanced.columns), 'event_coefficient_pp': float(fit[-1]),
             'design_rank': int(np.linalg.matrix_rank(x)),
             'correlation_exposure_time': float(np.corrcoef(exposure, series.step)[0, 1]),
             'interpretation': 'Descriptive association; selected overlapping events, no untreated counterfactual.'}
    write_json(OUT / 'event_association.json', assoc)
    return series, events


def complexity_experiment(pair, groups, extra, dates, periods, seeds=3):
    """Nested random Fourier dictionaries, fixed seeds, no outcome-guided widths.

    Width is scaled to each fold's n. Training features are standardized within
    the fold; random frequencies are outcome-independent. Kernel length scale is
    fixed at sqrt(input dimension). Intercept is included in reported p/rank.
    """
    sectors, sizes = sorted({g[0] for g in groups}), sorted({g[1] for g in groups})
    records = []
    ratios = [.05, .2, .5, .8, 1., 1.2, 2.]
    # Three origins span the evaluation interval. This is a capacity diagnostic,
    # not an additional model selected on the held-out outcomes.
    origins = sorted(pair[pair.step >= 10].step.unique())
    chosen = sorted(set([origins[0], origins[len(origins)//2], origins[-1]]))
    for step in chosen:
        issue = dates.loc[periods[int(step)], 'Publication Date']
        train = pair[pair.target_published <= issue]
        test = pair[pair.step == step]
        f = Features(sectors, sizes, extra)
        xa, xb = f.fit_transform(train, test)
        # Use only baseline and expected-use inputs so width, not changing input
        # information, drives this controlled experiment.
        cols = f.indices['base'] + f.indices['expected']
        xa, xb = xa[:, cols], xb[:, cols]
        y = train.target.to_numpy()
        yc = y-y.mean()
        n = len(train)
        widths = [max(2, int(round(r*n))-1) for r in ratios]
        for seed in range(seeds):
            rng = np.random.default_rng(SEED+seed)
            frequencies = rng.normal(size=(xa.shape[1], max(widths)))/np.sqrt(xa.shape[1])
            phases = rng.uniform(0, 2*np.pi, max(widths))
            aa, bb = np.cos(xa @ frequencies + phases), np.cos(xb @ frequencies + phases)
            for width in widths:
                a = aa[:, :width]*np.sqrt(2/width)
                b = bb[:, :width]*np.sqrt(2/width)
                center = a.mean(axis=0)
                a, b = a-center, b-center
                u, s, vt = linalg.svd(a, full_matrices=False, lapack_driver='gesdd')
                tol = max(a.shape)*np.finfo(float).eps*s[0]
                keep = s > tol
                rank = int(keep.sum())+1
                uy = u.T @ yc
                for method, alpha in [('minimum_norm', 0.), ('ridge', .01)]:
                    if alpha == 0:
                        weights = np.divide(uy, s, out=np.zeros_like(uy), where=keep)
                        df = rank
                    else:
                        weights = s*uy/(s*s+alpha)
                        df = float(np.sum(s*s/(s*s+alpha)))+1
                    coef = vt.T @ weights
                    train_pred = a @ coef+y.mean()
                    pred = b @ coef+y.mean()
                    records.append({'step': int(step), 'seed': seed, 'n': n, 'p': width+1,
                         'p_over_n': (width+1)/n, 'rank': rank, 'method': method,
                         'alpha': alpha, 'effective_df': df,
                         'train_rmse': np.sqrt(np.mean((train_pred-y)**2)),
                         'test_rmse': np.sqrt(np.mean((pred-test.target)**2)),
                         'test_mae': np.abs(pred-test.target).mean(),
                         'condition_number': float(s[0]/s[keep][-1]) if keep.any() else None})
            print(f'Complexity origin {step}, seed {seed+1}/{seeds}', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'complexity.csv', index=False)
    return result


def make_figures(result, perf, origins, growth, event_series, events, complexity, curves):
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figdir = OUT / 'figures'
    figdir.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(figdir / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(figdir / f'{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    labels = {'trailing_mean': 'Six-release mean', 'baseline': 'Baseline OLS',
              'subset_linear': 'Linear subset', 'subset_nonlinear': 'Nonlinear subset',
              'ridge': 'Ridge', 'elastic_net': 'Elastic net', 'boosting': 'Gradient boosting',
              'partial_pooling': 'Partial pooling', 'industry_models': 'Industry models',
              'full_ols': 'Full OLS', 'persistence': 'Persistence'}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    d = perf[perf.evaluation == 'all'].sort_values('mae')
    ax.barh([labels[m] for m in d.model], d.mae, color='#274b64')
    ax.invert_yaxis(); ax.set_xlabel('Forecast MAE (percentage points)')
    save(fig, 'forecast_comparison')
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for m in ['trailing_mean', 'baseline', 'subset_nonlinear', 'ridge', 'boosting']:
        ax.plot(origins.index, origins[m], marker='o', ms=3, label=labels[m])
    ax.set_ylabel('MAE (percentage points)'); ax.set_xlabel('Forecast issue date')
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.autofmt_xdate(); save(fig, 'errors_by_origin')
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(growth)*.25)))
    d = growth.iloc[::-1]
    ax.errorbar(d.growth_pp_per_4weeks, np.arange(len(d)),
                xerr=[d.growth_pp_per_4weeks-d.ci_low, d.ci_high-d.growth_pp_per_4weeks],
                fmt='o', ms=4, color='#274b64', capsize=2)
    ax.set_yticks(np.arange(len(d)), d.industry)
    ax.axvline(0, color='.6', lw=.8)
    ax.set_xlabel('Growth in percentage points per four weeks (95% block intervals)')
    save(fig, 'industry_growth')
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(growth)*.25)))
    for i, row in enumerate(growth.iloc[::-1].itertuples()):
        ax.plot([row.rank_low, row.rank_high], [i, i], color='#274b64')
        ax.plot(row.median_rank, i, 'o', color='#274b64', ms=4)
    ax.set_yticks(range(len(growth)), growth.iloc[::-1].industry)
    ax.set_xlabel('Growth rank: median and 95% bootstrap range (1 = fastest)')
    save(fig, 'industry_ranks')
    sel = pd.read_csv(OUT / 'subset_selection.csv')
    sel = sel[sel.model == 'subset_nonlinear']
    freq = pd.Series({b: sel.blocks.fillna('').str.split('|').map(lambda x: b in x).mean() for b in SUBSET_BLOCKS})
    freq.to_csv(OUT / 'selection_frequency.csv', header=['frequency'])
    fig, ax = plt.subplots(figsize=(7, 3))
    block_labels = {'expected':'Expected use', 'rate_sq':'Current use squared', 'time_sq':'Time squared',
                    'rate_time':'Current use x time', 'expected_sq':'Expected use squared',
                    'rate_expected':'Current x expected use', 'industry_trend':'Industry x time',
                    'size_trend':'Size x time', 'industry_rate':'Industry x current use'}
    ax.barh([block_labels[b] for b in freq.index], freq.values, color='#274b64'); ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of outer forecast origins selecting the block')
    save(fig, 'subset_stability')
    fig, axes = plt.subplots(2, curves.sector.nunique(), figsize=(7.3, 5.0), squeeze=False)
    for row, variable in enumerate(['rate', 'expected']):
        for ax, (sector, d) in zip(axes[row], curves[curves.variable==variable].groupby('sector')):
            ax.plot(d.value, d.prediction, color='#274b64', label='Selected subset')
            ax.plot(d.value, d.baseline, '--', color='.5', label='Baseline OLS')
            ax.set_title(SECTORS.get(sector, sector), fontsize=8)
            ax.set_xlabel('Current AI use (%)' if variable=='rate' else 'Expected AI use (%)')
        axes[row, 0].set_ylabel('Predicted use at t+3 (%)')
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, 'conditional_curves')
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(event_series.reference_end, event_series.mean_rate, 'o-', color='#274b64', ms=3)
    ymin, ymax = ax.get_ylim()
    for i, row in enumerate(events.itertuples()):
        ax.axvline(row.date, color='.65', lw=.7, ls='--')
        ax.text(row.date, ymin+(ymax-ymin)*(.1+.2*(i % 3)), row.model,
                rotation=90, fontsize=6.5, va='bottom')
    ax.set_ylabel('Mean AI-use rate (%)'); ax.set_xlabel('Survey reference-period end')
    fig.autofmt_xdate(); save(fig, 'release_timeline')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3))
    for method, color in [('minimum_norm', '#a64c3b'), ('ridge', '#274b64')]:
        d = complexity[complexity.method == method].copy()
        d['ratio'] = d.p_over_n.round(2)
        for ax, field in zip(axes, ['train_rmse', 'test_rmse']):
            med = d.groupby('ratio')[field].median()
            lo = d.groupby('ratio')[field].quantile(.1)
            hi = d.groupby('ratio')[field].quantile(.9)
            ax.plot(med.index, med, 'o-', color=color, ms=3, label=method.replace('_', ' '))
            ax.fill_between(med.index, lo, hi, color=color, alpha=.12)
            ax.axvline(1, color='.6', ls='--', lw=.7)
            ax.set_xlabel('Number of coefficients / training rows')
            ax.set_yscale('log')
    axes[0].set_ylabel('Training RMSE (pp, log scale)')
    axes[1].set_ylabel('Future RMSE (pp, log scale)')
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, 'complexity')


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', type=Path, default=ROOT/'data/sector_size.xlsx')
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--complexity-seeds', type=int, default=3)
    p.add_argument('--resume', action='store_true', help='Reuse forecasts only when the recorded input hash matches; regenerate diagnostics and figures.')
    args = p.parse_args()
    OUT = args.output
    OUT.mkdir(exist_ok=True, parents=True)
    if not args.workbook.exists():
        raise SystemExit('Workbook missing. Run python download_data.py or supply --workbook PATH. No results generated.')
    with threadpool_limits(limits=1):
        panel, pairs, groups, dates, periods, extra = prepare(args.workbook)
        quality = original.quality(panel)
        quality.to_csv(OUT/'data_quality.csv')
        if args.resume:
            saved = json.loads((OUT/'summary.json').read_text())
            if saved['sha256'] != hashlib.sha256(args.workbook.read_bytes()).hexdigest():
                raise ValueError('Cannot resume: input hash differs from the saved run')
            result = pd.read_csv(OUT/'predictions.csv', dtype={'Sector': str, 'Empsize': str},
                                 parse_dates=['published', 'target_published', 'origin'])
            check = result.merge(pairs[['Sector','Empsize','step','target']], on=['Sector','Empsize','step'],
                                  suffixes=('', '_check'), validate='one_to_one')
            assert len(check)==len(result) and np.allclose(check.target, check.target_check)
        else:
            result = forecast(pairs, dates, periods, groups, extra)
        result = ridge_ablations(pairs, result, dates, periods, groups, extra)
        perf, origins = summarize(result, args.bootstrap)
        growth = industry_trends(panel, groups, args.bootstrap)
        curves = nonlinear_curves(pairs, groups, extra, dates, periods)
        es, events = event_analysis(panel, groups, dates)
        complexity = pd.read_csv(OUT/'complexity.csv') if args.resume else complexity_experiment(pairs, groups, extra, dates, periods, args.complexity_seeds)
        make_figures(result, perf, origins, growth, es, events, complexity, curves)
    summary = {'workbook': str(args.workbook), 'sha256': hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
               'as_of': str(AS_OF.date()), 'periods': [int(x) for x in periods],
               'potential_cells': len(panel), 'suppressed_or_missing_rates': int(panel.rate.isna().sum()),
               'groups': len(groups), 'industries': len({g[0] for g in groups}), 'complete_pairs': len(pairs),
               'forecasts': len(result), 'origins': result.origin.nunique(),
               'bootstrap_draws': args.bootstrap, 'complexity_seeds': args.complexity_seeds,
               'median_target_se': float(result.target_se.median()),
               'source_type': 'Third-party Census extract' if args.workbook.suffix.lower()=='.csv' else 'Census workbook',
               'vintage_limitation': 'Original snapshot unavailable; this analysis uses the recorded downloaded vintage. Missing extract rows are unavailable estimates; their exact original flags cannot be recovered.',
               'extension_limitation': 'Later outcomes are retrospective extension tests; new methods were specified after their publication.'}
    write_json(OUT/'summary.json', summary)
    print(perf.to_string(index=False))
    print(growth.to_string(index=False))


if __name__ == '__main__':
    main()
