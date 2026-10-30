"""Benchmark-marked, nonhomogeneous Poisson release model and adoption forecasts.

python release_process.py --as-of 2026-10-30
The October 30 ECI vintage is retrospectively calibrated. Release-date filtering
does not reconstruct historical benchmark publication/availability vintages.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import extended_analysis as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/releases'
START = pd.Timestamp('2024-01-01')
SEED = 403130
RATE_MODELS = ['constant', 'linear', 'quadratic', 'linear_weekday', 'linear_annual']
ADOPTION_MODELS = ['control_ridge', 'control_boosting', 'counts_ridge', 'benchmark_ridge',
                   'nhpp_ridge', 'nhpp_interactions', 'nhpp_boosting']
EXTRA = ['lag1','lag2','lag3','lag6','mean6','change1','expected_gap','new_sample',
         'sin6','cos6','current_se','national_rate','national_expected',
         'sector_rate','sector_expected','size_rate','size_expected']
KNOWN = ['catalog_count42', 'eci_frontier', 'eci_mean42', 'eci_frontier_gain42',
         'gpqa_frontier', 'capability_stock']
FUTURE = ['expected_future_count', 'expected_mark_load', 'expected_frontier_gain']


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')


def load_catalog(as_of, scores=None, benchmarks=None):
    scores = Path(scores or ROOT/'data/eci_scores_2026-10-30.csv')
    benchmarks = Path(benchmarks or ROOT/'data/eci_benchmarks_2026-10-30.csv')
    d = pd.read_csv(scores, parse_dates=['date']).rename(columns={'Model':'model','Organization':'provider'})
    if d.model.duplicated().any():
        raise ValueError('Duplicate canonical ECI model records')
    if d[['date','eci']].isna().any().any():
        raise ValueError('Every event requires a release date and capability mark')
    b = pd.read_csv(benchmarks, parse_dates=['date','benchmark_release_date'])
    q = b[b.benchmark.str.lower() == 'gpqa diamond'].groupby('Model').performance.median()*100
    d['gpqa'] = d.model.map(q)
    # Missing providers are not collapsed into a single artificial release batch.
    d['provider'] = d.provider.fillna('Unknown: '+d.model)
    d = d[d.date <= pd.Timestamp(as_of)].sort_values(['date','provider','model']).reset_index(drop=True)
    assert d.eci.between(0, 300).all()
    return d


def design(dates, model):
    dates = pd.DatetimeIndex(dates)
    t = np.asarray((dates-START).days, dtype=float)/365.25
    columns = [np.ones(len(t))]
    names = ['intercept']
    if model != 'constant':
        columns.append(t); names.append('time_years')
    if model == 'quadratic':
        columns.append(t*t); names.append('time_years_squared')
    if model == 'linear_weekday':
        columns.append((dates.dayofweek < 5).astype(float)); names.append('weekday')
    if model == 'linear_annual':
        columns += [np.sin(2*np.pi*t), np.cos(2*np.pi*t)]
        names += ['annual_sine','annual_cosine']
    if model not in RATE_MODELS:
        raise ValueError(model)
    return np.column_stack(columns), names


def daily_counts(catalog, cutoff, batches=False):
    dates = pd.date_range(START, pd.Timestamp(cutoff), freq='D')
    eligible = catalog[catalog.date.between(START, cutoff)]
    if batches:
        eligible = eligible.drop_duplicates(['provider','date'])
    counts = eligible.groupby('date').size().reindex(dates, fill_value=0).to_numpy(float)
    return dates, counts


@dataclass
class RateFit:
    model: str
    coefficients: np.ndarray
    covariance: np.ndarray
    cutoff: pd.Timestamp
    pearson_dispersion: float
    residual_acf1: float
    loglike: float
    n: int
    event_count: int

    def intensity(self, dates):
        x, _ = design(dates, self.model)
        return np.exp(np.clip(x @ self.coefficients, -25, 15))

    def expected(self, start, end):
        if pd.Timestamp(end) < pd.Timestamp(start):
            return 0.
        return float(self.intensity(pd.date_range(start, end, freq='D')).sum())


def fit_rate(catalog, cutoff, model='linear', batches=False):
    cutoff = pd.Timestamp(cutoff)
    dates, y = daily_counts(catalog, cutoff, batches)
    x, _ = design(dates, model)
    if len(y) < 90 or y.sum() < 10:
        raise ValueError('Insufficient event history for rate regression')
    fit = sm.GLM(y, x, family=sm.families.Poisson()).fit(maxiter=100)
    mu = fit.predict(x)
    residual = (y-mu)/np.sqrt(mu)
    dispersion = float(np.sum(residual**2)/(len(y)-x.shape[1]))
    # HAC covariance addresses serial clustering approximately; it does not
    # make a misspecified Poisson predictive distribution correct.
    robust = sm.GLM(y, x, family=sm.families.Poisson()).fit(
        maxiter=100, cov_type='HAC', cov_kwds={'maxlags':7})
    return RateFit(model, np.asarray(fit.params), np.asarray(robust.cov_params()), cutoff,
                   dispersion, float(np.corrcoef(residual[:-1], residual[1:])[0,1]),
                   float(fit.llf), len(y), int(y.sum()))


def rate_cv(catalog, cutoff, batches=False):
    """Six disjoint 42-day validation windows, each with its own earlier fit."""
    cutoff = pd.Timestamp(cutoff)
    records = []
    for block in range(6):
        end = cutoff-pd.Timedelta(days=42*block)
        start = end-pd.Timedelta(days=41)
        train_end = start-pd.Timedelta(days=1)
        if (train_end-START).days < 365:
            continue
        testdates = pd.date_range(start, end)
        _, allcounts = daily_counts(catalog, end, batches)
        counts = allcounts[-42:]
        for model in RATE_MODELS:
            f = fit_rate(catalog, train_end, model, batches)
            mu = f.intensity(testdates)
            records.append({'cutoff':cutoff,'block':block,'model':model,'train_end':train_end,
                            'test_start':start,'test_end':end,'observed':int(counts.sum()),
                            'expected':float(mu.sum()),
                            'daily_nll':float(-stats.poisson.logpmf(counts,mu).mean()),
                            'count_abs_error':float(abs(counts.sum()-mu.sum()))})
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError('No earlier rate-validation blocks')
    chosen = result.groupby('model').daily_nll.mean().idxmin()
    return chosen, result


def fit_marks(catalog, cutoff):
    d = catalog[catalog.date.between(START, cutoff)].copy()
    x, _ = design(d.date, 'linear')
    fit = sm.OLS(d.eci.to_numpy(), x).fit()
    provider = pd.factorize(d.provider)[0]
    week = pd.factorize(d.date.dt.to_period('W'))[0]
    if len(np.unique(provider)) >= 2 and len(np.unique(week)) >= 2:
        covariance, _, _ = cov_cluster_2groups(fit, provider, week)
    else:
        covariance = fit.get_robustcov_results(cov_type='HC3').cov_params()
    # The mark distribution is across scored model releases, not just frontier models.
    residuals = np.asarray(fit.resid)
    return np.asarray(fit.params), np.asarray(covariance), residuals, len(d)


def psd(cov):
    eigen, vectors = np.linalg.eigh(cov)
    return (vectors*np.maximum(eigen, 0)) @ vectors.T


def simulate_marks(ratefit, catalog, cutoff, end, simulations=2000, parameter_uncertainty=False, dispersion=1.):
    """Independent Poisson increments, with regression-predicted capability marks.

    Counts arise from a Poisson process conditional on fitted parameters. With
    parameter draws, the reported interval is a predictive mixture. Same-day
    multiple arrivals are retained because event times are only dated to a day.
    """
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    dates = pd.date_range(cutoff+pd.Timedelta(days=1), end, freq='D')
    old = catalog[catalog.date <= cutoff]
    frontier = float(old.eci.max())
    if not len(dates):
        return np.zeros(simulations, int), np.full(simulations, frontier)
    beta, cov, residuals, _ = fit_marks(catalog, cutoff)
    xmark, _ = design(dates, 'linear')
    xrate, _ = design(dates, ratefit.model)
    rng = np.random.default_rng(SEED+int(cutoff.strftime('%Y%m%d')))
    counts = np.zeros(simulations, dtype=int)
    maxima = np.full(simulations, frontier)
    for i in range(simulations):
        br = rng.multivariate_normal(ratefit.coefficients, psd(ratefit.covariance)) if parameter_uncertainty else ratefit.coefficients
        bm = rng.multivariate_normal(beta, psd(cov)) if parameter_uncertainty else beta
        mu = np.exp(np.clip(xrate @ br, -25, 10))
        daily = rng.poisson(mu) if dispersion <= 1 else rng.negative_binomial(mu/(dispersion-1), 1/dispersion)
        counts[i] = daily.sum()
        if counts[i]:
            indices = np.repeat(np.arange(len(dates)), daily)
            marks = xmark[indices] @ bm+rng.choice(residuals, len(indices))
            maxima[i] = max(frontier, float(marks.max()))
    return counts, maxima


def history_features(catalog, issue):
    old = catalog[catalog.date <= issue]
    recent = old[old.date > issue-pd.Timedelta(days=42)]
    before = old[old.date <= issue-pd.Timedelta(days=42)]
    # A fixed ECI reference and decay scale define a capability-weighted index.
    # It is an index, not a count of economically equivalent releases.
    weight = np.maximum((old.eci.to_numpy()-100)/10, 0)
    age = (issue-old.date).dt.days.to_numpy()
    return {'catalog_count42':len(recent), 'eci_frontier':float(old.eci.max()),
            'eci_mean42':float(recent.eci.mean()) if len(recent) else float(old.eci.iloc[-1]),
            'eci_frontier_gain42':float(old.eci.max()-before.eci.max()) if len(before) else 0.,
            'gpqa_frontier':float(old.gpqa.max()) if old.gpqa.notna().any() else 0.,
            'capability_stock':float(np.sum(weight*np.exp(-age/90)))}


def make_issue_features(catalog, calendar, simulations=2000):
    records, fits, cvrecords = [], [], []
    for _, row in calendar.iterrows():
        issue = pd.Timestamp(row['Publication Date'])
        target_end = issue+pd.Timedelta(days=24)  # t+3 reference end = t publication +24 days.
        old = catalog[catalog.date <= issue]
        chosen, cv = rate_cv(old, issue)
        cvrecords.append(cv)
        f = fit_rate(old, issue, chosen)
        beta, _, mark_residuals, nmarks = fit_marks(old, issue)
        future = pd.date_range(issue+pd.Timedelta(days=1), target_end)
        mu = f.intensity(future)
        xm, _ = design(future, 'linear')
        markmean = xm @ beta
        counts, maxima = simulate_marks(f, old, issue, target_end, simulations)
        h = history_features(old, issue)
        h.update({'period':int(row.period), 'issue':issue, 'target_reference_end':target_end,
                  'expected_future_count':float(mu.sum()),
                  'expected_mark_load':float(np.sum(mu*np.maximum(
                      (markmean[:,None]+mark_residuals[None,:]-100)/10,0).mean(axis=1))),
                  'expected_frontier_gain':float((maxima-h['eci_frontier']).mean()),
                  'latest_catalog_date':old.date.max(),'rate_training_end':issue,
                  'rate_training_events':f.event_count,'mark_training_events':nmarks,'rate_model':chosen})
        records.append(h)
        fits.append({'issue':issue, 'model':chosen,'intercept':f.coefficients[0],
                     'slope_per_year':f.coefficients[1] if chosen!='constant' else 0.,
                     'daily_rate':float(f.intensity([issue])[0]), 'expected24':float(mu.sum())})
        print(f'Release features {issue.date()}: {f.event_count} catalog events, E[N24]={mu.sum():.2f}', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT/'issue_features.csv', index=False)
    pd.DataFrame(fits).to_csv(OUT/'rate_fits_by_issue.csv', index=False)
    pd.concat(cvrecords,ignore_index=True).to_csv(OUT/'rate_cv_by_issue.csv',index=False)
    return result


def adoption_design(train, test, model, sectors, sizes):
    f = previous.Features(sectors, sizes, EXTRA)
    xa, xb = f.fit_transform(train, test)
    if model in ['control_ridge','control_boosting']:
        return xa, xb
    columns = ['catalog_count42']
    if model != 'counts_ridge':
        columns = KNOWN.copy()
    if model in ['nhpp_ridge','nhpp_interactions','nhpp_boosting']:
        columns += FUTURE
    a, b = train[columns].to_numpy(float), test[columns].to_numpy(float)
    # Penalized sector interactions allow benchmark effects to differ by industry.
    if model == 'nhpp_interactions':
        interact = ['eci_frontier', 'capability_stock', 'expected_future_count', 'expected_frontier_gain']
        for sector in sectors[1:]:
            a = np.column_stack([a, train[interact].to_numpy()*(train.Sector==sector).to_numpy()[:,None]])
            b = np.column_stack([b, test[interact].to_numpy()*(test.Sector==sector).to_numpy()[:,None]])
        a = np.column_stack([a, train.eci_frontier.to_numpy()*train.rate.to_numpy()/100])
        b = np.column_stack([b, test.eci_frontier.to_numpy()*test.rate.to_numpy()/100])
    scale = StandardScaler()
    return np.column_stack([xa,scale.fit_transform(a)]), np.column_stack([xb,scale.transform(b)])


def fit_adoption(model, xa, y, xb, parameter):
    if model.endswith('boosting'):
        return previous.model_predict('boosting', xa, y, xb, parameter)
    return previous.ridge_predict(xa, y, xb, parameter)


def adoption_evaluate(features, simulations):
    pair = pd.read_csv(ROOT/'results/forecast_pairs.csv', dtype={'Sector':str,'Empsize':str},
                       parse_dates=['published','target_published'])
    calendar = pd.read_csv(ROOT/'results/release_calendar.csv', parse_dates=['Publication Date'])
    dates = calendar.set_index('period')
    periods = calendar.period.tolist()
    # Only periods for which an observed t+3 target exists enter the evaluation.
    pair = pair.merge(features[['period']+KNOWN+FUTURE], on='period', validate='many_to_one')
    sectors, sizes = sorted(pair.Sector.unique()), sorted(pair.Empsize.unique())
    predictions, tuning, audits = [], [], []
    for step in sorted(pair[pair.step >= 10].step.unique()):
        issue = dates.loc[periods[int(step)],'Publication Date']
        train = pair[pair.target_published <= issue]
        test = pair[pair.step == step].copy()
        folds = previous.inner_splits(train, dates, periods)
        assert train.target_published.max() <= issue < test.target_published.min()
        for model in ADOPTION_MODELS:
            parameters = [3,7,15] if model.endswith('boosting') else [.1,1,10,100,1000,10000,100000]
            losses=[]
            for a,b in folds:
                xa, xb = adoption_design(a,b,model,sectors,sizes)
                losses.append([np.abs(fit_adoption(model,xa,a.target,xb,v)-b.target).mean() for v in parameters])
            parameter = parameters[int(np.argmin(np.mean(losses,axis=0)))]
            xa, xb = adoption_design(train,test,model,sectors,sizes)
            test[model] = fit_adoption(model,xa,train.target,xb,parameter)
            tuning.append({'step':step,'model':model,'parameter':parameter,'p':xa.shape[1]+1})
        test['origin']=issue
        test['evaluation']=np.where(test.target_period<=previous.HISTORICAL_LAST,'historical','extension')
        predictions.append(test)
        audits.append({'step':step,'issue':issue,'training_n':len(train),'test_n':len(test),
                       'max_training_target':train.target_published.max(),'inner_origins':len(folds)})
        print(f'Benchmark/adoption outer origin {step}',flush=True)
    result=pd.concat(predictions,ignore_index=True)
    result.to_csv(OUT/'adoption_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(OUT/'adoption_tuning.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'adoption_audit.csv',index=False)
    records=[]
    for split,d in [('all',result)]+list(result.groupby('evaluation')):
        for m in ADOPTION_MODELS:
            error=d[m]-d.target
            records.append({'evaluation':split,'model':m,'n':len(d),'mae':error.abs().mean(),
                            'rmse':np.sqrt(np.mean(error**2)),'bias':error.mean()})
    perf=pd.DataFrame(records);perf.to_csv(OUT/'adoption_performance.csv',index=False)
    origins=pd.DataFrame({m:(result[m]-result.target).abs().groupby(result.origin).mean() for m in ADOPTION_MODELS})
    origins.to_csv(OUT/'adoption_mae_by_origin.csv')
    rng=np.random.default_rng(SEED)
    comparisons=[]
    for m in ADOPTION_MODELS[1:]:
        delta=(origins[m]-origins.control_ridge).to_numpy()
        draws=np.array([delta[previous.moving_blocks(len(delta),rng)].mean() for _ in range(2000)])
        comparisons.append({'model':m,'reference':'control_ridge','difference':delta.mean(),
                             'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),
                             'wins':int((delta<0).sum()),'origins':len(delta)})
    pd.DataFrame(comparisons).to_csv(OUT/'adoption_comparisons.csv',index=False)
    for m,ref in [('nhpp_ridge','benchmark_ridge'),('nhpp_interactions','nhpp_ridge'),('nhpp_boosting','control_boosting')]:
        delta=(origins[m]-origins[ref]).to_numpy()
        draws=np.array([delta[previous.moving_blocks(len(delta),rng)].mean() for _ in range(2000)])
        comparisons.append({'model':m,'reference':ref,'difference':delta.mean(),
                             'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),
                             'wins':int((delta<0).sum()),'origins':len(delta)})
    pd.DataFrame(comparisons).to_csv(OUT/'adoption_comparisons.csv',index=False)
    return result,perf


def rate_analysis(catalog, cutoff, simulations):
    chosen,cv=rate_cv(catalog,cutoff)
    cv.to_csv(OUT/'rate_temporal_validation.csv',index=False)
    scores=cv.groupby('model').agg(mean_daily_nll=('daily_nll','mean'),
                                   mean_42day_abs_error=('count_abs_error','mean'),blocks=('block','nunique'))
    scores.to_csv(OUT/'rate_model_comparison.csv')
    records=[]; coefficients=[]
    future=pd.date_range(cutoff+pd.Timedelta(days=1),periods=42)
    daily=[]
    for model in RATE_MODELS:
        f=fit_rate(catalog,cutoff,model)
        _,names=design([cutoff],model)
        se=np.sqrt(np.diag(f.covariance))
        records.append({'model':model,'events':f.event_count,'days':f.n,'daily_rate_at_cutoff':float(f.intensity([cutoff])[0]),
                         'expected_next42':float(f.intensity(future).sum()),'pearson_dispersion':f.pearson_dispersion,
                         'residual_acf1':f.residual_acf1,'aic':2*len(names)-2*f.loglike})
        for name,b,s in zip(names,f.coefficients,se):
            coefficients.append({'model':model,'term':name,'coefficient':b,'hac_se':s,'ci_low':b-1.96*s,'ci_high':b+1.96*s})
        for date,mu in zip(future,f.intensity(future)):
            daily.append({'date':date,'model':model,'lambda_per_day':mu})
    pd.DataFrame(records).to_csv(OUT/'rate_diagnostics.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'rate_coefficients.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'future_daily_intensity.csv',index=False)
    sensitivity=[]
    for batches in [False,True]:
        f=fit_rate(catalog,cutoff,'linear',batches)
        sensitivity.append({'unit':'provider_date_batches' if batches else 'canonical_models',
                            'events':f.event_count,'slope_per_year':f.coefficients[1],
                            'dispersion':f.pearson_dispersion,'expected42':float(f.intensity(future).sum())})
    pd.DataFrame(sensitivity).to_csv(OUT/'batch_sensitivity.csv',index=False)
    selected=fit_rate(catalog,cutoff,chosen)
    counts,frontiers=simulate_marks(selected,catalog,cutoff,future[-1],simulations,True)
    nbcounts,nbfrontiers=simulate_marks(selected,catalog,cutoff,future[-1],simulations,True,
                                      max(1.,selected.pearson_dispersion))
    np.savez_compressed(OUT/'future_simulations.npz',counts=counts,frontiers=frontiers,
                        overdispersed_counts=nbcounts,overdispersed_frontiers=nbfrontiers)
    param,markcov,residuals,nmarks=fit_marks(catalog,cutoff)
    dump(OUT/'mark_regression.json',{'intercept':float(param[0]),'eci_slope_per_year':float(param[1]),
         'slope_se_provider_week_clustered':float(np.sqrt(max(markcov[1,1],0))),
         'covariance':'Two-way provider and release-week clustered; PSD projection used for simulation if needed.',
         'residual_sd':float(residuals.std(ddof=2)),
         'releases':nmarks,'interpretation':'Regression of release marks, not the frontier trend.'})
    f=fit_rate(catalog,cutoff,'linear')
    dates,y=daily_counts(catalog,cutoff)
    mu=f.intensity(dates)
    weekly=pd.DataFrame({'observed':y,'expected':mu},index=dates).resample('W-SUN').sum()
    weekly.to_csv(OUT/'weekly_arrivals.csv')
    # Plug-in simulation diagnostic for daily batching and maximum count.
    rng=np.random.default_rng(SEED)
    sim=rng.poisson(mu,size=(2000,len(mu)))
    observed_batches=int((y>=2).sum())
    diagnostic={'observed_days_with_2plus':observed_batches,
                'poisson_expected_days_with_2plus':float(stats.poisson.sf(1,mu).sum()),
                'batch_tail_probability_exploratory':float(((sim>=2).sum(axis=1)>=observed_batches).mean()),
                'note':'Plug-in model check; catalog selection and provider batches violate ideal independent-arrival sampling.'}
    dump(OUT/'poisson_assumption_check.json',diagnostic)
    selected_mu=selected.intensity(dates)
    selected_sim=rng.poisson(selected_mu,size=(2000,len(selected_mu)))
    diagnostic.update({'selected_model':chosen,'selected_expected_days_with_2plus':float(stats.poisson.sf(1,selected_mu).sum()),
                       'selected_batch_tail_probability':float(((selected_sim>=2).sum(axis=1)>=observed_batches).mean())})
    dump(OUT/'poisson_assumption_check.json',diagnostic)
    return {'selected_rate_model':chosen,'forecast_start':str(future[0].date()),'forecast_end':str(future[-1].date()),
            'expected_count_plugin':float(selected.intensity(future).sum()),
            'simulated_count_mean':float(counts.mean()),'count_interval_90':np.quantile(counts,[.05,.95]).tolist(),
            'overdispersed_count_interval_90':np.quantile(nbcounts,[.05,.95]).tolist(),
            'current_frontier':float(catalog.eci.max()),'frontier_mean':float(frontiers.mean()),
            'frontier_interval_90':np.quantile(frontiers,[.05,.95]).tolist()}


def figures(catalog, cutoff, forecast, perf):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    d=catalog[catalog.date>=START]
    fig,axes=plt.subplots(2,1,figsize=(7.3,5.0),sharex=True)
    axes[0].scatter(d.date,d.eci,s=9,alpha=.6,color='#274b64')
    frontier=catalog.groupby('date').eci.max().cummax()
    initial=pd.Series([frontier.loc[frontier.index<START].iloc[-1]],index=[START])
    frontier=pd.concat([initial,frontier.loc[frontier.index>=START]]).groupby(level=0).max()
    axes[0].step(frontier.index,frontier,where='post',color='#9b503e')
    axes[0].set_ylabel('Release capability (ECI)')
    gp=d.dropna(subset=['gpqa']);axes[1].scatter(gp.date,gp.gpqa,s=9,alpha=.6,color='#274b64')
    axes[1].set_ylabel('GPQA Diamond (%)');axes[1].set_xlabel('Catalog release date')
    fig.autofmt_xdate();save(fig,'benchmark_marks')
    weekly=pd.read_csv(OUT/'weekly_arrivals.csv',index_col=0,parse_dates=True)
    weekly=weekly[weekly.index<=cutoff]  # Omit the incomplete current calendar week.
    future=pd.read_csv(OUT/'future_daily_intensity.csv',parse_dates=['date'])
    fig,ax=plt.subplots(figsize=(7.3,3.0))
    ax.bar(weekly.index,weekly.observed,width=5,color='.75',label='Observed catalog releases')
    ax.plot(weekly.index,weekly.expected,color='#274b64',label='Log-linear NHPP mean')
    for model,label,style in [('linear','Log-linear forecast','--'),(forecast['selected_rate_model'],'Selected rate forecast',':')]:
        frame=future[future.model==model].sort_values('date').reset_index(drop=True)
        blockends=[frame.date.iloc[i+6] for i in range(0,len(frame),7)]
        blockmeans=[frame.lambda_per_day.iloc[i:i+7].sum() for i in range(0,len(frame),7)]
        ax.plot(blockends,blockmeans,style,label=label,color='#9b503e')
    ax.axvline(cutoff,color='.4',lw=.8);ax.set_ylabel('Releases per week');ax.legend(fontsize=7,frameon=False,ncol=2)
    fig.autofmt_xdate();save(fig,'release_intensity')
    draws=np.load(OUT/'future_simulations.npz')
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8))
    axes[0].hist(draws['counts'],bins=np.arange(draws['counts'].max()+2)-.5,color='#274b64')
    axes[0].set_xlabel('Catalog releases in next 42 days');axes[0].set_ylabel('Simulation frequency')
    axes[1].hist(draws['frontiers'],bins=25,color='#274b64');axes[1].set_xlabel('Maximum ECI after 42 days')
    fig.tight_layout();save(fig,'release_forecast_distribution')
    labels={'control_ridge':'Ridge control','control_boosting':'Boosting control','counts_ridge':'Release counts','benchmark_ridge':'Observed benchmark marks',
            'nhpp_ridge':'Benchmarks + NHPP','nhpp_interactions':'NHPP + industry interactions','nhpp_boosting':'NHPP + boosting'}
    d=perf[perf.evaluation=='all'].set_index('model')
    fig,ax=plt.subplots(figsize=(7.3,3.1));ax.barh([labels[m] for m in ADOPTION_MODELS],d.loc[ADOPTION_MODELS,'mae'],color='#274b64')
    ax.axvline(3.039431,color='#9b503e',ls='--',label='Earlier industry-model MAE')
    ax.set_xlabel('Adoption forecast MAE (percentage points)');ax.invert_yaxis()
    ax.legend(fontsize=7,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01))
    save(fig,'adoption_release_comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of',default='2026-10-30')
    parser.add_argument('--simulations',type=int,default=2000)
    args=parser.parse_args();cutoff=pd.Timestamp(args.as_of)
    OUT.mkdir(parents=True,exist_ok=True)
    catalog=load_catalog(cutoff)
    catalog.to_csv(OUT/'catalog.csv',index=False)
    calendar=pd.read_csv(ROOT/'results/release_calendar.csv',parse_dates=['Publication Date'])
    with threadpool_limits(limits=1):
        features=make_issue_features(catalog,calendar,args.simulations)
        result,perf=adoption_evaluate(features,args.simulations)
        forecast=rate_analysis(catalog,cutoff,args.simulations)
        figures(catalog,cutoff,forecast,perf)
    summary={'as_of':args.as_of,'catalog_models_all_years':len(catalog),
             'catalog_models_since_2024':int((catalog.date>=START).sum()),
             'rate_start':str(START.date()),'simulations':args.simulations,**forecast,
             'adoption_forecasts':len(result),'adoption_origins':result.origin.nunique(),
             'scores_sha256':hashlib.sha256((ROOT/'data/eci_scores_2026-10-30.csv').read_bytes()).hexdigest(),
             'benchmarks_sha256':hashlib.sha256((ROOT/'data/eci_benchmarks_2026-10-30.csv').read_bytes()).hexdigest(),
             'adoption_pairs_sha256':hashlib.sha256((ROOT/'results/forecast_pairs.csv').read_bytes()).hexdigest(),
             'benchmark_vintage_limitation':'October 30 ECI/benchmark estimates are retrospectively calibrated; release-date filtering is not historical score-publication filtering.',
             'process_scope':'Arrivals of canonical models included in the scored Epoch catalog; not all LLM releases.',
             'forecast_scope':'Conditional model forecast; catalog coverage, batching, and misspecification are not resolved by simulation.'}
    dump(OUT/'summary.json',summary)
    print(perf.to_string(index=False));print(json.dumps(summary,indent=2,default=str))


if __name__=='__main__':main()
