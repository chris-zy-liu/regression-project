"""Independent consistency audit of release and benchmark outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import release_process as r


def main():
    out=r.OUT
    summary=json.loads((out/'summary.json').read_text())
    for name in ['eci_scores','eci_benchmarks']:
        p=r.ROOT/f'data/{name}_2026-10-30.csv'
        expected=json.loads(p.with_suffix('.provenance.json').read_text())['sha256']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    assert summary['scores_sha256']==hashlib.sha256((r.ROOT/'data/eci_scores_2026-10-30.csv').read_bytes()).hexdigest()
    catalog=r.load_catalog(pd.Timestamp(summary['as_of']))
    assert len(catalog)==summary['catalog_models_all_years']
    assert int((catalog.date>=r.START).sum())==summary['catalog_models_since_2024']
    features=pd.read_csv(out/'issue_features.csv',parse_dates=['issue','latest_catalog_date','target_reference_end'])
    assert (features.latest_catalog_date<=features.issue).all()
    assert ((features.target_reference_end-features.issue).dt.days==24).all()
    assert np.isfinite(features[r.KNOWN+r.FUTURE]).all().all()
    cv=pd.read_csv(out/'rate_cv_by_issue.csv',parse_dates=['cutoff','test_start','test_end','train_end'])
    assert (cv.train_end<cv.test_start).all() and (cv.test_end<=cv.cutoff).all()
    predictions=pd.read_csv(out/'adoption_predictions.csv',parse_dates=['origin','target_published'])
    assert len(predictions)==706 and predictions.origin.nunique()==10
    assert (predictions.target_published>predictions.origin).all()
    perf=pd.read_csv(out/'adoption_performance.csv')
    for split,d in [('all',predictions)]+list(predictions.groupby('evaluation')):
        for model in r.ADOPTION_MODELS:
            row=perf[(perf.evaluation==split)&(perf.model==model)].iloc[0]
            np.testing.assert_allclose(row.mae,np.abs(d[model]-d.target).mean(),atol=1e-11)
            np.testing.assert_allclose(row.rmse,np.sqrt(np.mean((d[model]-d.target)**2)),atol=1e-11)
    original=pd.read_csv(r.ROOT/'results/predictions.csv')
    np.testing.assert_allclose(predictions.control_ridge,original.ridge,atol=1e-9)
    intensity=pd.read_csv(out/'future_daily_intensity.csv')
    selected=intensity[intensity.model==summary['selected_rate_model']]
    assert len(selected)==42 and (selected.lambda_per_day>0).all()
    np.testing.assert_allclose(selected.lambda_per_day.sum(),summary['expected_count_plugin'])
    draws=np.load(out/'future_simulations.npz')
    assert len(draws['counts'])==summary['simulations']
    np.testing.assert_allclose(np.quantile(draws['counts'],[.05,.95]),summary['count_interval_90'])
    np.testing.assert_allclose(draws['frontiers'].mean(),summary['frontier_mean'])
    assert (draws['frontiers']>=summary['current_frontier']).all()
    for figure in ['benchmark_marks','release_intensity','release_forecast_distribution','adoption_release_comparison']:
        for suffix in ['pdf','png']:
            assert (out/'figures'/f'{figure}.{suffix}').stat().st_size>1000
    validation={'catalog_hashes_verified':True,'strict_release_date_filtering_verified':True,
                 'historical_score_availability_verified':False,'temporal_folds_verified':True,
                 'adoption_metrics_verified':True,'matched_ridge_verified':True,
                 'intensity_integration_verified':True,'simulation_quantiles_verified':True,
                 'models':len(r.ADOPTION_MODELS),'adoption_forecasts':len(predictions),'new_figures':4}
    r.dump(out/'validation.json',validation)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
