"""Tests of release-rate estimation, marks, date filtering and saved forecasts."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import release_process as r


def catalog(dates, scores=None):
    dates=pd.DatetimeIndex(dates)
    return pd.DataFrame({'model':[f'M{i}' for i in range(len(dates))],
                         'provider':['Lab']*len(dates),'date':dates,
                         'eci':scores if scores is not None else np.linspace(110,150,len(dates)),
                         'gpqa':np.linspace(30,80,len(dates))})


class ReleaseTests(unittest.TestCase):
    def test_counts_keep_zero_days_and_batches(self):
        d=catalog(['2024-01-02','2024-01-02','2024-01-04'])
        dates,y=r.daily_counts(d,pd.Timestamp('2024-01-05'))
        np.testing.assert_array_equal(y,[0,2,0,1,0])
        _,batch=r.daily_counts(d,pd.Timestamp('2024-01-05'),True)
        self.assertEqual(batch.sum(),2)

    def test_nonstationary_rate_recovery(self):
        rng=np.random.default_rng(71)
        days=pd.date_range('2024-01-01','2026-01-01')
        x,_=r.design(days,'linear');beta=np.array([-.4,.55])
        counts=rng.poisson(np.exp(x@beta))
        d=catalog(np.repeat(days,counts))
        with threadpool_limits(limits=1):f=r.fit_rate(d,days[-1],'linear')
        self.assertAlmostEqual(f.coefficients[1],beta[1],delta=.2)
        self.assertTrue((f.intensity([days[0],days[-1]])>0).all())
        left=f.expected('2026-01-02','2026-01-11')
        right=f.expected('2026-01-12','2026-01-21')
        self.assertAlmostEqual(left+right,f.expected('2026-01-02','2026-01-21'))

    def test_rate_and_history_ignore_later_releases(self):
        dates=pd.date_range('2024-01-01',periods=120,freq='5D')
        d=catalog(dates);cut=pd.Timestamp('2025-05-01')
        extended=pd.concat([d,catalog(['2026-01-01'],[290])],ignore_index=True)
        with threadpool_limits(limits=1):
            a=r.fit_rate(d,cut);b=r.fit_rate(extended,cut)
        np.testing.assert_allclose(a.coefficients,b.coefficients)
        self.assertEqual(r.history_features(d,cut),r.history_features(extended,cut))

    def test_future_simulation_is_reproducible_and_frontier_monotone(self):
        d=catalog(pd.date_range('2024-01-01',periods=120,freq='5D'))
        cutoff=d.date.max()
        with threadpool_limits(limits=1):
            f=r.fit_rate(d,cutoff)
            a,fa=r.simulate_marks(f,d,cutoff,cutoff+pd.Timedelta(days=42),100)
            b,fb=r.simulate_marks(f,d,cutoff,cutoff+pd.Timedelta(days=42),100)
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(fa,fb)
        self.assertTrue((fa>=d.eci.max()).all())
        self.assertTrue((a>=0).all())

    def test_catalog_schema_and_gpqa_units(self):
        d=r.load_catalog(pd.Timestamp('2026-10-30'))
        self.assertFalse(d.model.duplicated().any())
        self.assertTrue(d.gpqa.dropna().between(0,100).all())
        self.assertTrue((d.date<=pd.Timestamp('2026-10-30')).all())

    def test_saved_predictor_and_target_alignment(self):
        path=r.OUT/'issue_features.csv'
        if not path.exists():self.skipTest('Analysis not yet run')
        d=pd.read_csv(path,parse_dates=['issue','target_reference_end','latest_catalog_date','rate_training_end'])
        self.assertTrue((d.latest_catalog_date<=d.issue).all())
        self.assertTrue((d.rate_training_end<=d.issue).all())
        self.assertTrue(((d.target_reference_end-d.issue).dt.days==24).all())
        cal=pd.read_csv('results/release_calendar.csv',parse_dates=['Publication Date','Reference End Date'])
        for i in range(len(cal)-3):
            issue=cal.iloc[i]['Publication Date']
            self.assertEqual(d[d.issue==issue].target_reference_end.iloc[0],cal.iloc[i+3]['Reference End Date'])
        self.assertTrue((d[r.FUTURE]>=0).all().all())

    def test_saved_adoption_metrics_and_control(self):
        path=r.OUT/'adoption_predictions.csv'
        if not path.exists():self.skipTest('Analysis not yet run')
        d=pd.read_csv(path)
        perf=pd.read_csv(r.OUT/'adoption_performance.csv')
        for model in r.ADOPTION_MODELS:
            self.assertTrue(np.isfinite(d[model]).all())
            saved=perf[(perf.evaluation=='all')&(perf.model==model)].mae.iloc[0]
            self.assertAlmostEqual(saved,np.abs(d[model]-d.target).mean(),places=10)
        previous=pd.read_csv('results/predictions.csv')
        np.testing.assert_allclose(d.control_ridge,previous.ridge,atol=1e-9)
        self.assertFalse(d.duplicated(['Sector','Empsize','step']).any())

    def test_saved_rate_cv_is_strictly_temporal(self):
        path=r.OUT/'rate_cv_by_issue.csv'
        if not path.exists():self.skipTest('Analysis not yet run')
        d=pd.read_csv(path,parse_dates=['cutoff','train_end','test_start','test_end'])
        self.assertTrue((d.train_end<d.test_start).all())
        self.assertTrue((d.test_end<=d.cutoff).all())

    def test_mark_load_uses_empirical_positive_part(self):
        path=r.OUT/'issue_features.csv'
        if not path.exists():self.skipTest('Analysis not yet run')
        features=pd.read_csv(path,parse_dates=['issue','target_reference_end'])
        cat=r.load_catalog(pd.Timestamp('2026-10-30'))
        for i in [0,10,19]:
            row=features.iloc[i];dates=pd.date_range(row.issue+pd.Timedelta(days=1),row.target_reference_end)
            beta,_,residuals,_=r.fit_marks(cat,row.issue)
            x,_=r.design(dates,'linear')
            f=r.fit_rate(cat,row.issue,row.rate_model)
            expected=np.sum(f.intensity(dates)*np.maximum(((x@beta)[:,None]+residuals[None,:]-100)/10,0).mean(axis=1))
            self.assertAlmostEqual(expected,row.expected_mark_load,places=10)


if __name__=='__main__':unittest.main()
