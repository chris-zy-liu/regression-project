"""Audit saved results independently of the fitting loops."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from extended_analysis import MODELS, moving_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results'))
    r = parser.parse_args().results
    summary = json.loads((r/'summary.json').read_text())
    source = Path(summary['workbook'].replace('\\', '/'))
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != summary['sha256'] and b'\r\n' not in source_bytes:
        source_hash = hashlib.sha256(source_bytes.replace(b'\n', b'\r\n')).hexdigest()
    assert source_hash == summary['sha256'], 'Source differs from the recorded file beyond line endings'
    p = pd.read_csv(r/'predictions.csv', parse_dates=['origin','target_published'])
    assert len(p) == summary['forecasts']
    assert p.origin.nunique() == summary['origins']
    assert not p.duplicated(['Sector','Empsize','step']).any()
    assert (p.target_published > p.origin).all()
    assert np.isfinite(p[MODELS+['target']].to_numpy()).all()
    performance = pd.read_csv(r/'performance.csv')
    for split, d in [('all', p)] + list(p.groupby('evaluation')):
        for m in MODELS:
            row = performance[(performance.evaluation==split) & (performance.model==m)].iloc[0]
            np.testing.assert_allclose(row.mae, np.abs(d[m]-d.target).mean(), atol=1e-12)
            np.testing.assert_allclose(row.rmse, np.sqrt(np.mean((d[m]-d.target)**2)), atol=1e-12)
    audit = pd.read_csv(r/'validation_audit.csv', parse_dates=['origin','latest_training_target'])
    assert (audit.latest_training_target <= audit.origin).all()
    assert (audit.full_rank <= audit.full_p).all()
    assert (audit.full_rank <= audit.train_n).all()
    selections = pd.read_csv(r/'subset_selection.csv').fillna({'blocks':''})
    for chosen in selections.blocks.str.split('|'):
        if 'expected_sq' in chosen or 'rate_expected' in chosen:
            assert 'expected' in chosen
    complexity = pd.read_csv(r/'complexity.csv')
    assert (complexity.p > complexity.n).any()
    assert (complexity['rank'] <= np.minimum(complexity.p, complexity.n)).all()
    assert (complexity.effective_df <= complexity['rank']+1e-6).all()
    interpolating = complexity[(complexity.method=='minimum_norm') & (complexity.p_over_n >= 1)]
    assert interpolating.train_rmse.max() < 1e-4
    growth = pd.read_csv(r/'industry_growth.csv')
    assert len(growth) == summary['industries']
    assert (growth.ci_low <= growth.ci_high).all()
    assert growth.bootstrap_top3_share.between(0,1).all()
    assert np.isclose(growth.bootstrap_top3_share.sum(), 3)
    for figure in ['errors_by_origin','subset_stability','conditional_curves','industry_growth',
                   'industry_ranks','release_timeline','complexity']:
        for suffix in ['pdf','png']:
            path=r/'figures'/f'{figure}.{suffix}'
            assert path.exists() and path.stat().st_size>1000
    output={'source_hash_verified':True,'forecast_count':len(p),'model_count':len(MODELS),
            'origin_count':p.origin.nunique(),'empirical_metrics_verified':True,
            'validation_boundaries_verified':True,'interpolation_verified':True,
            'figures_verified':7}
    (r/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
