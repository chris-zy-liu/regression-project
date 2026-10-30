"""Generate release-process tables, macros and figures for the main paper."""
from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
from build_report import table, escape

ROOT=Path(__file__).resolve().parent
R=ROOT/'results/releases'
G=ROOT/'report/generated'
LABELS={'control_ridge':'Ridge control','control_boosting':'Boosting control',
        'counts_ridge':'Counts + ridge','benchmark_ridge':'Benchmarks + ridge',
        'nhpp_ridge':'Benchmarks + NHPP + ridge','nhpp_interactions':'NHPP + industry interactions',
        'nhpp_boosting':'Benchmarks + NHPP + boosting'}
RATE={'constant':'Constant','linear':'Linear time','quadratic':'Quadratic time',
      'linear_weekday':'Time + weekday','linear_annual':'Time + annual cycle'}


def main():
    s=json.loads((R/'summary.json').read_text())
    mark=json.loads((R/'mark_regression.json').read_text())
    diag=pd.read_csv(R/'rate_diagnostics.csv').set_index('model')
    cv=pd.read_csv(R/'rate_model_comparison.csv').set_index('model')
    co=pd.read_csv(R/'rate_coefficients.csv')
    coef=co[co.model==s['selected_rate_model']].set_index('term')
    macros={'CatalogAll':s['catalog_models_all_years'],'CatalogEvents':s['catalog_models_since_2024'],
            'ReleaseExpected':f"{s['expected_count_plugin']:.2f}",
            'ReleaseLow':f"{s['count_interval_90'][0]:.0f}",'ReleaseHigh':f"{s['count_interval_90'][1]:.0f}",
            'ReleaseNBLow':f"{s['overdispersed_count_interval_90'][0]:.0f}",
            'ReleaseNBHigh':f"{s['overdispersed_count_interval_90'][1]:.0f}",
            'FutureFrontier':f"{s['frontier_mean']:.2f}",
            'FutureFrontierLow':f"{s['frontier_interval_90'][0]:.2f}",
            'FutureFrontierHigh':f"{s['frontier_interval_90'][1]:.2f}",
            'ReleaseBetaZero':f"{coef.loc['intercept','coefficient']:.3f}",
            'ReleaseBetaTime':f"{coef.loc['time_years','coefficient']:.3f}",
            'ReleaseBetaWeekday':f"{coef.loc['weekday','coefficient']:.3f}",
            'ReleaseGrowthFactor':f"{np.exp(coef.loc['time_years','coefficient']):.2f}",
            'ReleaseDispersion':f"{diag.loc[s['selected_rate_model'],'pearson_dispersion']:.2f}",
            'MarkSlope':f"{mark['eci_slope_per_year']:.2f}",
            'MarkSlopeSE':f"{mark['slope_se_provider_week_clustered']:.2f}"}
    (G/'release_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
    rows=[]
    for model in RATE:
        d=diag.loc[model];c=cv.loc[model]
        rows.append([RATE[model],f'{c.mean_daily_nll:.3f}',f'{c.mean_42day_abs_error:.2f}',
                     f'{d.expected_next42:.2f}',f'{d.pearson_dispersion:.2f}'])
    table(G/'release_rate_comparison.tex',
          'Release-rate model selection using six earlier, disjoint 42-day validation windows. NLL is mean daily negative log likelihood; dispersion uses the final fit. Forecast counts cover October 31--December 11, 2026.',
          'tab:release-rate',['Rate regression','Validation NLL','Count MAE','Expected count','Dispersion'],rows)
    p=pd.read_csv(R/'adoption_performance.csv')
    allp=p[p.evaluation=='all'].set_index('model');h=p[p.evaluation=='historical'].set_index('model');e=p[p.evaluation=='extension'].set_index('model')
    rows=[]
    for m in LABELS:
        rows.append([LABELS[m],f'{allp.loc[m,"mae"]:.3f}',f'{allp.loc[m,"rmse"]:.3f}',
                     f'{h.loc[m,"mae"]:.3f}',f'{e.loc[m,"mae"]:.3f}'])
    table(G/'release_adoption.tex',
          'Adoption forecast errors in percentage points on the same 706 forecasts. Controls use the same feature backbone, tuning grids, and earlier validation dates as their augmented counterparts.',
          'tab:release-adoption',['Specification','All MAE','All RMSE','Historical MAE','Extension MAE'],rows)
    comparisons=pd.read_csv(R/'adoption_comparisons.csv')
    rows=[]
    for m,ref in [('benchmark_ridge','control_ridge'),('nhpp_ridge','control_ridge'),
                   ('nhpp_ridge','benchmark_ridge'),('nhpp_interactions','nhpp_ridge'),
                   ('nhpp_boosting','control_boosting')]:
        d=comparisons[(comparisons.model==m)&(comparisons.reference==ref)].iloc[0]
        short={'control_ridge':'Ridge control','benchmark_ridge':'Benchmark ridge',
               'nhpp_ridge':'NHPP ridge','nhpp_interactions':'Interactions','nhpp_boosting':'NHPP boosting','control_boosting':'Boosting control'}
        rows.append([short[m]+' vs. '+short[ref],f'{d.difference:+.3f}',f'[{d.ci_low:+.3f}, {d.ci_high:+.3f}]',f'{d.wins:.0f}/10'])
    table(G/'release_adoption_pairs.tex',
          r'Paired origin-level MAE differences. Negative values favor the first specification. Intervals are exploratory 95\% moving-block bootstrap intervals over ten dates.',
          'tab:release-pairs',['Comparison','Difference (pp)','Interval (pp)','Origins won'],rows)
    cat=pd.read_csv(R/'catalog.csv',parse_dates=['date']).set_index('model')
    names=['Gemini 3 Pro','Claude Opus 4.5','GPT-5.2','Claude Opus 4.6','Gemini 3.1 Pro','GPT-5.4']
    rows=[]
    for name in names:
        d=cat.loc[name]
        gp=f'{d.gpqa:.1f}' if pd.notna(d.gpqa) else '--'
        rows.append([escape(name),str(d.date.date()),f'{d.eci:.2f}',f'[{d.eci_ci_low:.2f}, {d.eci_ci_high:.2f}]',gp])
    table(G/'release_benchmarks.tex',
          r'Benchmark marks for the six announcements in the earlier chronology. ECI and its 90\% intervals use the October 30 Epoch vintage. GPQA Diamond is a percentage, summarized by the within-model median across listed evaluation variants; missing scores are not inferred.',
          'tab:release-marks',['Canonical model','Release date','ECI','ECI interval',r'GPQA (\%)'],rows)
    for f in (R/'figures').glob('*.pdf'):
        shutil.copy2(f,ROOT/'report/figures'/f.name)
    print('Generated benchmark and release-process report tables.')


if __name__=='__main__':main()
