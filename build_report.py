"""Build LaTeX tables and numerical macros from verified analysis outputs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import shutil

ROOT = Path(__file__).resolve().parent
R = ROOT/'results'
G = ROOT/'report/generated'
LABELS = {'persistence': 'Persistence', 'trailing_mean': 'Six-release mean',
          'baseline': 'Baseline OLS', 'subset_linear': 'Linear subset',
          'subset_nonlinear': 'Nonlinear subset', 'full_ols': 'Full OLS',
          'ridge': 'Ridge', 'elastic_net': 'Elastic net', 'boosting': 'Gradient boosting',
          'partial_pooling': 'Partial pooling', 'industry_models': 'Industry models'}


def escape(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('_', r'\_').replace('%', r'\%').replace('#', r'\#')


def table(path, caption, label, headers, rows, fmt=None):
    fmt = fmt or 'l'+'r'*(len(headers)-1)
    placement = '!htbp' if path.stem in ['performance', 'paired'] else 'H'
    text = '\\begin{table}['+placement+']\n\\centering\n\\small\n'
    text += f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text += f'\\begin{{tabular}}{{{fmt}}}\n\\toprule\n'
    text += ' & '.join(headers)+r' \\'+'\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row))+r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    path.write_text(text, encoding='utf-8')


def main():
    G.mkdir(exist_ok=True, parents=True)
    figures = ROOT/'report/figures'
    figures.mkdir(exist_ok=True)
    for source in (R/'figures').glob('*.pdf'):
        shutil.copy2(source, figures/source.name)
    s = json.loads((R/'summary.json').read_text())
    p = pd.read_csv(R/'performance.csv')
    allp = p[p.evaluation=='all'].set_index('model')
    best = allp.mae.idxmin()
    hist = p[p.evaluation=='historical'].set_index('model')
    ext = p[p.evaluation=='extension'].set_index('model')
    improvement = 100*(1-allp.loc[best,'mae']/allp.loc['trailing_mean','mae'])
    selection = pd.read_csv(R/'subset_selection.csv')
    nonlinear = selection[selection.model=='subset_nonlinear']
    macros = {'ForecastN':s['forecasts'], 'OriginN':s['origins'], 'CellN':s['potential_cells'],
              'ReleaseN':len(s['periods']), 'GroupN':s['groups'], 'IndustryN':s['industries'],
              'PairN':s['complete_pairs'], 'SubsetCandidates':int(nonlinear.candidates.iloc[0]),
              'AbstractResult':f"{LABELS[best]} achieve the lowest aggregate MAE, {allp.loc[best,'mae']:.2f} percentage points, compared with {allp.loc['trailing_mean','mae']:.2f} for the six-release mean and {allp.loc['baseline','mae']:.2f} for the baseline. Nonlinear terms are repeatedly selected, but their subset model does not improve aggregate accuracy over the baseline.",
              'ConclusionResult':f"Industry-specific regression reduces aggregate MAE by {improvement:.1f}\\% relative to a six-release mean in this recorded sample. Elastic net and gradient boosting also improve accuracy. Nonlinear subset selection yields unstable functional forms and no aggregate improvement over the baseline. Overparameterized minimum-norm fits interpolate the training data and produce excessive future errors; regularization stabilizes this experiment."}
    (G/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    rows = [[LABELS[m], f'{hist.loc[m,"mae"]:.2f}', f'{ext.loc[m,"mae"]:.2f}',
             f'{allp.loc[m,"mae"]:.2f}', f'{allp.loc[m,"rmse"]:.2f}',f'{allp.loc[m,"bias"]:+.2f}'] for m in LABELS]
    table(G/'performance.tex', 'Forecast errors in percentage points. Historical: 635 forecasts at nine origins; extension: 71 forecasts at one origin; all: 706 forecasts.',
          'tab:performance', ['Model','Historical MAE','Extension MAE','All MAE','All RMSE','All bias'], rows)
    c = pd.read_csv(R/'paired_comparisons.csv')
    rows = [[LABELS[r.model], f'{r.mean_origin_mae_difference:+.2f}',
             f'[{r.block_ci_low:+.2f}, {r.block_ci_high:+.2f}]', f'{r.wins}/{r.origins}'] for r in c.itertuples()]
    table(G/'paired.tex',r'Paired origin-level MAE differences relative to the six-release mean. Negative values favor the fitted model. Intervals are 95\% moving-block bootstrap intervals and are exploratory.',
          'tab:paired',['Model','Difference (pp)','Interval (pp)','Origins won'],rows)
    growth = pd.read_csv(R/'industry_growth.csv')
    rows = [[escape(r.industry),f'{r.growth_pp_per_4weeks:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]',
             f'{r.rank_low:.0f}--{r.rank_high:.0f}',f'{100*r.bootstrap_top3_share:.1f}\\%'] for r in growth.itertuples()]
    table(G/'growth.tex','Industry growth in percentage points per four weeks. Intervals and rank ranges summarize the conditional wild block bootstrap; top-three shares are resampling frequencies.',
          'tab:growth',['Industry','Growth',r'95\% interval','Rank range','Top three'],rows)
    c = pd.read_csv(R/'complexity.csv'); c['ratio']=c.p_over_n.round(2)
    rows = []
    for ratio, d in c.groupby('ratio'):
        mn = d[d.method=='minimum_norm']; ridge=d[d.method=='ridge']
        rows.append([f'{ratio:.2f}', f'{mn.train_rmse.median():.2g}',f'{mn.test_rmse.median():.2f}',
                     f'{ridge.train_rmse.median():.2f}',f'{ridge.test_rmse.median():.2f}',f'{ridge.effective_df.median():.1f}'])
    table(G/'complexity.tex','Median errors over three origins and three feature seeds. MN denotes minimum-norm least squares; degrees of freedom (df) are for ridge. Each origin has its own training-row count; medians are descriptive summaries.',
          'tab:complexity',['$p/n$','MN train RMSE','MN test RMSE','Ridge train','Ridge test','Ridge df'],rows)
    block_labels = {'expected':'Expected use', 'rate_sq':r'Current use squared ($y_t^2$)',
                    'time_sq':r'Time squared ($t^2$)', 'rate_time':r'Current use $\times$ time',
                    'expected_sq':r'Expected use squared ($e_t^2$)',
                    'rate_expected':r'Current $\times$ expected use',
                    'industry_trend':r'Industry $\times$ time', 'size_trend':r'Size $\times$ time',
                    'industry_rate':r'Industry $\times$ current use'}
    rows=[]
    for block in ['expected','rate_sq','time_sq','rate_time','expected_sq','rate_expected','industry_trend','size_trend','industry_rate']:
        count=nonlinear.blocks.fillna('').str.split('|').map(lambda x:block in x).sum()
        rows.append([block_labels[block],str(count),f'{count/len(nonlinear):.0%}'.replace('%',r'\%')])
    table(G/'selection.tex','Nonlinear-subset selection frequency over ten outer origins. Baseline main effects are mandatory.',
          'tab:selection',['Optional block','Selected origins','Frequency'],rows)
    ab=pd.read_csv(R/'ridge_ablations.csv')
    rows=[[escape(r.variant.replace('_',' ')),f'{r.mae:.3f}',f'{r.rmse:.3f}'] for r in ab.itertuples()]
    table(G/'ablations.tex','Ridge feature ablations, each tuned independently using the same inner temporal protocol.',
          'tab:ablations',['Dictionary','MAE (pp)','RMSE (pp)'],rows)
    ev=pd.read_csv(R/'release_event_windows.csv')
    rows=[[escape(r.model),str(r.date),str(r.pre_releases),str(r.post_releases),
           f'{r.mean_change_pp:+.2f}' if r.supported_window else 'Insufficient history'] for r in ev.itertuples()]
    table(G/'events.tex','Changes in the balanced-group mean from three pre-announcement reference periods to three wholly subsequent periods. These overlapping comparisons are descriptive.',
          'tab:events',['Announcement','Date','Pre','Post','Change (pp)'],rows)
    bysize=pd.read_csv(R/'mae_by_empsize.csv')
    size={'A':'1--4','B':'5--9','C':'10--19','D':'20--49','E':'50--99'}
    rows=[[size[r.Empsize]]+[f'{getattr(r,m):.2f}' for m in ['trailing_mean','baseline','industry_models','elastic_net','boosting']] for r in bysize.itertuples()]
    table(G/'size.tex','MAE by employment-size band over all ten evaluation origins.',
          'tab:size',['Employees','Six-release mean','Baseline','Industry','Elastic net','Boosting'],rows)
    rows=[[escape(r.industry),f'{r.growth_pp_per_4weeks:.2f}',f'{r.growth_sample_adjusted:.2f}',f'{r.growth_se_weighted:.2f}'] for r in growth.itertuples()]
    table(G/'growth_sensitivity.tex','Sensitivity of growth estimates (pp per four weeks) to a common sample-change indicator and inverse-variance weighting. Primary confidence intervals do not apply to these sensitivity fits.',
          'tab:sensitivity',['Industry','Primary','Sample adjusted','SE weighted'],rows)
    append = r'''\input{generated/growth.tex}
\input{generated/growth_sensitivity.tex}
\input{generated/selection.tex}
\input{generated/size.tex}
\input{generated/ablations.tex}
\input{generated/events.tex}
\input{generated/complexity.tex}
\par\medskip
\subsection{Computational record}
The repository includes the unchanged original report and code, an attributed source extract and hash, all predictions, selected blocks, tuning parameters, validation audits, industry bootstrap draws, event windows, complexity diagnostics, and the LaTeX source. Reproduction requires Python with the packages listed in \texttt{requirements.txt}. The commands are:
\begin{verbatim}
python -m unittest test_analysis test_release_process -v
python extended_analysis.py --workbook data/btos_ai_extract_2026-10-15.csv
python release_process.py --as-of 2026-10-30 --simulations 2000
python validate_release_results.py
python build_report.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
\end{verbatim}
Regression tests verify the original counts and rounded performance values, temporal training boundaries, hierarchy constraints, training-only preprocessing, rank handling, and recovery of a quadratic signal in synthetic data. Release-process checks verify zero-event days, date filtering, integrated intensities, simulation ranges, and matched adoption controls. Benchmark snapshots and provenance hashes are committed. The original baseline has 21 coefficients. Full-model column counts and ranks are recorded separately; redundant columns do not count as independent information.
'''
    (G/'appendix.tex').write_text(append,encoding='utf-8')
    findings = {'best_model':best,'best_mae':float(allp.loc[best,'mae']),
                'baseline_mae':float(allp.loc['baseline','mae']),
                'recent_mean_mae':float(allp.loc['trailing_mean','mae']),
                'relative_improvement_percent':float(improvement),
                'nonlinear_subset_mae':float(allp.loc['subset_nonlinear','mae'])}
    (R/'key_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    print(json.dumps(findings,indent=2))
    if (R/'releases/summary.json').exists():
        from build_release_report import main as build_release
        build_release()


if __name__=='__main__':
    main()
