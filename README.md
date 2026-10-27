# Industry Heterogeneity and Nonlinear Forecasts of Small-Business AI Use

ISYE 4031, Group 1. Census Business Trends and Outlook Survey (BTOS), employers with 1–99 employees, three-release forecast horizon (approximately six weeks).

## Report and principal findings

The revised paper is [`output/pdf/Group1_AI_Use_Report.pdf`](output/pdf/Group1_AI_Use_Report.pdf). Its self-contained LaTeX project is [`report/`](report/), with `main.tex` as the main document. The supplied Milestone 2 PDF and ZIP are preserved unchanged.

[`output/pdf/Group1_Changes_and_Questions.pdf`](output/pdf/Group1_Changes_and_Questions.pdf) summarizes the changes from Milestone 2 and remaining research questions. Its LaTeX source is [`revision_summary/main.tex`](revision_summary/main.tex).

The extended analysis reproduces the original historical baseline and evaluates **706 forecasts across 10 issue dates**, covering **71 screened groups in 15 industries**:

| Model | MAE (percentage points) |
|---|---:|
| Separate industry regressions | 3.04 |
| Elastic net | 3.07 |
| Gradient boosting | 3.12 |
| Ridge | 3.13 |
| Original pooled baseline | 3.30 |
| Nonlinear subset selection | 3.32 |
| Six-release average | 3.39 |
| Full unregularized OLS dictionary | 5.68 |

- Separate industry models reduce MAE by 10.3% relative to a recent average and 8.0% relative to baseline OLS. Gains vary across industries and sizes.
- The nonlinear subset selects expectation curvature and interactions, but does not improve aggregate accuracy over the baseline. Current use squared and time squared are never selected.
- Finance, Information, and professional services form the leading estimated growth group. Finance and Information cannot be confidently ordered.
- The random-feature experiment reaches interpolation and an overparameterized regime. Unregularized minimum-norm predictions become unstable; ridge controls their errors.
- Selected AI announcements coincide with some adoption increases, but overlapping events and the underlying trend prevent causal attribution. The short series cannot establish annual seasonality.

## Data and interpretation

The original frozen workbook was not supplied. Census's direct download returned server timeouts during acquisition. The committed input is an unchanged, attributed CSV redistribution of Census AI estimates by The Uptime Report, with a publisher extraction date of October 15, 2026. Its counts and original forecasting results reproduce the supplied report. It ends at release **202620**, published October 8; it is not the latest Census vintage as of October 26.

See [`data/README.md`](data/README.md) and the provenance JSON for attribution, license, source URL, hash, omitted suppression flags, and collection-date reconstruction. The extract lacks non-AI business-condition questions. The workbook parser can include those fields when an official workbook becomes available, but they are not part of the reported run.

The additional release is a **retrospective extension test** for the newly developed methods. The models were specified after that outcome was public. Results are associations and forecasts of published group estimates, with conditional uncertainty; they do not identify effects of AI releases or outreach.

## Reproduce

Python 3.11 or newer and a LaTeX installation with `pdflatex` are recommended. The recorded run used Python 3.13 and the package versions in `environment_versions.json`.

```powershell
python -m pip install -r requirements.txt
python -m unittest test_analysis -v
python extended_analysis.py --workbook data/btos_ai_extract_2026-10-15.csv
python build_report.py
Set-Location report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Copy the resulting `report/main.pdf` to the final deliverable path if rebuilding for distribution. The Codex LaTeX editor can also compile `report/` directly.

The full analysis performs exhaustive selection over 320 nonlinear block models at every forecast origin and 126 random-feature fits, and can take several minutes. `--resume` reuses saved forecasts and complexity results only when the input hash matches, then rebuilds diagnostics, ridge ablations, and figures. It is intended for manuscript regeneration following a completed run; use a full run after changing forecast specifications. Bootstrap draws and random-feature seed counts are configurable, with defaults of 2,000 and three.

To obtain the official workbook when the Census host is available:

```powershell
python download_data.py
python extended_analysis.py --workbook data/sector_size.xlsx
```

The downloadable workbook may be a different vintage. Do not substitute it silently when verifying the committed results.

## Analysis outputs

`results/` contains the reconstructed panel and forecast pairs; every prediction; selected subsets and tuning parameters; temporal validation audits; forecast errors by date, industry, and size; paired block intervals; growth estimates and bootstrap draws; ranking and sample-change sensitivities; ridge feature ablations; release-event windows; matrix ranks, effective degrees of freedom, and interpolation diagnostics; figures in PDF and PNG; and the input hash and run summary.

`test_analysis.py` checks original-report reproduction, temporal boundaries, hierarchical subset constraints, training-only imputation, rank handling, and recovery of a synthetic quadratic signal. `validate_results.py` checks the saved empirical outputs and manuscript figures.

The original `Milestone2_Code_Group1/analysis.py` and `prospective.py` retain their original behavior and workbook requirements. The new implementation is in `extended_analysis.py`.
