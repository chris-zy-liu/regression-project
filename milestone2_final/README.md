# Group 1 Milestone 2: code and data

Forecasts each industry-by-size group's Census BTOS AI-use rate three releases (about six weeks) ahead, and compares regression with simple historical forecasts. The report is `milestone2_final/report.pdf`.

## Data

| File | Source | Used for |
|---|---|---|
| `data/census-sector-size-2026-10-20.xlsx` | U.S. Census Bureau, BTOS "Sector by Employment Size Class" workbook, downloaded Oct 20, 2026 | All forecasts, tables, figures |
| `data/btos_ai_extract_2026-10-15.csv` | The Uptime Report's redistribution of the same Census estimates (CC BY 4.0); matches the workbook on every AI-use rate, SE, and missing cell | Adoption pairs for the release analysis |
| `data/eci_scores_2026-10-30.csv`, `data/eci_benchmarks_2026-10-30.csv` | Epoch AI, Epoch Capabilities Index, https://epoch.ai/data/eci_scores.csv and https://epoch.ai/data/eci_benchmarks.csv, Oct 30, 2026 (CC BY 4.0) | Release process and release features in adoption forecasts (report Section 5) |
| `data/epoch-ai-models-2026-10-29.csv`, `data/epoch-benchmark-data-2026-10-29.zip` | Epoch AI, Oct 29, 2026 | Earlier exploratory checks; not cited in the report |

All files are included. `python download_data.py` downloads the current Census workbook, which will be a newer vintage and give slightly different results. Use the included file to reproduce the report. Epoch benchmark entries may carry separate licenses: https://epoch.ai/benchmarks/use-this-data.

## Setup

Python 3.13, `pip install -r requirements.txt` (includes statsmodels for the release model), and `pdflatex` (TeX Live or TinyTeX) for the PDF. Results were produced with numpy 2.3.1, pandas 2.3.1, scipy 1.16.2, scikit-learn 1.8.0, matplotlib 3.10.0, and reproduced exactly with numpy 2.5.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.7.2, matplotlib 3.10.6.

## Run, from the repository root, in order

```bash
python -m unittest test_analysis test_release_process test_release_forecast test_benchmark_exploration
python extended_analysis.py --workbook data/census-sector-size-2026-10-20.xlsx --output official_workbook_run
python validate_results.py --results official_workbook_run
mkdir -p Milestone2_Code_Group1/data && cp data/census-sector-size-2026-10-20.xlsx Milestone2_Code_Group1/data/sector_size.xlsx
python Milestone2_Code_Group1/analysis.py
python Milestone2_Code_Group1/prospective.py Milestone2_Code_Group1/data/sector_size.xlsx
python extended_analysis.py --workbook data/btos_ai_extract_2026-10-15.csv
python release_process.py --as-of 2026-10-30 --simulations 2000
python validate_release_results.py
python milestone2_final/make_figures.py
cd milestone2_final && pdflatex report.tex && pdflatex report.tex
```

| Step | Produces | Report |
|---|---|---|
| `extended_analysis.py` (several minutes) | `official_workbook_run/`: panel, 706 forecasts, errors, industry growth | Data table, Table 1, Figure 1 |
| `Milestone2_Code_Group1/analysis.py`, `prospective.py` | Original 9-date baseline: diagnostics, coefficient, Cook's distances; release 202620 check | Section 4 diagnostics, forward check |
| `extended_analysis.py` on the extract, then `release_process.py` | `results/` adoption pairs (same targets), then `results/releases/`: release-rate model, simulations, adoption forecasts with release features | Section 5 |
| `make_figures.py` | `milestone2_final/figures/`, `numbers.json` | Figures 1–2, Table 1 values |

**Seeds.** `extended_analysis.py` uses seed 4031 for the moving-block bootstrap (2,000 draws), gradient boosting, and random-feature experiments. `release_process.py` uses seed 403130 for its simulations and folds. The other scripts are deterministic.

**Time ordering.** Each forecast is issued on a release's publication date. Models train only on pairs whose target was already published; the code asserts this at every date, and `validate_results.py` rechecks it.
