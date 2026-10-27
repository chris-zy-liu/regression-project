# Forecasting Small-Business AI Use

ISYE 4031, Group 1 — Milestone 2.

This project evaluates whether pooled linear regression can forecast an industry-by-employment-size group's published AI-use rate three Census Business Trends and Outlook Survey (BTOS) releases (about six weeks) ahead. It covers employers with 1–99 employees.

## Files

- `Milestone2_Report_Group1.pdf`: supplied report, including methods, results, diagnostics, and next steps.
- `Milestone2_Code_Group1.zip`: original supplied code archive.
- `Milestone2_Code_Group1/analysis.py`: data preparation, historical expanding-window evaluation, diagnostics, tables, and figures.
- `Milestone2_Code_Group1/prospective.py`: evaluation on newer releases using the same model specifications and training rules.

## Reported findings

The historical evaluation uses 22 releases, 71 screened industry-size groups, and 635 forecasts across nine issue dates. Regression has mean absolute error of 3.33 percentage points, versus 3.40 for the latest-six-release average and 4.18 for carrying the latest rate forward. Adding expected AI use provides essentially no gain. Performance varies by size band, with larger firms producing noisier estimates. One subsequent release favors regression (3.09 versus 3.34 percentage points for the recent average), but more prospective evaluation is needed.

These results are transcribed from the report; they have not been independently reproduced in this checkout.

## Running the analysis

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The supplied ZIP contains only the two Python scripts. It does **not** contain the frozen Census workbook, the newer workbook, or generated outputs. To reproduce the historical results, obtain the original frozen workbook and place it at:

```text
Milestone2_Code_Group1/data/sector_size.xlsx
```

The report describes that snapshot as accessed October 3, 2026, with a historical analysis window ending at release 202619. A current download may contain revisions and is not a substitute for the original snapshot when verifying the reported results.

```powershell
python Milestone2_Code_Group1/analysis.py
python Milestone2_Code_Group1/prospective.py path/to/newer/sector_size.xlsx
```

Run the historical analysis first to create the output directory. Outputs are written under `Milestone2_Code_Group1/output/`. The prospective script also requires the frozen workbook and checks that historical AI-use rates and screened groups match.

## Next steps described in the report

Continue prospective evaluation; compare regularized regression with lagged predictors and gradient boosting under the same expanding-window protocol; evaluate gains by size band and relative to survey standard errors.
