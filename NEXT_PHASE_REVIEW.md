# Milestone 2 follow-up — review draft

This work builds on the group's October 27 repository state. The four-page PDF is a team review draft; meeting details, contributions, destination, and instructor expectations remain to be confirmed. No submission is claimed.

## Official-data comparison

The October 20 official workbook is `data/census-sector-size-2026-10-20.xlsx`. Run `python audit_official_workbook.py data/census-sector-size-2026-10-20.xlsx` to compare it with the included extract. Across 2,300 common cells, 1,781 observed use rates, 1,825 expected-use rates and 1,781 standard errors match exactly; missingness matches. Publication date 202526 differs: official December 31, 2025; extract January 1, 2026. A full workbook run completed after fixing the reference-end column alias; its outputs are in `official_workbook_run/`. Baseline and industry-model MAEs match the extract run; richer-model results differ because the workbook supplies additional business-condition predictors. Compare the saved tables before selecting results for the team's report.

The CSV was normalized to LF by Git, while the source hash describes CRLF bytes. `validate_results.py` now accepts the recorded bytes reconstructed from LF only when the hash matches exactly; it still rejects other content changes.

## Release-rate feasibility experiment

The October 29 snapshot of the official model catalogue from https://epoch.ai/data/all_ai_models.csv is in `data/epoch-ai-models-2026-10-29.csv`. Run `python release_forecast.py data/epoch-ai-models-2026-10-29.csv`. Its SHA-256 and results are in `release_exploration/summary.json`. The experiment counts Epoch catalogue rows with Language in Domain by publication month, January 2024–September 2026. It compares expanding-window one-month forecasts from a constant rate and a log-linear time-varying Poisson rate; results are exploratory and depend on the chosen population and window. The catalogue is retrospective, publication date need not be product release, and changes in coverage can affect counts. The nominal Poisson interval omits parameter and catalogue uncertainty. Do not use the result as an established release forecast or causal input to adoption.

Epoch's benchmark archive is separately available at https://epoch.ai/data/benchmark_data.zip. The October 29 snapshot is `data/epoch-benchmark-data-2026-10-29.zip`. Run `python benchmark_exploration.py data/epoch-benchmark-data-2026-10-29.zip` for an exploratory two-month GPQA Diamond best-score comparison. In seven retrospectively selected comparisons, carry-forward MAE was 0.55 score points and a six-month linear trend had MAE 2.10 points. These are not an untouched evaluation: the present catalogue and analysis choices were selected after the fact. GPQA includes model release dates and run start times, but a run start is not proof of when a result became public. A leakage-safe link to business adoption remains unimplemented. External benchmark entries can have separate licenses; consult https://epoch.ai/benchmarks/use-this-data.

## Reproduce and verify

Use Python 3.13 and dependencies in `requirements.txt`. Run `python -m unittest test_analysis test_release_forecast test_benchmark_exploration -v`, then `python validate_results.py`, `python validate_results.py --results official_workbook_run`, and `python build_milestone2_draft.py`. The saved-results validator checks forecast metrics, timing boundaries, and figure presence; it is not a refit. The generated PDF contains unconfirmed process placeholders and requires human and team review.

Open work: obtain exact instructor feedback and current Canvas instructions; verify the team’s current source and individual work; evaluate benchmark models using a defensible availability date; test later Census releases prospectively; reassess the date discrepancy; and review the draft and artifacts with the team.
