Group 1 Poisson release investigation

Use Python 3.13. Run from the extracted ZIP root.

python -m pip install -r requirements.txt
python -m unittest test_poisson_investigation -v
python poisson_investigation.py
python build_poisson_section.py

The analysis compares 12 specifications on six outer 42-day windows.
Nested selection minimizes count log loss on four earlier windows.
Exact probabilities use compound-Poisson recursion. No random seed is needed.
Every candidate and unsuccessful variation is recorded under
results/poisson_investigation/, along with all validation scores.

Compound variants vary time and weekday effects, history length, recent batch
sizes, and provider-specific batch-size distributions. Shrinkage constants are
fixed for this run and documented in the report and code.
The cutoff remains October 30, 2026. The outer windows were examined previously.
This is retrospective analysis. forecast.json extrapolates from October 30 and
must not be described as a November 17 forecast. Predictive intervals condition
on fitted parameters and batch probabilities, excluding estimation uncertainty.

The analysis generates comparison.pdf/png. The build script generates busy_days
figures and a table. Enter poisson_report/paper and run pdflatex main.tex twice
to compile the section. The PDF is also delivered separately.

Epoch AI snapshots and provenance are included. See data/README.md for sources.
Shared release_process.py imports extended_analysis.py and the original
Milestone2_Code_Group1/analysis.py. All dependencies are included.
Four tests verify analytical probabilities and filtering of future events.
AI assistance supported implementation, checking, and drafting.
