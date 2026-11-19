# ISyE 4031 Milestone 3: report sections for the professor's feedback

Source: `two_stage_diagnostics.py` (added to the group repo). Same 71 groups, 10 issue dates (Apr 23 to Aug 27, 2026), 706 forecasts, 3-release horizon, expanding window, same ridge tuning as `release_process.py`. The direct model reproduces the repo's `control_ridge` (MAE 3.135).

---

## 1. Direct vs. two-stage forecasting

We compare three versions of the same ridge model. They differ only in one input.

- **Direct:** current rate, lags, industry, size and trend (the control model).
- **Two-stage:** direct + the Poisson forecast of AI model releases in the target window (stage 1).
- **Oracle:** direct + the *actual* number of releases in that window. This is not a usable forecast because it uses future data. It shows the best case if stage 1 were perfect.

**Table A. Adoption forecast error (percentage points), 706 forecasts**

| Forecast | MAE | RMSE | Bias | MAE, 1-4 emp. | MAE, 50-99 emp. |
|---|---|---|---|---|---|
| Carry latest rate forward | 4.15 | 5.75 | -1.05 | 2.29 | 6.66 |
| Mean of latest six releases | 3.39 | 4.64 | -2.07 | 2.53 | 4.65 |
| **Direct** | **3.14** | **4.32** | 0.47 | **2.03** | 4.99 |
| Two-stage (Poisson forecast) | 3.24 | 4.41 | 0.65 | 2.14 | 4.94 |
| Oracle (actual releases) | 3.17 | 4.37 | 0.54 | 2.08 | 4.98 |

**Table B. Paired comparison across the 10 issue dates**

| Comparison | Mean MAE change | Dates won | Paired t p-value | Sign test p-value |
|---|---|---|---|---|
| Two-stage vs. direct | +0.10 | 2 of 10 | 0.25 | 0.11 |
| Oracle vs. direct | +0.04 | 4 of 10 | 0.33 | 0.75 |
| Two-stage vs. oracle | +0.06 | 1 of 10 | 0.20 | 0.02 |
| Direct vs. six-release mean | -0.26 | 10 of 10 | 0.002 | 0.002 |

**Answer.** The two-stage approach does not improve prediction. It is 0.10 pp worse than direct forecasting and wins at only 2 of 10 dates. The difference is not significant, so the fair reading is "no gain for extra complexity." The direct model, by contrast, beats the six-release mean at all 10 dates.

---

## 2. Why the models perform differently

The oracle arm splits the problem into two questions.

**(a) Is release information useful at all? No.** Even with perfect knowledge of how many models came out, the oracle is 0.04 pp *worse* than direct and wins at only 4 of 10 dates. If knowing the true count does not help, no forecast of it can help. Three reasons:

1. **One number for all 71 groups.** The release count is the same for every industry and size on a given date. It can only explain movement that all groups share, and the trend term already captures that.
2. **Stage 1 is almost a time trend.** The Poisson forecast correlates 0.95 with time (VIF about 9.4 against the trend). It adds a second, noisier copy of information the model already has.
3. **Releases do not move adoption within six weeks.** Actual release counts correlate only 0.35 with time and bounce from 0 to 12 per window. Businesses do not adopt AI in response to a given month's releases, so that bounce is noise for our target.

**(b) Do stage-1 errors make it worse? Yes, a little.** Two-stage is worse than the oracle at 9 of 10 dates (sign test p = 0.02). The Poisson forecast and the actual count correlate only 0.12, so stage 1 hands stage 2 a noisy input. This is error propagation.

**(c) Overfitting.** Two-stage has the lowest training error (3.21) but the highest test error (3.24). Direct is the reverse (3.30 train, 3.14 test). Most of the loss comes from the first issue date (Apr 23). There, tuning chose weaker regularization for the two-stage model (alpha 100 vs. 1000). It fit training data better (2.88 vs. 3.44) but forecast worse (3.69 vs. 2.86). The release input takes only one value per issue date, so it acts like a few extra date dummies the model can bend to fit.

**(d) Survey noise sets a floor.** 86% of direct errors are already within two survey standard errors of the target. There is little room left for any added input to help.

**Takeaway:** a more complex model is not better here. The extra stage adds noise and a chance to overfit, and the information it carries is already in the trend.

---

## 3. Evaluating the Poisson model

**Why Poisson fits.** Model releases are counts (0, 1, 2, ...) of separate events over time. A Poisson process models exactly that: a non-negative integer count whose rate can change. We use a log-linear rate, λ(t) = exp(b0 + b1·years + b2·weekday). This keeps the rate positive and lets it grow by a fixed percent per year. Linear regression could predict negative or fractional counts and assumes constant variance, which counts break (for Poisson, variance equals the mean).

**Is the assumption met? Mostly, with overdispersion.** Pearson dispersion is 1.37 for the selected model (1.0 is ideal). Providers batch releases: 47 days had 2+ releases vs. about 26 expected. So intervals are too narrow. The nominal 90% interval covered 78% of actual counts. We report the overdispersion-adjusted interval (6 to 21 releases for Oct 31 to Dec 11) next to the plain one (6 to 20).

**Is the error good or bad? Good, given the noise.** Forecasting releases in the 24-day target window, over 23 issue dates:

| Stage-1 forecast | MAE (releases) |
|---|---|
| Carry last 24 days forward | 3.61 |
| Historical average rate | 2.83 |
| **Poisson, log-linear rate** | **2.58** |
| Noise floor (true rate known) | 1.82 |

The average window has 6.2 releases, so a 2.58 error is large in relative terms (about 40%). But it beats both naive forecasts (29% skill vs. carry-forward, 9% vs. the historical average). The noise floor matters most. Even if we knew the true rate exactly, Poisson randomness alone would give an MAE of about 1.82. Our model closes most of the gap between naive forecasting and that floor. In the 42-day cross-validation, the log-linear model (MAE 3.39) also beats a constant rate (4.75), while a quadratic rate (3.69) does worse. That is another case where more complexity did not help.

**The catch:** the forecast is mostly bias-free trend (bias -0.97, slightly under-predicting), and it correlates only 0.12 with actual counts window to window. It gets the level right but not the ups and downs. That is why it helps little as an input to stage 2.

---

## 4. Additional variables (starting point for whoever owns #4)

The repo already tested added release variables under the same protocol (`results/releases/adoption_comparisons.csv`, ridge, 10 issue dates). None helped:

| Added to direct model | MAE change (pp) | Dates won | 95% block CI |
|---|---|---|---|
| Past release count (42 days) | +0.07 | 5 of 10 | -0.06 to +0.22 |
| Benchmark scores (ECI, GPQA) | +0.05 | 5 of 10 | -0.05 to +0.18 |
| Benchmarks + Poisson forecast | +0.13 | 5 of 10 | -0.02 to +0.36 |
| Same + industry interactions | +0.11 | 6 of 10 | -0.07 to +0.36 |
| Boosting with all of the above | -0.01 | 4 of 10 | -0.15 to +0.11 |

Expected use (Q24) was already dropped in Milestone 2 for no gain. Rule to state up front: keep a variable only if it cuts MAE by more than 0.1 pp and wins at most dates. Nothing passes, so the direct model stays.

---

## 5. Conclusion (draft)

**What we observed.** Direct regression beat the six-release mean at all 10 issue dates (3.14 vs. 3.39 pp). Adding a stage-1 Poisson forecast of AI releases made adoption forecasts slightly worse (3.24 pp). Even the true release counts did not help (3.17 pp).

**Why.** Release counts are one number shared by every group, so they only explain common movement, which the trend already captures. The Poisson forecast is itself about 95% correlated with time. Stage-1 errors then add noise, and the extra input let the model overfit at one issue date. Survey noise caps how much any model can improve.

**Which approach is more reasonable.** The direct model. It is simpler, more accurate, easier to explain, and has nothing upstream that can fail. The Poisson model is still a sound way to forecast releases themselves (it beats both naive forecasts), but its output should not feed adoption forecasts at a six-week horizon.

**Limitations and lessons.**
- Direct: still above the noise floor for 50-99 employee firms (MAE 4.99), and 22 releases cannot show seasonality.
- Two-stage: depends on a retrospective Epoch catalog, release batching breaks independence, and only 10 test dates limit power.
- Both: forecasts of published group estimates, not causal effects of releases on adoption.
- Lesson: test whether a new input helps *when known perfectly* before building a model to forecast it. The oracle check would have told us early that stage 1 could not pay off.
