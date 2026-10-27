"""ISYE 4031 Milestone 2 (Group 1): BTOS AI-use pipeline, data quality, baseline.

Run: python3 analysis.py  ->  writes output/ (tables, figures, results.json).
Target: an industry-by-size group's published AI-use rate three releases ahead.
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.dates
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
WORKBOOK = DATA / "sector_size.xlsx"
FIRST, LAST = 202524, 202619       # comparable window: revised AI question (Nov 17, 2025) to Sept 24, 2026
H = 3                              # forecast horizon in releases (about six weeks)
SCREEN, MIN_PUBLISHED = 8, 7       # screen groups on the first 8 releases only
FIRST_ORIGIN = 10
SIZES = {"A": "1-4", "B": "5-9", "C": "10-19", "D": "20-49", "E": "50-99"}
MODELS = {"persistence": "Carry latest rate forward",
          "trailing_mean": "Mean of latest six releases",
          "regression": "Regression (baseline)",
          "regression_expected": "Regression + expected use"}


def to_long(sheet, question, value, last=LAST):
    """One row per sector x size x release for answer 1 of a question; 'S' (suppressed) -> NaN."""
    rows = sheet[(sheet["Question ID"] == question) & (sheet["Answer ID"] == 1) & sheet["Empsize"].isin(list(SIZES))]
    periods = [c for c in rows.columns if re.fullmatch(r"20\d{4}", str(c)) and FIRST <= int(c) <= last]
    long = rows.melt(id_vars=["Sector", "Empsize"], value_vars=periods, var_name="period", value_name="raw")
    long["period"], long["Sector"] = long["period"].astype(int), long["Sector"].astype(str)
    long[value] = pd.to_numeric(long["raw"].astype(str).str.rstrip("%"), errors="coerce")
    return long.drop(columns="raw")


def load(workbook=WORKBOOK, last=LAST):
    estimates = pd.read_excel(workbook, "Response Estimates")
    errors = pd.read_excel(workbook, "Response Standard Errors")
    keys = ["Sector", "Empsize", "period"]
    panel = (to_long(estimates, 7, "rate", last)
             .merge(to_long(estimates, 24, "expected", last), on=keys, validate="one_to_one")
             .merge(to_long(errors, 7, "se", last), on=keys, validate="one_to_one"))
    dates = pd.read_excel(workbook, "Collection and Reference Dates").dropna(subset=["Smpdt"])
    dates = dates.assign(period=dates["Smpdt"].astype(int)).set_index("period")
    periods = sorted(panel["period"].unique())
    panel["step"] = panel["period"].map({p: i for i, p in enumerate(periods)})
    panel["published"] = panel["period"].map(dates["Publication Date"])
    return panel, dates, periods


def quality(panel):
    """Cell counts, suppression and survey noise by size band (all 20 sectors)."""
    assert not panel.duplicated(["Sector", "Empsize", "period"]).any()
    assert panel["rate"].dropna().between(0, 100).all()
    table = panel.groupby("Empsize").agg(cells=("rate", "size"), suppressed=("rate", lambda s: s.isna().mean() * 100),
                                         median_rate=("rate", "median"), median_se=("se", "median"))
    table.loc["All"] = [len(panel), panel["rate"].isna().mean() * 100, panel["rate"].median(), panel["se"].median()]
    return table.rename(index=SIZES).round(1)


def build_pairs(panel):
    """Pair each issue release t with the same group's rate at t+H, after screening groups on early releases."""
    early = panel[(panel["step"] < SCREEN) & (panel["Sector"] != "XX")]
    counts = early.assign(ok=early[["rate", "expected"]].notna().all(axis=1)).groupby(["Sector", "Empsize"])["ok"].sum()
    groups = set(counts[counts >= MIN_PUBLISHED].index)
    target = panel[["Sector", "Empsize", "step", "period", "rate", "se", "published"]].rename(
        columns={"period": "target_period", "rate": "target", "se": "target_se", "published": "target_published"})
    target["step"] -= H
    pairs = panel.drop(columns="se").merge(target, on=["Sector", "Empsize", "step"], validate="one_to_one")
    pairs = pairs[[k in groups for k in zip(pairs["Sector"], pairs["Empsize"])]]
    return pairs, groups


def design(rows, sectors, sizes, expected=False):
    cols = [np.ones(len(rows)), rows["rate"], rows["step"]]
    cols += [rows["Sector"] == s for s in sectors[1:]] + [rows["Empsize"] == s for s in sizes[1:]]
    if expected:
        cols.append(rows["expected"])
    return np.column_stack(cols).astype(float)


def evaluate(panel, pairs, groups, dates, periods, origins=None):
    """Expanding-window evaluation: train only on pairs whose target was published by the issue date."""
    sectors, sizes = sorted({g[0] for g in groups}), sorted({g[1] for g in groups})
    complete = pairs.dropna(subset=["rate", "expected", "target"])
    results, last_fit = [], None
    for step in origins or range(FIRST_ORIGIN, len(periods) - H):
        issue = dates.loc[periods[step], "Publication Date"]
        test = complete[complete["step"] == step].copy()
        train = complete[complete["target_published"] <= issue]
        assert (test["target_published"] > issue).all() and train["step"].max() <= step - H
        recent = panel[(panel["step"] <= step) & (panel["step"] > step - 6)].groupby(["Sector", "Empsize"])["rate"].mean()
        test["persistence"] = test["rate"]
        test["trailing_mean"] = [recent[k] for k in zip(test["Sector"], test["Empsize"])]
        for name, expected in [("regression", False), ("regression_expected", True)]:
            x = design(train, sectors, sizes, expected)
            coef, _, rank, _ = np.linalg.lstsq(x, train["target"], rcond=None)
            assert rank == x.shape[1]
            test[name] = design(test, sectors, sizes, expected) @ coef
            if not expected:
                last_fit = (train, x, coef)
        test["origin"] = issue
        results.append(test)
    return pd.concat(results, ignore_index=True), last_fit


def scores(results):
    """Overall MAE/RMSE plus MAE by size band; MAE by forecast origin for Figure 2."""
    err = {n: results[n] - results["target"] for n in MODELS}
    table = pd.DataFrame({"MAE": {n: e.abs().mean() for n, e in err.items()},
                          "RMSE": {n: np.sqrt((e ** 2).mean()) for n, e in err.items()}})
    by_size = pd.DataFrame({n: e.abs().groupby(results["Empsize"]).mean() for n, e in err.items()}).T
    table = table.join(by_size.rename(columns=SIZES)).rename(index=MODELS)
    by_origin = pd.DataFrame({n: e.abs().groupby(results["origin"]).mean() for n, e in err.items()})
    return table.round(2), by_origin


def influence(train, x, coef):
    """Residuals, leverage and Cook's distance for the final training fit."""
    fitted = x @ coef
    resid = train["target"].to_numpy() - fitted
    hat = np.einsum("ij,ji->i", x, np.linalg.pinv(x.T @ x) @ x.T)
    p = x.shape[1]
    mse = resid @ resid / (len(resid) - p)
    cooks = resid ** 2 / (p * mse) * hat / (1 - hat) ** 2
    return pd.DataFrame({"Sector": train["Sector"].to_numpy(), "Empsize": train["Empsize"].to_numpy(),
                         "fitted": fitted, "resid": resid, "leverage": hat, "cooks": cooks})


def figures(pairs, by_origin, diag, median_se):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = dict(zip(SIZES, ["#193b54", "#317c89", "#9c6a27", "#725783", "#b44c45"]))
    OUT.mkdir(exist_ok=True)

    # Figure 1: predictor-outcome relationship (current rate vs rate three releases later).
    seen = pairs.dropna(subset=["rate", "target"])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for size, label in SIZES.items():
        s = seen[seen["Empsize"] == size]
        ax.scatter(s["rate"], s["target"], s=7, alpha=.55, color=colors[size], label=label, linewidths=0)
    ax.plot([0, 60], [0, 60], color="0.4", lw=.8, ls="--")
    ax.set(xlabel="AI use at issue release t (%)", ylabel="AI use at release t+3 (%)", xlim=(0, 60), ylim=(0, 60))
    ax.legend(title="Employees", fontsize=7, title_fontsize=7, frameon=False, loc="upper left", markerscale=1.6)
    ax.set_title(f"n = {len(seen):,} group-release pairs, {seen.groupby(['Sector','Empsize']).ngroups} groups",
                 fontsize=7.5, loc="left")
    fig.tight_layout(pad=.4)
    fig.savefig(OUT / "fig1_persistence.png", dpi=250)
    plt.close(fig)

    # Figure 2: (a) MAE by forecast origin vs survey noise; (b) residuals vs fitted, final training fit.
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for name, style in [("persistence", ":"), ("trailing_mean", "--"), ("regression", "-")]:
        a.plot(by_origin.index, by_origin[name], style, marker="o", ms=3, lw=1.3, label=MODELS[name], color="#193b54" if name == "regression" else "0.45")
    a.axhline(median_se, color="#b44c45", lw=.9)
    a.text(by_origin.index[0], median_se + .12, "median standard error of target", color="#b44c45", fontsize=7)
    a.set(xlabel="Issue date (2026)", ylabel="MAE (percentage points)", title="(a) Error by forecast issue date")
    a.legend(fontsize=6.8, frameon=False, loc="upper right")
    a.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    a.tick_params(axis="x", labelsize=7)
    a.title.set_size(8.5)
    for size, label in SIZES.items():
        d = diag[diag["Empsize"] == size]
        b.scatter(d["fitted"], d["resid"], s=6, alpha=.5, color=colors[size], label=label, linewidths=0)
    b.axhline(0, color="0.4", lw=.8)
    b.set(xlabel="Fitted AI use (%)", ylabel="Residual (pp)", title=f"(b) Residuals, final training fit (n = {len(diag):,})")
    b.title.set_size(8.5)
    fig.tight_layout(pad=.4)
    fig.savefig(OUT / "fig2_baseline.png", dpi=250)
    plt.close(fig)


def main():
    panel, dates, periods = load()
    qual = quality(panel)
    pairs, groups = build_pairs(panel)
    results, (train, x, coef) = evaluate(panel, pairs, groups, dates, periods)
    perf, by_origin = scores(results)
    diag = influence(train, x, coef)
    median_se = float(results["target_se"].median())
    figures(pairs, by_origin, diag, median_se)

    reg, trail = (results[n] - results["target"] for n in ["regression", "trailing_mean"])
    top = diag.nlargest(5, "cooks")
    # Sensitivity: refit the final model without its 5 most influential rows; compare final-origin test MAE.
    keep = ~diag.index.isin(top.index)
    coef_trim = np.linalg.lstsq(x[keep], train["target"].to_numpy()[keep], rcond=None)[0]
    final = results[results["origin"] == results["origin"].max()]
    xt = design(final, sorted({g[0] for g in groups}), sorted({g[1] for g in groups}))
    summary = {
        "sector_size_sha256": hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
        "releases": len(periods), "window": [FIRST, LAST], "horizon_releases": H,
        "cells_all_sectors": len(panel), "suppressed_cells": int(panel["rate"].isna().sum()),
        "screened_groups": len(groups),
        "groups_by_size": {SIZES[s]: sum(g[1] == s for g in groups) for s in SIZES},
        "pairs_screened": len(pairs), "pairs_complete": int(pairs.dropna(subset=["rate", "expected", "target"]).shape[0]),
        "scored_predictions": len(results), "origins": int(results["origin"].nunique()),
        "final_train_rows": len(train), "parameters": x.shape[1], "slope_on_current_rate": float(coef[1]),
        "regression_beats_trailing_at_origins": int((by_origin["regression"] < by_origin["trailing_mean"]).sum()),
        "mean_bias_pp": {n: float((results[n] - results["target"]).mean()) for n in MODELS},
        "median_target_se_pp": median_se,
        "share_regression_errors_within_2se": float((reg.abs() < 2 * results["target_se"]).mean()),
        "corr_abs_error_target_se": float(np.corrcoef(reg.abs(), results["target_se"])[0, 1]),
        "max_cooks": float(diag["cooks"].max()),
        "top_influential": top[["Sector", "Empsize", "resid", "cooks"]].round(3).to_dict("records"),
        "final_origin_mae_full_vs_trimmed": [float(np.abs(xt @ coef - final["target"]).mean()),
                                             float(np.abs(xt @ coef_trim - final["target"]).mean())],
    }
    qual.to_csv(OUT / "table1_quality.csv")
    perf.to_csv(OUT / "table2_performance.csv")
    by_origin.round(2).to_csv(OUT / "mae_by_origin.csv")
    results.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(qual, perf, json.dumps(summary, indent=2, default=str), sep="\n\n")


if __name__ == "__main__":
    main()
