"""Who should advisors contact, and when?"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from earlyflag.audit import calibration_table, capacity_curve, flag_top, group_audit, summary
from earlyflag.causal import e_value, ipw_risk_difference
from earlyflag.data import ENROLLMENT, FEATURE_SETS, add_derived, available, load
from earlyflag.model import make_model, out_of_fold
from earlyflag.plotting import BLUE, INK_2, MUTED, ORANGE, VIOLET, save, style

OUT = Path(__file__).resolve().parents[1] / "results"
CAPACITY = 0.25
SEED = 0


def main():
    OUT.mkdir(exist_ok=True)
    df = add_derived(load())
    res = {"students": len(df), "dropout_rate": float(df.dropout.mean()),
           "outcomes": df.target.value_counts().to_dict(), "capacity": CAPACITY}

    preds, perf = {}, {}
    for when, feats in FEATURE_SETS.items():
        feats = available(feats, df)
        for kind in ("logistic", "lightgbm"):
            p = out_of_fold(df, feats, kind, seed=SEED)
            preds[(when, kind)] = p
            perf[f"{when} | {kind}"] = summary(df.dropout, p) | {"features": len(feats)}
    res["performance"] = perf

    # sensitivity: finance flags may be recorded after the fact
    no_fin = available([f for f in FEATURE_SETS["After semester 1"] if f not in ("debtor", "tuition_fees_up_to_date")], df)
    preds[("Semester 1, no finance flags", "logistic")] = out_of_fold(df, no_fin, "logistic", seed=SEED)
    perf["Semester 1, no finance flags | logistic"] = summary(df.dropout, preds[("Semester 1, no finance flags", "logistic")])

    # one-line rule an advisor could run without a model: fewest courses passed first
    rule = (1 - df.sem1_pass_rate) + 1e-3 * (df.curricular_units_1st_sem_enrolled == 0)
    perf["Rule: lowest semester-1 pass rate"] = summary(df.dropout, rule, probabilities=False)

    caps = np.round(np.arange(0.05, 0.61, 0.05), 2)
    curves = {}
    for when in FEATURE_SETS:
        lg, gb = perf[f"{when} | logistic"]["pr_auc"], perf[f"{when} | lightgbm"]["pr_auc"]
        best = "lightgbm" if gb - lg > 0.01 else "logistic"
        curves[when] = (best, capacity_curve(df.dropout, preds[(when, best)], caps))
    curves["Rule: lowest semester-1 pass rate"] = ("rule", capacity_curve(df.dropout, rule, caps))
    curves["Semester 1, no finance flags"] = ("logistic", capacity_curve(df.dropout, preds[("Semester 1, no finance flags", "logistic")], caps))
    res["capacity"] = {when: {"model": m, "curve": c.to_dict("records")} for when, (m, c) in curves.items()}

    when = "After semester 1"
    model_name, _ = curves[when]
    p = preds[(when, model_name)]
    flags = flag_top(p, CAPACITY)
    groups = {
        "Gender": df.gender.map({0: "Female", 1: "Male"}),
        "Scholarship": df.scholarship_holder.map({0: "No scholarship", 1: "Scholarship"}),
        "Age at enrollment": pd.cut(df.age_at_enrollment, [0, 20, 25, 200], labels=["20 or under", "21 to 25", "Over 25"]),
        "Attendance": df.evening_attendance.map({0: "Daytime", 1: "Evening"}),
        "Displaced": df.displaced.map({0: "Not displaced", 1: "Displaced"}),
    }
    res["fairness"] = {name: group_audit(df.dropout, p, flags, g).to_dict("records") for name, g in groups.items()}
    res["calibration"] = {w: calibration_table(df.dropout, preds[(w, curves[w][0])]).to_dict("records") for w in FEATURE_SETS}
    early_leavers = (df.curricular_units_1st_sem_evaluations == 0)
    res["already_disengaged"] = {
        "share_of_students": float(early_leavers.mean()),
        "dropout_rate": float(df.loc[early_leavers, "dropout"].mean()),
        "share_of_flagged_dropouts": float((flags & early_leavers & (df.dropout == 1)).sum() / (flags & (df.dropout == 1)).sum()),
    }

    # what drives the semester-1 model
    feats = available(FEATURE_SETS[when], df)
    tr, te = train_test_split(df.index, test_size=0.3, stratify=df.dropout, random_state=SEED)
    m = make_model(model_name, feats, SEED).fit(df.loc[tr, feats], df.loc[tr, "dropout"])
    imp = permutation_importance(m, df.loc[te, feats], df.loc[te, "dropout"], scoring="average_precision",
                                 n_repeats=20, random_state=SEED)
    res["importance"] = (pd.DataFrame({"feature": feats, "drop_in_pr_auc": imp.importances_mean, "sd": imp.importances_std})
                         .sort_values("drop_in_pr_auc", ascending=False).head(12).to_dict("records"))

    # scholarships: an observational comparison, stated as such
    conf = [c for c in available(ENROLLMENT, df) if c != "scholarship_holder"]
    ipw = ipw_risk_difference(df, "scholarship_holder", "dropout", conf, n_boot=300)
    p1 = df.loc[df.scholarship_holder == 0, "dropout"].mean() + ipw["ipw_rd"]
    rr = p1 / df.loc[df.scholarship_holder == 0, "dropout"].mean()
    res["scholarship"] = ipw | {"rr_from_ipw": rr, "e_value": e_value(rr),
                                "dropout_scholarship": float(df.loc[df.scholarship_holder == 1, "dropout"].mean()),
                                "dropout_no_scholarship": float(df.loc[df.scholarship_holder == 0, "dropout"].mean())}

    (OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=float))
    plot_capacity(curves, df.dropout.mean())
    plot_calibration(res["calibration"])
    plot_fairness(res["fairness"])
    print(json.dumps({k: res[k] for k in ["performance", "scholarship", "already_disengaged"]}, indent=2, default=float))
    for w, (mname, c) in curves.items():
        print(w, mname)
        print(c.round(3).to_string(index=False))
    print(pd.DataFrame(res["importance"]).round(4).to_string(index=False))
    for name, rows in res["fairness"].items():
        print(name)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def plot_capacity(curves, base):
    style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    show = [("At enrollment", ORANGE, "At enrollment (model)"),
            ("Rule: lowest semester-1 pass rate", VIOLET, "After semester 1: fewest courses passed"),
            ("After semester 1", BLUE, "After semester 1 (model)")]
    for key, color, label in show:
        c = curves[key][1]
        ax.plot(c.capacity, c.recall, marker="o", markersize=4, color=color, label=label)
    xs = np.linspace(0.05, 0.6, 20)
    ax.plot(xs, xs, color=MUTED, lw=1.2, label="Random outreach")
    ax.axvline(CAPACITY, color=INK_2, lw=0.8)
    ax.text(CAPACITY + 0.008, 0.97, "advisors can see\n1 in 4 students", color=INK_2, fontsize=9, va="top")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of students advisors can contact")
    ax.set_ylabel("Share of eventual dropouts reached")
    ax.set_title("After one semester, a quarter of the outreach reaches two-thirds of dropouts", pad=22)
    ax.text(0, 1.03, f"Out-of-fold predictions for 4,424 students; base dropout rate {base:.0%}",
            transform=ax.transAxes, color=INK_2, fontsize=9)
    ax.legend(loc="lower right")
    save(fig, "capacity_curve.png")


def plot_calibration(cal):
    style()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, label="Perfect calibration")
    for (when, rows), color in zip(cal.items(), [ORANGE, BLUE]):
        t = pd.DataFrame(rows)
        ax.plot(t.predicted, t.observed, marker="o", markersize=5, color=color, label=when)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Predicted dropout risk (decile average)")
    ax.set_ylabel("Observed dropout rate")
    ax.set_title("A predicted 60% means about 60%", pad=22)
    ax.text(0, 1.03, "Each point is one tenth of students, grouped by predicted risk",
            transform=ax.transAxes, color=INK_2, fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(axis="x")
    save(fig, "calibration.png")


def plot_fairness(fair):
    style()
    rows = []
    for name, recs in fair.items():
        for r in recs:
            rows.append({"attr": name, **r})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    labels = [f"{r.group}" for r in t.itertuples()]
    y = np.arange(len(t))[::-1]
    for ax, col, color, title in [(axes[0], "recall", BLUE, "Dropouts reached (recall)"),
                                  (axes[1], "false_positive_rate", ORANGE, "Non-dropouts flagged (false positive rate)")]:
        ax.barh(y, t[col], color=color, height=0.6)
        for yi, v in zip(y, t[col]):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10, loc="left")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Error rates by group when advisors contact the top 25%", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, "fairness_audit.png")


if __name__ == "__main__":
    main()
