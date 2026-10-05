"""Paper figures, called by analyze.py. Each figure is written as .pdf (for the paper) and .png (for a quick look).

  heatmap_proceed_<variant>   P(final = PROCEED) for every cell x model: the whole result on one page
  trajectory_<variant>        P(PROCEED) after each turn, per scenario: where models cave or back off
  leaderboard                 violation rate vs over-caution rate per model (arrow: neutral -> ihl_aware)
  forest_<variant>            odds ratios of the pre-registered factor tests, per model and pooled
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, NA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
GROUP_COLOR = {"standard": "#2a78d6", "uncensored": "#eb6834"}
OTHER = "#898781"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                    "#256abf", "#184f95", "#0d366b"])
PHASE_LABELS = ["brief", "order", "new info", "pressure", "final"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "savefig.bbox": "tight", "figure.facecolor": "white", "pdf.fonttype": 42,
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def model_order(df):
    g = df.drop_duplicates("model").set_index("model")["group"].fillna("")
    return sorted(g.index, key=lambda m: (g[m] != "standard", m))


def short(sid):
    return sid.split("_")[0].upper()


def cell_rows(scenarios, present):
    """(scenario, cond, label) in YAML order; the label marks reference cells (+ violation if PROCEED,
    * over-cautious if ABORT), which is where the headline metrics come from."""
    from runner import cells, reference
    rows = []
    for sid, s in scenarios.items():
        for lv in cells(s):
            cond = ",".join(f"{k}={v}" for k, v in lv.items())
            if (sid, cond) not in present:
                continue
            ref = reference(s, lv)
            mark = (" †" if ref["violation"] else "") + (" *" if ref["overcautious"] else "")
            rows.append((sid, cond, f"{short(sid)}  {', '.join(lv.values())}{mark}"))
    return rows


def heatmap(df, by_cell, scenarios, out):
    for variant, t in by_cell.groupby("variant"):
        models = model_order(df[df.variant == variant])
        rows = cell_rows(scenarios, set(zip(t.scenario, t.cond)))
        if not rows or not models:
            continue
        rate = t.set_index(["scenario", "cond", "model"])["rate"]
        M = np.array([[rate.get((s, c, m), np.nan) for m in models] for s, c, _ in rows])
        fig, ax = plt.subplots(figsize=(2.6 + 0.42 * len(models), 0.9 + 0.2 * len(rows)))
        cmap = BLUES.copy()
        cmap.set_bad(NA)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(models)), models, rotation=40, ha="right")
        ax.set_yticks(range(len(rows)), [r[2] for r in rows])
        # 2px surface gaps between cells and a wider one between scenarios
        ax.set_xticks(np.arange(-.5, len(models)), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(i - .5, color="white", linewidth=4)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("P(final decision = PROCEED)")
        cb.outline.set_visible(False)
        ax.set_title(f"Final recommendation by cell ({variant} system prompt)", loc="left", pad=22)
        ax.text(0, 1.005, "† PROCEED not IHL-defensible here (draft reference labels)\n"
                "* ABORT over-cautious (lawful target)    grey = no parsed runs", transform=ax.transAxes,
                va="bottom", color=MUTED, fontsize=6.5)
        save(fig, out, f"heatmap_proceed_{variant}")


def trajectory_plot(traj, scenarios, out):
    for variant, t in traj.groupby("variant"):
        sids = [s for s in scenarios if s in set(t.scenario)]
        fig, axes = plt.subplots(1, len(sids), figsize=(1.9 * len(sids) + 0.6, 2.4), sharey=True, squeeze=False)
        x = np.arange(len(PHASE_LABELS))
        for ax, sid in zip(axes[0], sids):
            ts = t[t.scenario == sid]
            for _, r in ts.iterrows():
                ax.plot(x, [r[p] for p in ["brief", "order", "new_info", "pressure", "final"]], linewidth=1,
                        alpha=0.35, color=GROUP_COLOR.get(r["group"], OTHER), solid_capstyle="round")
            for g, tg in ts.groupby("group"):
                mean = tg[["brief", "order", "new_info", "pressure", "final"]].mean()
                ax.plot(x, mean.values, linewidth=2, color=GROUP_COLOR.get(g, OTHER), marker="o", markersize=4,
                        markeredgecolor="white", markeredgewidth=1, label=f"{g} (mean)", solid_capstyle="round")
            ax.set_title(f"{short(sid)}  {scenarios[sid]['id'].split('_', 1)[1].replace('_', ' ')}", loc="left",
                         fontsize=8)
            ax.set_xticks(x, PHASE_LABELS, rotation=45, ha="right")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("P(PROCEED) after turn")
        axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Recommendation over the conversation ({variant}); thin lines = single models",
                     x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        save(fig, out, f"trajectory_{variant}")


def leaderboard_plot(lb, out):
    lb = lb[lb.variant.isin(["neutral", "ihl_aware"])].dropna(subset=["violation_rate", "overcautious_rate"])
    if lb.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m, d in lb.groupby("model"):
        d = d.set_index("variant")
        color = GROUP_COLOR.get(d["group"].iloc[0], OTHER)
        if {"neutral", "ihl_aware"} <= set(d.index):
            a, b = d.loc["neutral"], d.loc["ihl_aware"]
            ax.annotate("", xy=(b.overcautious_rate, b.violation_rate), xytext=(a.overcautious_rate, a.violation_rate),
                        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8, "alpha": 0.6})
        for v, r in d.iterrows():
            ax.errorbar(r.overcautious_rate, r.violation_rate, color=color, alpha=0.35, linewidth=0.6,
                        xerr=[[r.overcautious_rate - r.overcautious_lo], [r.overcautious_hi - r.overcautious_rate]],
                        yerr=[[r.violation_rate - r.violation_lo], [r.violation_hi - r.violation_rate]])
            ax.plot(r.overcautious_rate, r.violation_rate, "o", markersize=6, markeredgewidth=1.2,
                    color=color if v == "neutral" else "white", markeredgecolor=color, zorder=3)
        lab = d.loc["neutral"] if "neutral" in d.index else d.iloc[0]
        ax.annotate(m, (lab.overcautious_rate, lab.violation_rate), xytext=(5, 3), textcoords="offset points",
                    fontsize=6.5, color=INK2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("Over-caution: ABORT where the strike is plainly lawful")
    ax.set_ylabel("Violation: PROCEED where not IHL-defensible")
    ax.set_title("Lower left is better", loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in GROUP_COLOR.items()
               if g in set(lb.group)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=MUTED, label="neutral prompt"),
                plt.Line2D([], [], marker="o", linestyle="", color="white", markeredgecolor=MUTED, label="ihl_aware prompt")]
    ax.legend(handles=handles, loc="upper right")
    save(fig, out, "leaderboard")


def forest(hyp, df, out):
    if hyp.empty:
        return
    groups = df.drop_duplicates("model").set_index("model")["group"].to_dict()
    for variant, t in hyp.groupby("variant"):
        tests = list(dict.fromkeys(zip(t.hypothesis, t.scenario, t.factor)))
        models = model_order(df) + ["ALL"]
        models = [m for m in models if m in set(t.model)]
        ncol = min(5, len(tests))
        nrow = math.ceil(len(tests) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.9, 0.5 + (0.17 * len(models) + 0.5) * nrow),
                                 sharey=True, squeeze=False)
        for ax in axes.flat[len(tests):]:
            ax.set_visible(False)
        for ax, (h, sid, f) in zip(axes.flat, tests):
            ts = t[(t.hypothesis == h) & (t.scenario == sid) & (t.factor == f)].set_index("model")
            for i, m in enumerate(models):
                if m not in ts.index:
                    continue
                r = ts.loc[m]
                color = INK if m == "ALL" else GROUP_COLOR.get(groups.get(m), OTHER)
                lo, hi = np.clip([r.or_lo, r.or_hi], 1e-3, 1e3)
                ax.plot([lo, hi], [i, i], color=color, linewidth=1, solid_capstyle="round")
                ax.plot(r.odds_ratio if np.isfinite(r.odds_ratio) else hi, i, "D" if m == "ALL" else "o",
                        markersize=4.5, color=color if r.supported else "white", markeredgecolor=color,
                        markeredgewidth=1, zorder=3)
            ax.axvline(1, color=AXIS, linewidth=0.8, zorder=0)
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1e3)
            ax.set_xticks([1e-2, 1, 1e2], ["0.01", "1", "100"])
            ax.minorticks_off()
            sign = ts["predicted"].iloc[0] if len(ts) else ""
            ax.set_title(f"{h}  {short(sid) if sid != 'all' else ''} {f} ({sign})", loc="left", fontsize=7.5)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
        axes[0][0].set_yticks(range(len(models)), models)
        axes[0][0].invert_yaxis()
        for row in axes:
            row[0].tick_params(axis="y", length=0)
        fig.supxlabel("odds ratio for P(PROCEED), 95% profile CI (log scale)", fontsize=8, color=INK2)
        fig.suptitle(f"Hypothesis tests ({variant})\nfilled = supported after Holm correction",
                     x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        save(fig, out, f"forest_{variant.replace(' ', '_')}")


def make_all(df, by_cell, lb, hyp, traj, scenarios, out):
    heatmap(df, by_cell, scenarios, out)
    trajectory_plot(traj, scenarios, out)
    leaderboard_plot(lb, out)
    forest(hyp, df, out)
