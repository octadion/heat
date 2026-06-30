"""
Analyze the drift-budget-law (p*) sweep and emit the decisive deliverable.

Reads all run JSONs in RESULTS_DIR (written by run_pstar_sweep.py via
run_tier2.py), then for each (arch, severity, eta) cell:
  * measures ||g_bar|| from the known-stable reference run (section 6.1),
  * derives p* and its bracket from the collapse criterion (sections 6.2-6.3),
  * uses the run's OWN eta (pc.heat_lr_of) for the x-axis, not a constant,
and fits a per-architecture least-squares line of p* vs (eta * ||g_bar||),
pooling all eta-cells of that arch -- the eta-sweep test of the law.

Outputs (exactly the three artifacts in section 7):
  <RESULTS_DIR>/analysis/pstar_law.json   -- machine-readable table
  <RESULTS_DIR>/analysis/pstar_law.png    -- the plot (one line per arch)
  <RESULTS_DIR>/analysis/pstar_verdict.md -- numeric verdict

Also prints a Markdown table and the verdict to stdout (the notebook displays
both inline). This script is the SINGLE SOURCE OF TRUTH for the verdict; it
recomputes p* from every run present and never silently drops a cell. It also
flags any p* whose collapsing edge was a SOFT criterion as a possible confound.

Usage:
  python scripts/analyze_pstar_law.py \
      --results-dir /content/drive/MyDrive/pstar_results \
      --archs wrn28_10 --severities 5 --etas 2e-4 5e-4 1e-3 2e-3 4e-3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc


# R^2 threshold for "linear", and intercept-smallness fraction (section 7c).
R2_GOOD = 0.90
INTERCEPT_FRAC = 0.20  # |intercept| small if <= this * y-range


def parse_args():
    p = argparse.ArgumentParser(description="Fit and judge the p* drift-budget law")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--archs", type=str, nargs="+", default=list(pc.DEFAULT_ARCHS),
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--severities", type=int, nargs="+", default=list(pc.DEFAULT_SEVERITIES))
    p.add_argument("--etas", type=float, nargs="+", default=None,
                   help="Learning rates to include. Default: auto-discover every "
                        "eta with runs on disk for each (arch, severity).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_p_run(results_dir, arch, severity, seed, p, eta):
    path = pc.run_output_path(results_dir, arch, severity, seed, p=p, eta=eta)
    if not path.exists():
        return None
    try:
        return pc.load_json(path)
    except Exception:
        return None


def derive_pref(results_dir, arch, severity, seed, eta, source_acc):
    """Read-only re-derivation of the reference p_ref (mirrors the sweep).

    Uses the eta-SCALED ladder and eta-tagged paths. Stability uses hard
    collapse + below-source only (the drift-ratio test is defined relative to
    p_ref). Returns (p_ref, pref_drift, gbar)."""
    candidates = pc.scaled_pref_ladder(arch, eta)
    last = (None, None, None)
    for cand in candidates:
        data = load_p_run(results_dir, arch, severity, seed, cand, eta)
        if data is None:
            continue
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        drift = pc.drift_stationary(data)
        last = (cand, drift, gbar)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False)
        if not v["collapsed"]:
            return cand, drift, gbar
    return last  # no stable reference found; best effort


def analyze_cell(results_dir, arch, severity, eta, seed):
    """Build the full row for one (arch, severity, eta) cell."""
    src_path = pc.run_output_path(results_dir, arch, severity, seed,
                                  source=True, eta=eta)
    source_acc = None
    if src_path.exists():
        try:
            source_acc = pc.source_mean_acc(pc.load_json(src_path))
        except Exception:
            source_acc = None

    p_ref, pref_drift, gbar = derive_pref(results_dir, arch, severity, seed,
                                          eta, source_acc)

    p_values = pc.discover_p_values(results_dir, arch, severity, seed, eta)
    points = []
    criteria: dict[float, str] = {}
    mean_accs: dict[float, float] = {}
    eta_used = None
    for p in p_values:
        data = load_p_run(results_dir, arch, severity, seed, p, eta)
        if data is None:
            points.append({"p": p, "collapsed": False, "valid": False})
            continue
        # x-axis uses the run's OWN eta (CHANGE A), not the hardcoded pc.ETA.
        if eta_used is None:
            eta_used = pc.heat_lr_of(data)
        v = pc.classify_run(data, source_acc, pref_drift)
        points.append({"p": p, "collapsed": v["collapsed"], "valid": True})
        criteria[p] = v["criterion"]
        if v["mean_acc"] is not None:
            mean_accs[p] = v["mean_acc"]

    sel = pc.choose_pstar(points)
    p_star = sel["p_star"]
    blo, bhi = sel["bracket_low"], sel["bracket_high"]

    # collapse criterion that defines p* = the criterion of the run at the
    # collapsing edge of the bracket (or none if p*~0 / undefined).
    if blo is not None and blo in criteria and any(
        pt["p"] == blo and pt["collapsed"] for pt in points
    ):
        collapse_criterion = criteria.get(blo, "unknown")
    elif p_star == 0.0:
        collapse_criterion = "none(p*~0)"
    else:
        collapse_criterion = sel.get("note") or "none"

    stable_mean_acc = mean_accs.get(bhi) if bhi is not None else None

    if eta_used is None:
        eta_used = eta  # no valid run loaded; fall back to the intended eta
    eta_gbar = (gbar * eta_used) if gbar is not None else None

    return {
        "arch": arch,
        "severity": severity,
        "source_acc": source_acc,
        "p_ref_used": p_ref,
        "grad_norm_gbar": gbar,
        "eta": eta_used,
        "eta_times_gbar": eta_gbar,
        "p_star": p_star,
        "p_star_bracket_low": blo,
        "p_star_bracket_high": bhi,
        "collapse_criterion": collapse_criterion,
        "stable_mean_acc": stable_mean_acc,
        "monotone_bracket": sel["monotone"],
        "bracket_note": sel["note"],
        "p_values_run": p_values,
    }


def fit_arch(rows: list[dict]):
    """Least-squares fit of p* vs eta*||g_bar|| for one arch's rows."""
    usable = [r for r in rows
              if r["eta_times_gbar"] is not None and r["p_star"] is not None]
    xs = [r["eta_times_gbar"] for r in usable]
    ys = [r["p_star"] for r in usable]
    fit = pc.least_squares_line(xs, ys)
    return fit, xs, ys, usable


def monotone_in_x(rows: list[dict]) -> bool:
    """Is p* non-decreasing as eta*||g_bar|| (the x-axis) increases? This is the
    meaningful monotonicity for the eta-sweep (eta is the lever, not severity)."""
    seq = [(r["eta_times_gbar"], r["p_star"]) for r in rows
           if r.get("eta_times_gbar") is not None and r.get("p_star") is not None]
    seq.sort()
    vals = [y for _x, y in seq]
    return all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def make_plot(per_arch_fit: dict, out_png: Path):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = {"resnet18": "#1f77b4", "wrn28_10": "#d62728"}
    for arch, info in per_arch_fit.items():
        fit, xs, ys, usable = info["fit"], info["xs"], info["ys"], info["rows"]
        c = colors.get(arch, None)
        etas = [r["eta"] for r in usable]
        ax.scatter(xs, ys, color=c, s=70, zorder=3,
                   label=f"{arch} (etas {[f'{e:g}' for e in sorted(set(etas))]})")
        for x, y, e in zip(xs, ys, etas):
            ax.annotate(f"η={e:g}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=c)

        # Censored cells: p* unresolved ("law region exhausted" -- every tested
        # tether collapsed) but ||g_bar|| is known. Show them as lower-bound
        # up-arrows at p* > bracket_low so the plot never hides a real point.
        # NOT included in the least-squares fit.
        censored = [r for r in info.get("all_rows", [])
                    if r.get("p_star") is None and r.get("eta_times_gbar") is not None]
        if censored:
            cx = [r["eta_times_gbar"] for r in censored]
            cy = [r["p_star_bracket_low"] if r.get("p_star_bracket_low") is not None
                  else 0.0 for r in censored]
            ax.scatter(cx, cy, marker="^", s=110, facecolors="none",
                       edgecolors=c, linewidths=1.6, zorder=3,
                       label=f"{arch} censored (p* > bound, excl. from fit)")
            for x, y, r in zip(cx, cy, censored):
                ax.annotate(f"η={r['eta']:g}↑", (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=c)
        if fit is not None:
            xmin, xmax = 0.0, max(xs) * 1.1 if xs else 1.0
            xx = [xmin, xmax]
            yy = [fit["slope"] * x + fit["intercept"] for x in xx]
            slope = fit["slope"]
            R = (1.0 / slope) if slope not in (0, None) else float("inf")
            r2 = fit["r2"]
            r2s = f"{r2:.3f}" if r2 is not None else "n/a"
            ax.plot(xx, yy, color=c, linestyle="--", linewidth=1.6, zorder=2,
                    label=(f"  fit: slope={slope:.3g} (R={R:.3g}), "
                           f"b={fit['intercept']:.3g}, R²={r2s}"))
    ax.set_xlabel(r"$\eta \cdot \|\bar{g}\|$   (swept via $\eta$, the lever)")
    ax.set_ylabel(r"$p^*$  (minimum tether to avoid collapse)")
    ax.set_title("Drift-budget law (eta-sweep):  $p^* \\approx (\\eta\\,\\|\\bar g\\|)/R$")
    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_verdict(per_arch_fit: dict, all_rows: list[dict]) -> tuple[str, str]:
    """Return (verdict_label, markdown_text)."""
    lines = []
    arch_status = {}
    for arch, info in per_arch_fit.items():
        fit, ys = info["fit"], info["ys"]
        rows = info["rows"]
        if fit is None or len(ys) < 2:
            lines.append(
                f"- **{arch}**: only {len(ys)} usable point(s) "
                f"(need >=2 to fit a line); cannot assess linearity."
            )
            arch_status[arch] = {"good": False, "linear": False, "fit": fit}
            continue
        slope = fit["slope"]
        R = (1.0 / slope) if slope not in (0, None) else float("inf")
        r2 = fit["r2"] if fit["r2"] is not None else float("nan")
        yspan = (max(ys) - min(ys)) if len(ys) > 1 else max(ys)
        ref = max(yspan, max(ys), 1e-9)
        intercept_small = abs(fit["intercept"]) <= INTERCEPT_FRAC * ref
        mono_p = monotone_in_x(rows)
        linear = (not math.isnan(r2)) and r2 >= R2_GOOD
        good = linear and intercept_small
        # CHANGE E: flag p* points whose collapsing edge was a SOFT criterion
        # (soft:below_source / soft:drift_blowup) -- a possible confound. For
        # WRN sev5 the boundary should normally be hard:nan_inf.
        soft_rows = [r for r in rows
                     if str(r.get("collapse_criterion", "")).startswith("soft:")]
        arch_status[arch] = {"good": good, "linear": linear,
                             "intercept_small": intercept_small,
                             "mono_p": mono_p, "soft_rows": soft_rows,
                             "R": R, "r2": r2, "fit": fit}
        lines.append(
            f"- **{arch}**: slope={slope:.4g} (=> R={R:.4g}), "
            f"intercept={fit['intercept']:.4g} "
            f"({'small' if intercept_small else 'NON-trivial'} vs y-range {ref:.4g}), "
            f"R²={r2:.3f}. "
            f"p* {'monotone' if mono_p else 'NON-monotone'} in eta*||g_bar||."
        )
        if soft_rows:
            tags = ", ".join(f"eta={r['eta']:g}:{r['collapse_criterion']}"
                             for r in soft_rows)
            lines.append(
                f"    ⚠ soft-criterion boundary at [{tags}] — possible confound; "
                f"a clean signal should be hard:nan_inf (esp. WRN sev5)."
            )

    n_good = sum(1 for s in arch_status.values() if s.get("good"))
    n_linear = sum(1 for s in arch_status.values() if s.get("linear"))
    n_arch = len(arch_status)
    any_nonmono = any(s.get("mono_p") is False for s in arch_status.values())

    if n_arch >= 1 and n_good == n_arch:
        verdict = "LAW SUPPORTED"
        why = (f"All {n_arch} architecture(s) are linear (R²>={R2_GOOD}) with a "
               f"small intercept. p* tracks eta*||g_bar|| as predicted across the "
               f"eta sweep; each arch has its own slope 1/R, consistent with R "
               f"being per-architecture.")
    elif n_linear >= 1:
        verdict = "LAW PARTIAL"
        bad = [a for a, s in arch_status.items() if not s.get("good")]
        why = (f"{n_good}/{n_arch} architecture(s) clean; concern(s) on: "
               f"{', '.join(bad) if bad else 'none'}. ")
        causes = []
        for a, s in arch_status.items():
            if s.get("good"):
                continue
            if s.get("fit") is None:
                causes.append(f"{a}: too few usable points")
            elif not s.get("linear"):
                causes.append(f"{a}: low R² ({s.get('r2'):.3f})")
            elif not s.get("intercept_small"):
                causes.append(f"{a}: non-trivial intercept")
            if s.get("mono_p") is False:
                causes.append(f"{a}: p* not monotone in eta*||g_bar||")
            if s.get("soft_rows"):
                causes.append(f"{a}: soft-criterion boundary on "
                              f"{len(s['soft_rows'])} point(s) (possible confound)")
        if causes:
            why += "Likely cause(s): " + "; ".join(causes) + "."
    else:
        verdict = "LAW NOT SUPPORTED"
        why = ("No architecture shows a clean line through the origin. ")
        if any_nonmono:
            why += ("Most likely cause: p* is not monotone in eta*||g_bar||, so "
                    "the predicted scaling does not hold. ")
        else:
            why += "Points scatter; p* does not track eta*||g_bar|| linearly. "

    md = []
    md.append(f"# Drift-budget law verdict (eta-sweep): {verdict}\n")
    md.append("Prediction: p* ~= (eta * ||g_bar||) / R, with R a per-architecture "
              "constant. At fixed arch + severity, ||g_bar|| is ~constant near the "
              "source, so sweeping eta moves the x-axis ~linearly and p* should "
              "scale linearly with eta -- a straight line through the origin "
              "(slope 1/R).\n")
    md.extend(lines)
    md.append("")
    md.append(f"**Verdict: {verdict}.** {why}")
    return verdict, "\n".join(md)


def print_table(rows: list[dict]):
    cols = ["arch", "severity", "eta", "source_acc", "p_ref_used", "grad_norm_gbar",
            "eta_times_gbar", "p_star", "p_star_bracket_low",
            "p_star_bracket_high", "collapse_criterion", "stable_mean_acc"]

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for arch in args.archs:
        for sev in args.severities:
            # Each (arch, severity, eta) is its own p* cell. Use the explicit
            # --etas if given, else auto-discover every eta with runs on disk.
            etas = args.etas if args.etas else pc.discover_etas(
                results_dir, arch, sev, args.seed)
            for eta in sorted(set(etas)):
                row = analyze_cell(results_dir, arch, sev, eta, args.seed)
                all_rows.append(row)

    # Per-arch fits.
    per_arch_fit = {}
    for arch in args.archs:
        rows = [r for r in all_rows if r["arch"] == arch]
        fit, xs, ys, usable = fit_arch(rows)
        per_arch_fit[arch] = {"fit": fit, "xs": xs, "ys": ys, "rows": usable,
                              "all_rows": rows}

    make_plot(per_arch_fit, analysis_dir / "pstar_law.png")
    verdict, verdict_md = build_verdict(per_arch_fit, all_rows)

    # Write artifacts.
    law_json = {
        "etas": sorted({r["eta"] for r in all_rows if r.get("eta") is not None}),
        "r2_good_threshold": R2_GOOD,
        "rows": all_rows,
        "fits": {
            arch: (None if per_arch_fit[arch]["fit"] is None else {
                "slope": per_arch_fit[arch]["fit"]["slope"],
                "intercept": per_arch_fit[arch]["fit"]["intercept"],
                "R": (1.0 / per_arch_fit[arch]["fit"]["slope"]
                      if per_arch_fit[arch]["fit"]["slope"] else None),
                "r2": per_arch_fit[arch]["fit"]["r2"],
                "n_points": per_arch_fit[arch]["fit"]["n"],
            })
            for arch in args.archs
        },
        "verdict": verdict,
    }
    (analysis_dir / "pstar_law.json").write_text(
        json.dumps(law_json, indent=2), encoding="utf-8")
    (analysis_dir / "pstar_verdict.md").write_text(verdict_md, encoding="utf-8")

    # Console output (notebook displays these).
    print("\n================ p* DRIFT-BUDGET LAW: TABLE ================\n")
    print_table(all_rows)
    print("\n================ VERDICT ================\n")
    print(verdict_md)
    print(f"\n[saved] {analysis_dir / 'pstar_law.json'}")
    print(f"[saved] {analysis_dir / 'pstar_law.png'}")
    print(f"[saved] {analysis_dir / 'pstar_verdict.md'}")


if __name__ == "__main__":
    main()
