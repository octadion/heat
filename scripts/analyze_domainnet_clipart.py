"""
Analyze the DomainNet-126 eta-sweep (resnet50, source=real, target=clipart).

Answers THREE questions explicitly (reports all three; does not collapse into a
single pass/fail). NAMING: the code identifier is `heat`; here we label it TFF.

  Q1 (LAW): does p* scale linearly with eta*||g_bar|| on resnet50/DomainNet
     (a per-architecture R)? -> slope=1/R, intercept, R^2.
  Q2 (USEFULNESS): does TFF's best-over-p accuracy beat BN-adapt at ANY eta?
     -> yes/no + which eta. (Beating merely source is not enough.)
  Q3 (MECHANISM): at each collapse boundary, is it drift-collapse (drift_l2
     large/diverging -> the law's mechanism) or signal-collapse (accuracy
     craters to chance while drift_l2 stays bounded -> a different axis)?

Boundary = HARD collapse only (nan_inf OR mean_acc <= chance_acc ~= 0.02), since
soft:below_source fires at all p here. p* = min p that avoids hard collapse
(via pstar_common.choose_pstar on the hard-collapse labels). Confound check at
the top etas excludes optimizer-blowup points from the law fit.

Reuses pstar_common (unchanged collapse math / ||g_bar|| window). Outputs:
  <RESULTS_DIR>/analysis/domainnet_clipart_law.png
  <RESULTS_DIR>/analysis/domainnet_clipart_usefulness.{md,json}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc

ARCH = "resnet50"
SEV = 5
DRIFT_MULT = 5.0     # boundary drift > MULT * ref_drift => drift-collapse
R2_GOOD = 0.90


def parse_args():
    p = argparse.ArgumentParser(description="DomainNet clipart eta-sweep analysis (TFF)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chance-acc", type=float, default=0.02)
    p.add_argument("--etas", type=float, nargs="+", default=None,
                   help="Default: auto-discover every eta with runs on disk.")
    return p.parse_args()


def _load(path):
    return pc.load_json(path) if path.exists() else None


def _baseline_acc(results_dir, tag, method, seed):
    path = Path(results_dir) / f"p9_{ARCH}_{tag}_seed{seed}_sev{SEV}.json"
    data = _load(path)
    if data is None:
        return None
    return pc.safe_float(data.get("results", {}).get("summary", {})
                         .get(method, {}).get("mean_accuracy"))


def analyze_eta(results_dir, seed, eta, chance, gbar_baseline):
    p_values = pc.discover_p_values(results_dir, ARCH, SEV, seed, eta)
    points, per_p = [], {}
    for p in p_values:
        data = _load(pc.run_output_path(results_dir, ARCH, SEV, seed, p=p, eta=eta))
        if data is None:
            points.append({"p": p, "collapsed": False, "valid": False})
            continue
        v = pc.hard_collapse(data, chance)
        points.append({"p": p, "collapsed": v["collapsed"], "valid": True})
        per_p[p] = {"data": data, "v": v}

    sel = pc.choose_pstar(points)
    p_star, blo, bhi = sel["p_star"], sel["bracket_low"], sel["bracket_high"]

    # p_ref = smallest stable p -> ||g_bar|| + reference drift + per-run eta.
    stable_ps = sorted(p for p in per_p if not per_p[p]["v"]["collapsed"])
    p_ref = stable_ps[0] if stable_ps else None
    gbar = ref_drift = eta_used = None
    if p_ref is not None:
        gr = pc.grad_norm_gbar(per_p[p_ref]["data"])
        gbar = gr[0] if gr else None
        ref_drift = pc.drift_stationary(per_p[p_ref]["data"])
        eta_used = pc.heat_lr_of(per_p[p_ref]["data"])
    if eta_used is None:
        eta_used = eta

    # TFF best-over-p accuracy + argmax.
    accs = [(p, pc.safe_float(per_p[p]["v"]["mean_acc"])) for p in per_p
            if per_p[p]["v"]["mean_acc"] is not None]
    tff_best_acc, argmax_p = (max(accs, key=lambda t: t[1])[::-1]
                              if accs else (None, None))

    # Mechanism at the collapsing edge (bracket_low).
    mechanism, drift_at_boundary, boundary_crit = "none", None, "none"
    if blo is not None and blo in per_p and per_p[blo]["v"]["collapsed"]:
        bv = per_p[blo]["v"]
        boundary_crit = bv["criterion"]
        drift_at_boundary = bv["drift"]
        if boundary_crit == "hard:nan_inf":
            mechanism = "drift-collapse"          # params diverged
        else:  # hard:chance_acc
            if (drift_at_boundary is not None and ref_drift not in (None, 0)
                    and drift_at_boundary <= DRIFT_MULT * ref_drift):
                mechanism = "signal-collapse"     # chance acc, bounded drift
            else:
                mechanism = "drift-collapse"

    # Confound: is the p=0 run an optimizer blow-up (exclude from law fit)?
    p0 = per_p.get(0.0, per_p.get(min(per_p) if per_p else None, None))
    cf = pc.confound_flag(p0["data"] if p0 else None, gbar, gbar_baseline)

    return {
        "eta": eta_used, "grad_norm_gbar": gbar,
        "eta_times_gbar": (gbar * eta_used) if gbar is not None else None,
        "p_star": p_star, "p_star_bracket_low": blo, "p_star_bracket_high": bhi,
        "p_ref": p_ref, "ref_drift": ref_drift,
        "boundary_mechanism": mechanism, "drift_at_boundary": drift_at_boundary,
        "boundary_criterion": boundary_crit,
        "tff_best_acc": tff_best_acc, "argmax_p": argmax_p,
        "confound": cf["flag"], "confound_detail": cf,
        "acc_by_p": {pc.format_p(p): pc.safe_float(per_p[p]["v"]["mean_acc"])
                     for p in sorted(per_p)},
        "n_p": len(per_p),
    }


def make_plot(rows, source_acc, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    fit_pts = [(r["eta_times_gbar"], r["p_star"]) for r in rows
               if r["confound"] != "optimizer_blowup" and r["eta_times_gbar"] is not None
               and r["p_star"] is not None]
    # color by mechanism
    cmap = {"drift-collapse": "#d62728", "signal-collapse": "#1f77b4", "none": "#7f7f7f"}
    for r in rows:
        x, y = r["eta_times_gbar"], r["p_star"]
        if x is None or y is None:
            continue
        blow = r["confound"] == "optimizer_blowup"
        ax.scatter([x], [y], s=80, zorder=3,
                   color=cmap.get(r["boundary_mechanism"], "#7f7f7f"),
                   marker=("x" if blow else "o"))
        ax.annotate(f"η={r['eta']:g}" + (" (blowup)" if blow else ""),
                    (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    fit = pc.least_squares_line([x for x, _ in fit_pts], [y for _, y in fit_pts])
    if fit is not None:
        xs = [0.0, max(x for x, _ in fit_pts) * 1.1]
        ax.plot(xs, [fit["slope"] * x + fit["intercept"] for x in xs],
                "--", color="0.3", zorder=2,
                label=(f"fit: slope={fit['slope']:.3g} (R={1.0/fit['slope']:.3g}), "
                       f"b={fit['intercept']:.3g}, R²={fit['r2']:.3f}"))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0); ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\eta \cdot \|\bar{g}\|$   (swept via $\eta$)")
    ax.set_ylabel(r"$p^*$ (min tether avoiding HARD collapse)")
    ax.set_title("TFF drift-budget law — resnet50 / DomainNet-126 (real→clipart)\n"
                 "red=drift-collapse, blue=signal-collapse, ×=optimizer-blowup")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return fit


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    rd = Path(args.results_dir)
    analysis = rd / "analysis"; analysis.mkdir(parents=True, exist_ok=True)

    source_acc = _baseline_acc(rd, "pstar_source", "source", args.seed)
    bn_adapt_acc = _baseline_acc(rd, "pstar_bnadapt", "bn_adapt", args.seed)

    etas = sorted(set(args.etas)) if args.etas else pc.discover_etas(rd, ARCH, SEV, args.seed)
    # baseline gbar for the confound ratio = eta closest to 1e-3.
    gbar_base = None
    if etas:
        e0 = min(etas, key=lambda e: abs(e - pc.ETA))
        r0 = analyze_eta(rd, args.seed, e0, args.chance_acc, None)
        gbar_base = r0["grad_norm_gbar"]

    rows = [analyze_eta(rd, args.seed, e, args.chance_acc, gbar_base) for e in etas]
    rows.sort(key=lambda r: r["eta"])

    fit = make_plot(rows, source_acc, analysis / "domainnet_clipart_law.png")

    # ---- Q2: TFF best-over-p (per eta) vs BN-adapt ----
    tff_beats_bn = [r for r in rows if r["tff_best_acc"] is not None
                    and bn_adapt_acc is not None and r["tff_best_acc"] > bn_adapt_acc]
    best_row = max([r for r in rows if r["tff_best_acc"] is not None],
                   key=lambda r: r["tff_best_acc"], default=None)

    # ---- Q3: mechanism counts (boundaries only) ----
    n_drift = sum(1 for r in rows if r["boundary_mechanism"] == "drift-collapse")
    n_signal = sum(1 for r in rows if r["boundary_mechanism"] == "signal-collapse")
    n_blow = sum(1 for r in rows if r["confound"] == "optimizer_blowup")

    def f(v, nd=4):
        return "n/a" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    # ---- Usefulness table ----
    cols = ["eta", "eta*||g_bar||", "p*", "boundary_mechanism", "TFF_best_acc",
            "bn_adapt_acc", "source_acc", "argmax_p", "drift_at_boundary", "confound"]
    md = [f"# TFF on resnet50 / DomainNet-126 (real→clipart) — usefulness & law\n",
          f"source_acc = {f(source_acc)} | bn_adapt_acc = {f(bn_adapt_acc)}\n",
          "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md.append("| " + " | ".join([
            f"{r['eta']:g}", f(r["eta_times_gbar"]), f(r["p_star"]),
            r["boundary_mechanism"], f(r["tff_best_acc"]), f(bn_adapt_acc),
            f(source_acc), f(r["argmax_p"]), f(r["drift_at_boundary"]), r["confound"],
        ]) + " |")

    # ---- Three-line verdict ----
    R = (1.0 / fit["slope"]) if (fit and fit["slope"]) else None
    q1 = (f"Q1 (LAW): {'LINEAR' if (fit and fit['r2'] is not None and fit['r2']>=R2_GOOD) else 'NOT clean'} "
          f"— slope={f(fit['slope']) if fit else 'n/a'} (R={f(R)}), "
          f"R²={f(fit['r2']) if fit else 'n/a'} over {fit['n'] if fit else 0} genuine points.")
    if best_row is not None and bn_adapt_acc is not None:
        q2 = (f"Q2 (USEFULNESS): TFF best_acc={f(best_row['tff_best_acc'])} "
              f"(eta={best_row['eta']:g}) vs bn_adapt={f(bn_adapt_acc)} — "
              f"{'BEATS BN at ' + str(len(tff_beats_bn)) + ' eta(s) => TFF WINS (Branch 1)' if tff_beats_bn else 'never beats BN => GRACEFUL FALLBACK only (Branch 2)'}.")
    else:
        q2 = "Q2 (USEFULNESS): insufficient data (missing TFF or bn_adapt accuracy)."
    q3 = (f"Q3 (MECHANISM): {n_drift} drift-collapse, {n_signal} signal-collapse boundaries"
          f"{f' (+{n_blow} optimizer-blowup excluded)' if n_blow else ''}. "
          f"{'p* comparable to CIFAR (drift-budget).' if n_drift>=n_signal and n_drift>0 else ('signal-collapse dominates => p* measures a DIFFERENT axis than CIFAR.' if n_signal>0 else 'no genuine boundary triggered.')}")

    verdict = "\n".join([q1, q2, q3])
    md += ["", "## Verdict", "", q1, "", q2, "", q3]
    (analysis / "domainnet_clipart_usefulness.md").write_text("\n".join(md), encoding="utf-8")
    (analysis / "domainnet_clipart_usefulness.json").write_text(json.dumps({
        "source_acc": source_acc, "bn_adapt_acc": bn_adapt_acc,
        "fit": (None if fit is None else {"slope": fit["slope"], "R": R,
                "intercept": fit["intercept"], "r2": fit["r2"], "n": fit["n"]}),
        "rows": rows, "q1": q1, "q2": q2, "q3": q3,
    }, indent=2), encoding="utf-8")

    print("\n================ DomainNet clipart — table ================\n")
    print("\n".join(md[2:4 + len(rows)]))
    print("\n================ VERDICT (Q1/Q2/Q3) ================\n")
    print(verdict)
    print(f"\n[saved] {analysis/'domainnet_clipart_law.png'}")
    print(f"[saved] {analysis/'domainnet_clipart_usefulness.md'}")
    print(f"[saved] {analysis/'domainnet_clipart_usefulness.json'}")


if __name__ == "__main__":
    main()
