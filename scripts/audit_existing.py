"""
Three ZERO-GPU audits — re-analysis of existing run JSONs only. No new runs / data.

NAMING: the code identifier is `heat`; all reporting here says TFF.

Reuses the validated analysis code (scripts/pstar_common, scripts/analyze_pstar_law)
— does NOT touch src/, the collapse math, or heat.py.

  Audit 1: is the DomainNet real->clipart "optimizer_blowup" real, or mislabeled
           drift-collapse? (trajectory SHAPE of drift/grad/energy vs step)
  Audit 2: is ||g_bar|| ~constant across eta in the CIFAR sweeps? (clean-lever test)
  Audit 3: consolidate all CIFAR law evidence (WRN+ResNet, soft+hard) into one
           master figure + table.

Any needed-but-missing JSON is listed, NOT re-run.

Usage:
  python scripts/audit_existing.py \
      --cifar-results-dir  /content/drive/MyDrive/pstar_results \
      --domainnet-results-dir /content/drive/MyDrive/pstar_domainnet_clipart \
      --out-dir /content/drive/MyDrive/pstar_domainnet_clipart/analysis --seed 42
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc
from scripts import analyze_pstar_law as apl

MISSING: list[str] = []
DN_ARCH, DN_SEV = "resnet50", 5
CIFAR = [("wrn28_10", 5), ("resnet18", 5)]
EARLY_STEP = 50            # NaN before this (with small drift) => optimizer blow-up
CV_CLEAN = 0.15            # ||g_bar|| CV below this => eta is a clean single lever
R2_STRONG = 0.99


def _f(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def _load(path):
    if not Path(path).exists():
        MISSING.append(str(path))
        return None
    try:
        return pc.load_json(Path(path))
    except Exception:
        MISSING.append(str(path) + " (unreadable)")
        return None


# ===========================================================================
# AUDIT 1 — clipart blow-up vs drift-collapse (trajectory shape)
# ===========================================================================

def _series(rows, key):
    xs, ys = [], []
    for r in sorted(rows, key=lambda r: (r.get("step") if r.get("step") is not None else 1 << 60)):
        v = r.get(key)
        s = r.get("step")
        if s is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float("nan")
        xs.append(int(s)); ys.append(fv)
    return xs, ys


def audit1(dn_dir, out_dir, seed, blowup_etas=(0.004, 0.008, 0.016), stable_eta=0.001):
    print("\n================ AUDIT 1 — clipart blow-up vs drift ================")
    etas = list(blowup_etas) + [stable_eta]
    fig, axes = plt.subplots(len(etas), 1, figsize=(8, 2.6 * len(etas)), sharex=True)
    if len(etas) == 1:
        axes = [axes]
    verdicts = {}
    # reference stable drift magnitude (from the stable eta's p=0 run)
    for ax, eta in zip(axes, etas):
        data = _load(pc.run_output_path(dn_dir, DN_ARCH, DN_SEV, seed, p=0.0, eta=eta))
        if data is None:
            ax.set_title(f"eta={eta:g}  [MISSING p=0 run]"); continue
        rows = pc.stream_rows(data)
        info = pc.first_nonfinite_info(rows)
        sx, sdrift = _series(rows, "drift_l2")
        _, sgrad = _series(rows, "grad_l2")
        _, sen = _series(rows, "energy")
        ax.plot(sx, sdrift, label="drift_l2", color="#1f77b4")
        ax.plot(sx, sgrad, label="grad_l2", color="#d62728", alpha=0.8)
        ax2 = ax.twinx(); ax2.plot(sx, sen, label="energy", color="#2ca02c", alpha=0.5)
        ax2.set_ylabel("energy", fontsize=7)
        fn = info["step"] if info else None
        # drift just before divergence
        pre = [d for s, d in zip(sx, sdrift) if (fn is None or s < fn) and d == d]  # d==d: finite
        pre_drift = max(pre) if pre else None
        if fn is not None:
            ax.axvline(fn, color="k", ls=":", lw=1)
            verdict = ("optimizer-blowup" if fn < EARLY_STEP else "drift-collapse")
        else:
            verdict = "no-divergence"
        verdicts[eta] = {"first_nan_step": fn, "pre_nan_drift": pre_drift, "verdict": verdict}
        ax.set_title(f"eta={eta:g}  first-NaN step={fn}  pre-NaN max drift={_f(pre_drift)}  "
                     f"-> {verdict}", fontsize=9)
        ax.set_ylabel("drift / grad"); ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("adaptation step")
    fig.suptitle("Audit 1 — TFF real→clipart (resnet50): trajectory shape at high eta",
                 fontsize=11)
    fig.tight_layout()
    out = Path(out_dir) / "audit1_clipart_blowup.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)

    blow = [e for e in blowup_etas if verdicts.get(e, {}).get("verdict") == "optimizer-blowup"]
    drift = [e for e in blowup_etas if verdicts.get(e, {}).get("verdict") == "drift-collapse"]
    if drift:
        concl = (f"MISLABELED DRIFT-COLLAPSE at eta={drift}: drift rose over many steps "
                 f"then grad/energy diverged late -> the law IS measurable on DomainNet; "
                 f"re-fit p* INCLUDING these points.")
    elif blow:
        concl = (f"GENUINE OPTIMIZER-BLOWUP at eta={blow}: grad/energy exploded in the first "
                 f"<{EARLY_STEP} steps before drift accumulated -> Q1/law unmeasurable on "
                 f"clipart (signal-/LR-dominated regime confirmed); excluding them is correct.")
    else:
        concl = ("No divergence found in the inspected p=0 runs (blow-up flag was via the "
                 "||g_bar|| ratio, not a NaN) -> inspect the ||g_bar|| magnitudes; not a "
                 "step-wise divergence.")
    for e in blowup_etas:
        v = verdicts.get(e, {})
        print(f"  eta={e:g}: first-NaN={v.get('first_nan_step')} pre-NaN drift="
              f"{_f(v.get('pre_nan_drift'))} -> {v.get('verdict')}")
    print(f"  [saved] {out}")
    print(f"  CONCLUSION: {concl}")
    return {"per_eta": verdicts, "conclusion": concl, "png": str(out)}


# ===========================================================================
# AUDIT 2 — ||g_bar|| vs eta (CIFAR); is eta a clean lever?
# ===========================================================================

def audit2(cifar_dir, out_dir, seed):
    print("\n================ AUDIT 2 — ||g_bar|| vs eta (CIFAR) ================")
    fig, ax = plt.subplots(figsize=(7, 5))
    md = ["# Audit 2 — TFF ||g_bar|| vs eta (CIFAR eta-sweeps)\n",
          "| arch | eta | ||g_bar|| |", "| --- | --- | --- |"]
    per_arch = {}
    for arch, sev in CIFAR:
        etas = pc.discover_etas(cifar_dir, arch, sev, seed)
        if not etas:
            MISSING.append(f"{cifar_dir}: no eta-tagged runs for {arch} sev{sev}")
            continue
        gbars, xs = [], []
        for eta in etas:
            row = apl.analyze_cell(cifar_dir, arch, sev, eta, seed)
            g = row["grad_norm_gbar"]
            md.append(f"| {arch} | {eta:g} | {_f(g)} |")
            if g is not None:
                gbars.append(g); xs.append(eta)
        if len(gbars) >= 2:
            mean = statistics.mean(gbars); sd = statistics.pstdev(gbars)
            cv = sd / mean if mean else float("inf")
            per_arch[arch] = {"gbars": gbars, "etas": xs, "mean": mean, "cv": cv}
            ax.plot(xs, gbars, "o-", label=f"{arch} (CV={cv*100:.1f}%)")
    ax.set_xscale("log"); ax.set_xlabel("eta (log)"); ax.set_ylabel("stationary ||g_bar||")
    ax.set_title("Audit 2 — TFF ||g_bar|| vs eta (CIFAR)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "audit2_gbar_vs_eta.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)

    lines = []
    for arch, s in per_arch.items():
        clean = s["cv"] < CV_CLEAN
        lines.append(f"{arch}: ||g_bar|| mean={_f(s['mean'])}, CV={s['cv']*100:.1f}% -> "
                     f"{'CONSTANT (eta is a clean lever)' if clean else 'VARIES with eta'}")
    if per_arch and all(s["cv"] < CV_CLEAN for s in per_arch.values()):
        concl = ("||g_bar|| ~constant across eta (all CV<%.0f%%): eta is a clean single "
                 "lever; the x-axis moved essentially via eta -> eta-sweep narrative holds."
                 % (CV_CLEAN * 100))
    else:
        concl = ("||g_bar|| VARIES with eta: p* vs the MEASURED eta*||g_bar|| product can "
                 "still be linear (we plot the measured product), BUT eta and ||g_bar|| are "
                 "NOT independent, so the eta-sweep does not isolate the eta factor. An "
                 "INDEPENDENT ||g_bar|| test (vary shift at fixed eta, e.g. DomainNet across "
                 "domains) is required to test the ||g_bar|| factor separately.")
    md += ["", "## Conclusion", ""] + [f"- {l}" for l in lines] + ["", concl]
    (Path(out_dir) / "audit2_gbar_vs_eta.md").write_text("\n".join(md), encoding="utf-8")
    for l in lines:
        print("  " + l)
    print(f"  [saved] {out}")
    print(f"  CONCLUSION: {concl}")
    return {"per_arch": {a: {"mean": s["mean"], "cv": s["cv"]} for a, s in per_arch.items()},
            "conclusion": concl, "png": str(out)}


# ===========================================================================
# AUDIT 3 — master law artifact (CIFAR, WRN+ResNet, soft+hard)
# ===========================================================================

def audit3(cifar_dir, out_dir, seed):
    print("\n================ AUDIT 3 — master law artifact (CIFAR) ================")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    md = ["# Audit 3 — TFF drift-budget law master (CIFAR)\n",
          "| arch | criterion | n | eta range | ||g_bar|| range | slope | R | intercept | R² | x-range factor |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    series = []
    colors = {"wrn28_10": "#d62728", "resnet18": "#1f77b4"}
    styles = {False: "-", True: "--"}
    for arch, sev in CIFAR:
        etas = pc.discover_etas(cifar_dir, arch, sev, seed)
        if not etas:
            MISSING.append(f"{cifar_dir}: no eta-tagged runs for {arch} sev{sev}"); continue
        for hard_only in (False, True):
            crit = "hard-only" if hard_only else "soft+hard"
            pts, gbars, es = [], [], []
            for eta in etas:
                row = apl.analyze_cell(cifar_dir, arch, sev, eta, seed, hard_only=hard_only)
                x, y, g = row["eta_times_gbar"], row["p_star"], row["grad_norm_gbar"]
                if x is not None and y is not None:
                    pts.append((x, y))
                if g is not None:
                    gbars.append(g)
                es.append(eta)
            fit = pc.least_squares_line([p[0] for p in pts], [p[1] for p in pts])
            xr = (max(p[0] for p in pts), min(p[0] for p in pts)) if pts else (None, None)
            xfac = (xr[0] / xr[1]) if (xr[1] and xr[1] > 0) else None
            slope = fit["slope"] if fit else None
            R = (1.0 / slope) if slope else None
            r2 = fit["r2"] if fit else None
            series.append({"arch": arch, "criterion": crit, "n": len(pts),
                           "slope": slope, "R": R, "r2": r2, "xfac": xfac})
            md.append(f"| {arch} | {crit} | {len(pts)} | "
                      f"{_f(min(es))}–{_f(max(es))} | "
                      f"{_f(min(gbars)) if gbars else 'n/a'}–{_f(max(gbars)) if gbars else 'n/a'} | "
                      f"{_f(slope)} | {_f(R)} | {_f(fit['intercept']) if fit else 'n/a'} | "
                      f"{_f(r2)} | {_f(xfac)}x |")
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=colors.get(arch), s=55,
                       marker=("o" if not hard_only else "^"),
                       label=f"{arch} / {crit} (n={len(pts)})")
            if fit and pts:
                xx = [0.0, max(xs) * 1.05]
                rtxt = f"{R:.3g}" if R else "inf"
                r2t = f"{r2:.3f}" if r2 is not None else "n/a"
                ax.plot(xx, [slope * x + fit["intercept"] for x in xx],
                        styles[hard_only], color=colors.get(arch), alpha=0.7,
                        label=f"    fit R={rtxt}, R²={r2t}")
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel(r"$\eta \cdot \|\bar g\|$"); ax.set_ylabel(r"$p^*$")
    ax.set_title("Audit 3 — TFF drift-budget law master (CIFAR): p* vs η·‖ḡ‖")
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = Path(out_dir) / "audit3_law_master.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)

    strong = [s for s in series if s["r2"] is not None and s["r2"] >= R2_STRONG]
    archs_strong = sorted({s["arch"] for s in strong})
    xfacs = [s["xfac"] for s in strong if s["xfac"]]
    concl = (f"Law holds at R²>={R2_STRONG} on {len(strong)}/{len(series)} (arch,criterion) "
             f"series across {len(archs_strong)} architecture(s) {archs_strong}, over an "
             f"x-range factor up to {_f(max(xfacs)) if xfacs else 'n/a'}x."
             if strong else
             f"No (arch,criterion) series reached R²>={R2_STRONG} — the law is weaker than "
             f"remembered; inspect the table.")
    md += ["", "## Summary", "", concl]
    (Path(out_dir) / "audit3_law_master.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[1:3 + len(series)]))
    print(f"  [saved] {out}")
    print(f"  SUMMARY: {concl}")
    return {"series": series, "summary": concl, "png": str(out)}


def parse_args():
    p = argparse.ArgumentParser(description="Three zero-GPU re-analysis audits (TFF)")
    p.add_argument("--cifar-results-dir", type=str, required=True)
    p.add_argument("--domainnet-results-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--which", type=str, default="all", choices=["1", "2", "3", "all"])
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    r1 = r2 = r3 = None
    if a.which in ("1", "all"):
        r1 = audit1(a.domainnet_results_dir, a.out_dir, a.seed)
    if a.which in ("2", "all"):
        r2 = audit2(a.cifar_results_dir, a.out_dir, a.seed)
    if a.which in ("3", "all"):
        r3 = audit3(a.cifar_results_dir, a.out_dir, a.seed)

    print("\n================ COMBINED SUMMARY ================")
    if r1:
        print(f"Audit 1 (clipart blow-up): {r1['conclusion']}")
    if r2:
        print(f"Audit 2 (||g_bar|| vs eta): {r2['conclusion']}")
    if r3:
        print(f"Audit 3 (law strength): {r3['summary']}")
    if MISSING:
        print("\n[MISSING JSONs — NOT re-run; decide manually]:")
        for m in MISSING:
            print("  -", m)
    else:
        print("\n[all required JSONs present]")
    Path(a.out_dir, "audit_summary.json").write_text(json.dumps(
        {"audit1": r1, "audit2": r2, "audit3": r3, "missing": MISSING}, indent=2),
        encoding="utf-8")
    print(f"\n[saved] {Path(a.out_dir,'audit_summary.json')}")


if __name__ == "__main__":
    main()
