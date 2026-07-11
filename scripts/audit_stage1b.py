"""
STAGE-1B FORENSIC AUDIT (Part A) — zero GPU, READ-ONLY. heat==TFF.

Answers A1-A4 of the audit instruction mechanically from the files in
--results-dir, and prints ONE-LINE ROOT CAUSES for every detected issue.
Nothing is modified, re-run, or fixed by this script.

  A1  Environment/path check: which RESULTS_DIR the E6/E7 runs actually wrote
      to (read back from each run JSON's recorded args.out_dir), how many E1
      JSONs are discoverable HERE under the current tag filters, raw-glob vs
      filtered counts (a gap = filter bug), and — if E1 is absent — a scan of
      sibling directories to locate where the E1 files actually live.
  A2  Trajectory forensics on one E6 cell (default gaussian_noise, eta=2e-3):
      the lambda=0 run's drift_l2/grad_l2/acc per step, side-by-side with the
      E1 Bernoulli p=0 run for the same cell; the executed lambda grid and the
      full choose_pstar trace (incl. the monotone/note fields that the
      campaign md dropped — the disambiguator for "p*=0 bracket [0,0]").
  A3  Same for one E7 cell (entropy drive, p=0 + reference runs): drift
      trajectory and whether the entropy loss actually produced gradients
      (grad_l2 > 0 and 0 <= energy <= ln(num_classes) varying per step).
  A4  Dispatch verification: for EVERY anchor/drive run JSON, the recorded
      args (heat_anchor_lambda / drive / heat_restore_prob / heat_lr /
      checkpoint / out_dir) vs what the filename tag claims. Missing keys =>
      the runner never received the flag (stale repo or foreign writer).

Also cross-checks the campaign's own e6_crossmech.json / e7_driveswap.json
claims against re-derived classifications (a mismatch = classification bug).

Usage:
  python scripts/audit_stage1b.py --results-dir <dir> \
      [--e1-results-dir <dir>] [--cell gaussian_noise:2e-3]
  python scripts/audit_stage1b.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts import stage1b_common as sb

FINDINGS: list[str] = []


def finding(tag, text):
    line = f"[{tag}] ROOT-CAUSE CANDIDATE: {text}"
    FINDINGS.append(line)
    print(f"  >>> {line}")


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1b forensic audit (Part A)")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--e1-results-dir", type=str, default="",
                   help="Where the E1 JSONs are believed to live, if different "
                        "from --results-dir.")
    p.add_argument("--cell", type=str, default="gaussian_noise:2e-3",
                   help="corruption:eta for the A2/A3 deep dive.")
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Trajectory summary
# ----------------------------------------------------------------------------

PROBE_STEPS = [0, 10, 25, 50, 75, 100, 125, 156]


def traj_summary(data):
    rows = sorted(pc.stream_rows(data),
                  key=lambda r: (r.get("local_step") or 0))
    summ = pc.heat_summary(data)
    out = {"n_rows": len(rows),
           "first_nan_step": pc.first_nonfinite_step(rows),
           "mean_acc": summ.get("mean_accuracy"),
           "last_acc": summ.get("last_accuracy"),
           "drift_probe": {}, "grad_probe": {}, "energy_probe": {}}
    finite_drifts, finite_grads, energies = [], [], []
    for r in rows:
        for key, acc in (("drift_l2", finite_drifts), ("grad_l2", finite_grads),
                         ("energy", energies)):
            v = r.get(key)
            try:
                if v is not None and math.isfinite(float(v)):
                    acc.append(float(v))
            except (TypeError, ValueError):
                pass
    bylocal = {int(r.get("local_step") or 0): r for r in rows}
    for s in PROBE_STEPS:
        r = bylocal.get(s)
        if r is None:
            continue
        out["drift_probe"][s] = r.get("drift_l2")
        out["grad_probe"][s] = r.get("grad_l2")
        out["energy_probe"][s] = r.get("energy")
    out["drift_max_finite"] = max(finite_drifts) if finite_drifts else None
    out["grad_mean_finite"] = (sum(finite_grads) / len(finite_grads)
                               if finite_grads else None)
    out["energy_min"] = min(energies) if energies else None
    out["energy_max"] = max(energies) if energies else None
    return out


def print_traj(label, t):
    print(f"    {label}: rows={t['n_rows']} first_nan={t['first_nan_step']} "
          f"mean_acc={_f(t['mean_acc'], 4)} last_acc={_f(t['last_acc'], 4)} "
          f"drift_max={_f(t['drift_max_finite'], 4)} "
          f"grad_mean={_f(t['grad_mean_finite'], 4)}")
    steps = sorted(t["drift_probe"])
    print("      step   : " + " ".join(f"{s:>9d}" for s in steps))
    print("      drift  : " + " ".join(f"{_f(t['drift_probe'][s], 3):>9s}" for s in steps))
    print("      grad   : " + " ".join(f"{_f(t['grad_probe'][s], 3):>9s}" for s in steps))
    print("      energy : " + " ".join(f"{_f(t['energy_probe'][s], 3):>9s}" for s in steps))


# ----------------------------------------------------------------------------
# A1 — environment / path / filter check
# ----------------------------------------------------------------------------

def recorded_args_of(path):
    try:
        return pc.load_json(path).get("args") or {}
    except Exception:
        return {}


def a1_paths(args, rdir: Path):
    print("\n================ A1 — ENVIRONMENT / PATH CHECK ================")
    print(f"  audited results dir: {rdir.resolve()}")

    # E1 discoverability under the CURRENT filters.
    n_e1, n_e1_src = 0, 0
    per_cell = {}
    for c in sc.E1_CORRUPTIONS:
        for eta in sc.E1_ETAS:
            ps = sc.discover_p_values_sc(rdir, sc.E1_ARCH, c, args.severity,
                                         args.seed, eta)
            per_cell[f"{c}:{pc.format_p(eta)}"] = len(ps)
            n_e1 += len(ps)
            if sc.run_output_path_sc(rdir, sc.E1_ARCH, c, args.severity,
                                     args.seed, source=True, eta=eta).exists():
                n_e1_src += 1
    raw_e1 = len(list(rdir.glob("p9_wrn28_10_pstar_*_lr*_p*_seed"
                                f"{args.seed}_sev{args.severity}.json")))
    raw_e1 -= len(list(rdir.glob("p9_wrn28_10_pstar_*_drive_*_seed"
                                 f"{args.seed}_sev{args.severity}.json")))
    n_anchor = len(list(rdir.glob("p9_*_anchor_lam*.json")))
    n_drive = len(list(rdir.glob("p9_*_drive_*.json")))
    n_gate = len(list(rdir.glob(f"p9_wrn28_10_pstar_lr*_p*_seed{args.seed}"
                                f"_sev{args.severity}.json")))
    print(f"  E1 heat runs discoverable by the CURRENT tag filter: {n_e1} "
          f"(per cell: {per_cell})")
    print(f"  E1 source runs present: {n_e1_src}/12 | continual (gate) runs: "
          f"{n_gate} | E6 anchor files: {n_anchor} | E7 drive files: {n_drive}")
    if raw_e1 > n_e1:
        finding("A1", f"tag-filter bug: raw glob sees {raw_e1} E1-like files "
                      f"but the filter recovers only {n_e1} — inspect names.")
    if n_e1 == 0:
        finding("A1", "ZERO E1 files here — explains '[A2] slope=None' and "
                      "'0 Bernoulli hard points'. The campaign dir does not "
                      "contain the E1 JSONs (path mismatch), so the analysis "
                      "was blind to the Bernoulli reference.")
        # scan siblings for where E1 actually lives
        parent = rdir.resolve().parent
        hits = []
        try:
            for sib in parent.iterdir():
                if sib.is_dir() and sib != rdir.resolve():
                    n = len(list(sib.glob("p9_wrn28_10_pstar_gaussian_noise_"
                                          "lr*_p*.json")))
                    if n:
                        hits.append((str(sib), n))
        except Exception:
            pass
        print(f"  sibling-dir scan for E1 files: {hits or 'none found'}")

    # Where did the campaign RUNS say they were written? (args.out_dir)
    outdirs, ckpts = {}, {}
    for f in list(rdir.glob("p9_*_anchor_lam*.json")) + \
             list(rdir.glob("p9_*_drive_*.json")):
        a = recorded_args_of(f)
        outdirs.setdefault(str(a.get("out_dir")), 0)
        outdirs[str(a.get("out_dir"))] += 1
        ckpts.setdefault(str(a.get("checkpoint")), 0)
        ckpts[str(a.get("checkpoint"))] += 1
    print(f"  out_dir recorded INSIDE E6/E7 run JSONs: {outdirs or 'n/a'}")
    print(f"  checkpoint recorded INSIDE E6/E7 run JSONs: {ckpts or 'n/a'}")
    # compare with checkpoints recorded in E1 files (if any here or in --e1 dir)
    e1dir = Path(args.e1_results_dir) if args.e1_results_dir else rdir
    e1_ck = {}
    for f in list(Path(e1dir).glob("p9_wrn28_10_pstar_*_lr*_p*.json"))[:40]:
        if "_drive_" in f.name or "anchor_lam" in f.name:
            continue
        a = recorded_args_of(f)
        e1_ck.setdefault(str(a.get("checkpoint")), 0)
        e1_ck[str(a.get("checkpoint"))] += 1
    print(f"  checkpoint recorded in E1 JSONs ({e1dir}): {e1_ck or 'n/a'}")
    if ckpts and e1_ck and set(ckpts) != set(e1_ck) and "n/a" not in ckpts:
        finding("A1", f"checkpoint path differs between E1 {sorted(e1_ck)} and "
                      f"E6/E7 {sorted(ckpts)} — a different source model would "
                      "change collapse behavior (lambda=0 stable becomes "
                      "possible). Verify artifact identity (size/hash).")
    return n_e1


# ----------------------------------------------------------------------------
# A2 — E6 lambda=0 forensics + choose_pstar trace
# ----------------------------------------------------------------------------

def choose_trace(rdir, args, corruption, eta, kind):
    """Print the boundary-resolution trace for one cell. kind='anchor'|'drive'."""
    source_acc = sc.source_acc_sc(rdir, sc.E1_ARCH, corruption, args.severity,
                                  args.seed, eta)
    if kind == "anchor":
        values = sb.discover_lambdas(rdir, sc.E1_ARCH, corruption,
                                     args.severity, args.seed, eta)
        loadf = lambda v: (pc.load_json(sb.run_output_path_anchor(
            rdir, sc.E1_ARCH, corruption, args.severity, args.seed, v, eta))
            if sb.run_output_path_anchor(rdir, sc.E1_ARCH, corruption,
                                         args.severity, args.seed, v,
                                         eta).exists() else None)
    else:
        values = sb.discover_drive_p_values(rdir, sc.E1_ARCH, corruption,
                                            args.severity, args.seed,
                                            "entropy", eta)
        loadf = lambda v: (pc.load_json(sb.run_output_path_drive(
            rdir, sc.E1_ARCH, corruption, args.severity, args.seed, "entropy",
            v, eta))
            if sb.run_output_path_drive(rdir, sc.E1_ARCH, corruption,
                                        args.severity, args.seed, "entropy",
                                        v, eta).exists() else None)
    points = {}
    print(f"    executed {kind} grid for {corruption} eta={pc.format_p(eta)}: "
          f"{[pc.format_p(v) for v in values]} (source_acc={_f(source_acc, 4)})")
    for v in values:
        data = loadf(v)
        rec = sb.store_point(points, v, data, source_acc)
        if rec["valid"]:
            rows = pc.stream_rows(data)
            print(f"      v={pc.format_p(v):>10s} hard="
                  f"{'COLLAPSE' if rec['collapsed'] else 'stable':8s} "
                  f"({rec['criterion']}) soft={rec['soft_criterion']:18s} "
                  f"acc={_f(rec['mean_acc'], 4)} "
                  f"first_nan={pc.first_nonfinite_step(rows)}")
        else:
            print(f"      v={pc.format_p(v):>10s} INVALID (unreadable/missing)")
    sel = sb.boundary_from_points(points)
    sel_soft = sb.boundary_from_points(points, use_soft=True)
    print(f"    choose_pstar(hard): p*={_f(sel['p_star'])} bracket="
          f"[{_f(sel['bracket_low'])},{_f(sel['bracket_high'])}] "
          f"monotone={sel['monotone']} note='{sel['note']}'")
    print(f"    choose_pstar(soft): p*={_f(sel_soft['p_star'])} "
          f"monotone={sel_soft['monotone']} note='{sel_soft['note']}'")
    if sel["p_star"] == 0.0 and sel["monotone"] is False:
        finding("A2" if kind == "anchor" else "A3",
                f"{corruption} eta={pc.format_p(eta)} ({kind}): 'p*=0 bracket "
                "[0,0]' is the NON-MONOTONE pathology — v=0 classified stable "
                "while LARGER tethers collapse. The campaign md dropped the "
                "monotone/note fields, so this printed as a legitimate zero.")
    return points, sel


def a2_e6(args, rdir, e1dir, corruption, eta):
    print("\n================ A2 — E6 lambda=0 TRAJECTORY FORENSICS "
          "================")
    lam0 = sb.run_output_path_anchor(rdir, sc.E1_ARCH, corruption,
                                     args.severity, args.seed, 0.0, eta)
    e1p0 = sc.run_output_path_sc(e1dir, sc.E1_ARCH, corruption, args.severity,
                                 args.seed, p=0.0, eta=eta)
    if lam0.exists():
        d6 = pc.load_json(lam0)
        t6 = traj_summary(d6)
        print(f"  (i) E6 lambda=0: {lam0.name}")
        print_traj("lambda=0", t6)
        drift_max = t6["drift_max_finite"]
        if (drift_max is not None and drift_max < 1e-6
                and (t6["grad_mean_finite"] or 0) > 1.0):
            finding("A2", "lambda=0 run: drift_l2 ~ 0 while grad_l2 is large — "
                          "parameters never moved (detached loss / optimizer "
                          "not stepping / variant mis-dispatch).")
        elif t6["first_nan_step"] is not None:
            src_acc = sc.source_acc_sc(rdir, sc.E1_ARCH, corruption,
                                       args.severity, args.seed, eta)
            hard, _ = sb.classify_both(d6, src_acc)
            if not hard["collapsed"]:
                finding("A2", "lambda=0 run DOES contain NaN but re-derived "
                              "hard classification says stable — "
                              "classification/key bug for variant runs.")
            else:
                print("    lambda=0 correctly re-classifies as HARD-COLLAPSE "
                      "here; if the campaign reported it stable, the bug is in "
                      "the campaign's analysis pass, not the data.")
        else:
            finding("A2", "lambda=0 run genuinely never diverged in this "
                          "session (no NaN, drift moves) — dynamics anomaly "
                          "vs E1: verify checkpoint identity, GPU/precision, "
                          "and that E1's p=0 for this cell truly hard-"
                          "collapsed (compare (ii)).")
    else:
        print(f"  (iii) E6 lambda=0 file DOES NOT EXIST: {lam0.name}")
        finding("A2", "lambda=0 was never run — 'lambda*=0, bracket [0,0]' was "
                      "produced without testing lambda=0 (orchestrator "
                      "boundary-resolution bug). See the executed grid below.")

    if e1p0.exists():
        t1 = traj_summary(pc.load_json(e1p0))
        print(f"  (ii) E1 Bernoulli p=0 (same cell): {e1p0.name}")
        print_traj("p=0 (E1)", t1)
        if lam0.exists():
            d0 = traj_summary(pc.load_json(lam0))
            if (t1["first_nan_step"] is not None
                    and d0["first_nan_step"] is None):
                finding("A2", "E1 p=0 hard-collapses but E6 lambda=0 does not, "
                              "on the same cell/seed — the two runs are NOT "
                              "dynamically identical in these sessions. "
                              "Localize via the drift rows above (checkpoint/"
                              "environment difference, or the lambda=0 run was "
                              "produced by a different code path).")
    else:
        print(f"  (ii) E1 p=0 JSON not found at {e1p0} — pass "
              f"--e1-results-dir to locate it; side-by-side skipped.")

    choose_trace(rdir, args, corruption, eta, "anchor")

    # Cross-check the campaign's own claims, if its analysis JSON exists.
    cm = rdir / "analysis" / "e6_crossmech.json"
    if cm.exists():
        try:
            claimed = {r["cell"]: r for r in
                       json.loads(cm.read_text(encoding="utf-8"))["rows"]}
            key = f"{corruption}_lr{pc.format_p(eta)}"
            if key in claimed:
                print(f"    campaign claimed for {key}: lambda*_hard="
                      f"{_f(claimed[key].get('lambda_star_hard'))} "
                      f"bracket={claimed[key].get('bracket_lambda')}")
        except Exception as exc:
            print(f"    (could not read e6_crossmech.json: {exc})")


def a3_e7(args, rdir, e1dir, corruption, eta):
    print("\n================ A3 — E7 entropy TRAJECTORY FORENSICS "
          "================")
    p0 = sb.run_output_path_drive(rdir, sc.E1_ARCH, corruption, args.severity,
                                  args.seed, "entropy", 0.0, eta)
    if p0.exists():
        d = pc.load_json(p0)
        t = traj_summary(d)
        print(f"  E7 entropy p=0: {p0.name}")
        print_traj("entropy p=0", t)
        gm = t["grad_mean_finite"]
        if gm is not None and gm < 1e-8:
            finding("A3", "entropy p=0 run: grad_l2 ~ 0 at every step — the "
                          "entropy loss did NOT backpropagate in the runner "
                          "(detached logits).")
        if (t["energy_min"] is not None and t["energy_max"] is not None
                and abs(t["energy_max"] - t["energy_min"]) < 1e-9):
            finding("A3", "entropy p=0 run: 'energy' (=entropy) is constant "
                          "across steps — loss frozen, adaptation inert.")
        if t["energy_max"] is not None and t["energy_max"] > math.log(10) + 0.05:
            finding("A3", "entropy run's 'energy' exceeds ln(10) — the loss "
                          "recorded is NOT prediction entropy; the drive flag "
                          "was likely ignored (dispatched to free-energy).")
    else:
        print(f"  E7 entropy p=0 file not found: {p0.name} (grid may not have "
              f"included p=0 if all predictions/refs failed — see trace).")
    choose_trace(rdir, args, corruption, eta, "drive")


# ----------------------------------------------------------------------------
# A4 — dispatch verification from recorded args
# ----------------------------------------------------------------------------

def a4_dispatch(args, rdir: Path):
    print("\n================ A4 — DISPATCH VERIFICATION (recorded args) "
          "================")
    bad = 0
    anchor_files = sorted(rdir.glob("p9_*_anchor_lam*.json"))
    drive_files = sorted(rdir.glob("p9_*_drive_*.json"))
    print(f"  anchor files: {len(anchor_files)} | drive files: {len(drive_files)}")
    for f in anchor_files:
        a = recorded_args_of(f)
        tag_lam = f.name.split("anchor_lam")[1].split("_seed")[0]
        rec_lam = a.get("heat_anchor_lambda", "<MISSING>")
        rec_drive = a.get("drive", "<MISSING>")
        rec_p = a.get("heat_restore_prob")
        ok = (rec_lam != "<MISSING>"
              and abs(float(rec_lam) - float(tag_lam)) < 1e-9
              and rec_drive in ("energy", "<MISSING>") and float(rec_p or 0) == 0.0)
        if not ok:
            bad += 1
            print(f"    [BAD] {f.name}: tag lam={tag_lam} args(lam={rec_lam}, "
                  f"drive={rec_drive}, restore={rec_p})")
            if rec_lam == "<MISSING>":
                finding("A4", f"{f.name}: runner never received/recorded "
                              "--heat-anchor-lambda — produced by a repo "
                              "WITHOUT the Stage-1b factory (stale clone), so "
                              "the 'anchor' runs were plain no-tether heat.")
    for f in drive_files:
        a = recorded_args_of(f)
        tag_p = f.name.split("_p")[-1].split("_seed")[0]
        rec_drive = a.get("drive", "<MISSING>")
        rec_p = a.get("heat_restore_prob")
        ok = rec_drive == "entropy" and rec_p is not None \
            and abs(float(rec_p) - float(tag_p)) < 1e-9
        if not ok:
            bad += 1
            print(f"    [BAD] {f.name}: tag p={tag_p} args(drive={rec_drive}, "
                  f"restore={rec_p})")
            if rec_drive != "entropy":
                finding("A4", f"{f.name}: args.drive={rec_drive!r} — the "
                              "entropy drive was NOT dispatched (stale repo "
                              "or flag dropped); the 'entropy' runs were the "
                              "free-energy drive.")
    if bad == 0 and (anchor_files or drive_files):
        print("  all variant files record the expected flags (dispatch "
              "reached the runner). If dynamics are still wrong, the bug is "
              "inside the variant classes, not the plumbing.")
    return bad


# ----------------------------------------------------------------------------
# self-test — synthetic scenarios exercising every detection branch
# ----------------------------------------------------------------------------

def self_test():
    import tempfile
    from scripts.analyze_stage1 import _synthetic_run, _synthetic_source
    global FINDINGS
    ok = True

    def scenario(name, build, expect_substrings):
        global FINDINGS
        FINDINGS = []
        print(f"\n##### scenario: {name} #####")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "analysis").mkdir()
            build(d)
            ns = argparse.Namespace(results_dir=str(d), e1_results_dir="",
                                    cell="gaussian_noise:2e-3", severity=5,
                                    seed=42, self_test=False)
            a1_paths(ns, d)
            a2_e6(ns, d, d, "gaussian_noise", 2e-3)
            a3_e7(ns, d, d, "gaussian_noise", 2e-3)
            a4_dispatch(ns, d)
        joined = "\n".join(FINDINGS)
        missed = [s for s in expect_substrings if s not in joined]
        print(f"##### {name}: {'DETECTED as expected' if not missed else 'MISSED ' + str(missed)}")
        return not missed

    def write(path, payload):
        path.write_text(json.dumps(payload))

    def anchor_file(d, lam, eta, collapsed, extra_args=None):
        p_eff = 0.0001 if collapsed else 0.05  # vs p*_true=eta*16/12
        payload = _synthetic_run("wrn28_10", "gaussian_noise", 5, 42, eta,
                                 p_eff, gbar=16.0, R=12.0)
        payload["args"].update({"heat_anchor_lambda": lam, "drive": "energy",
                                "heat_restore_prob": 0.0,
                                "out_dir": str(d), "checkpoint": "ck.pt"})
        if extra_args is not None:
            payload["args"] = extra_args
        write(sb.run_output_path_anchor(d, "wrn28_10", "gaussian_noise", 5,
                                        42, lam, eta), payload)

    def src_file(d, eta):
        write(sc.run_output_path_sc(d, "wrn28_10", "gaussian_noise", 5, 42,
                                    source=True, eta=eta),
              _synthetic_source("gaussian_noise"))

    # S1: empty-of-E1 dir (path mismatch) + non-monotone anchor cell.
    def build_s1(d):
        src_file(d, 2e-3)
        anchor_file(d, 0.0, 2e-3, collapsed=False)   # lam=0 "stable"
        anchor_file(d, 5.0, 2e-3, collapsed=True)    # larger lambdas collapse
        anchor_file(d, 10.0, 2e-3, collapsed=True)
    ok &= scenario("S1 path-mismatch + non-monotone [0,0]", build_s1,
                   ["ZERO E1 files", "NON-MONOTONE pathology"])

    # S2: lambda=0 never run (orchestrator gap).
    def build_s2(d):
        src_file(d, 2e-3)
        anchor_file(d, 5.0, 2e-3, collapsed=False)
    ok &= scenario("S2 lambda=0 missing", build_s2,
                   ["lambda=0 was never run"])

    # S3: dispatch failure — anchor file with args lacking the flag; entropy
    # file whose args.drive is 'energy'.
    def build_s3(d):
        src_file(d, 2e-3)
        anchor_file(d, 0.0, 2e-3, collapsed=True,
                    extra_args={"arch": "wrn28_10", "heat_lr": 2e-3,
                                "heat_restore_prob": 0.0})
        payload = _synthetic_run("wrn28_10", "gaussian_noise", 5, 42, 2e-3,
                                 0.01, gbar=9.5, R=12.0)
        payload["args"].update({"drive": "energy", "heat_restore_prob": 0.01})
        write(sb.run_output_path_drive(d, "wrn28_10", "gaussian_noise", 5, 42,
                                       "entropy", 0.01, 2e-3), payload)
    ok &= scenario("S3 stale-repo dispatch", build_s3,
                   ["--heat-anchor-lambda", "NOT dispatched"])

    # S4: entropy loss frozen (constant energy, zero grads).
    def build_s4(d):
        src_file(d, 2e-3)
        payload = _synthetic_run("wrn28_10", "gaussian_noise", 5, 42, 2e-3,
                                 0.0, gbar=9.5, R=12.0)
        for r in payload["results"]["summary"]["heat"]["stream_diagnostics"]:
            r["grad_l2"] = 0.0
            r["total_grad_norm"] = 0.0
            r["energy"] = 1.234
        payload["results"]["summary"]["heat"]["mean_accuracy"] = 0.7
        payload["args"].update({"drive": "entropy", "heat_restore_prob": 0.0})
        write(sb.run_output_path_drive(d, "wrn28_10", "gaussian_noise", 5, 42,
                                       "entropy", 0.0, 2e-3), payload)
    ok &= scenario("S4 entropy loss frozen", build_s4,
                   ["did NOT backpropagate", "constant across steps"])

    print(f"\nself-test: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.results_dir:
        raise SystemExit("--results-dir is required (or --self-test)")
    rdir = Path(args.results_dir)
    e1dir = Path(args.e1_results_dir) if args.e1_results_dir else rdir
    corruption, eta_s = args.cell.split(":")
    eta = float(eta_s)

    a1_paths(args, rdir)
    a2_e6(args, rdir, e1dir, corruption, eta)
    a3_e7(args, rdir, e1dir, corruption, eta)
    a4_dispatch(args, rdir)

    print("\n================ AUDIT SUMMARY (one-line root causes) "
          "================")
    if FINDINGS:
        for line in FINDINGS:
            print("  " + line)
    else:
        print("  no mechanical issue detected by A1-A4 filters — escalate to "
              "manual trajectory comparison (the side-by-side above).")
    out = rdir / "analysis" / "stage1b_audit.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Stage-1b forensic audit — findings\n\n"
                       + "\n".join(f"- {l}" for l in FINDINGS)
                       + "\n\n(Full traces are in the console log.)\n",
                       encoding="utf-8")
        print(f"[saved] {out}")
    except Exception:
        pass
    print("\nSTOP — Part A report complete. No fixes applied by this script.")


if __name__ == "__main__":
    main()
