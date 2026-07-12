"""
STAGE 1 RESTART (v2) — clean, verified, provenance-safe protocol layer.
heat==TFF. ADDITIVE module: heat.py, the runner, pstar_common's frozen math,
and all completed continual JSONs are untouched.

PROTOCOL LOCK (STEP 1 — the root cause of the prior failure):
  The E1 stream is ONE corruption's severity-5 split CYCLED x15 (~2355 steps
  at batch 64), matching the continual horizon so genuine hard collapse can
  occur (validated: gaussian x15, p=0, eta=1e-3 -> first NaN = 601). The cycle
  count is a RECORDED protocol field (in the tag AND the embedded fingerprint),
  never a hidden default. Single-pass 157-step streams are FORBIDDEN for law
  measurement. Cycling is implemented by repeating the corruption in
  run_tier2.py's --corruptions list (natively supported); the per_corruption
  summary key collides across cycles (benign: mean/last accuracy degrade to
  the LAST cycle's accuracy, which is exactly the conservative quantity the
  hard/soft criteria need), while stream_diagnostics accumulate correctly and
  carry everything the analysis reads.

PROVENANCE GUARDS (STEP 2 — permanent):
  (a) require_drive_dir: RESULTS_DIR must resolve under /content/drive/... ,
      else ABORT unless --allow-ephemeral; prints the absolute resolved path
      and performs a write+read sentinel check BEFORE any run.
  (b) Every run JSON gets an embedded protocol_fingerprint {n_steps,
      corruption, cycle_count, eta, p, checkpoint_hash, seed, batch_size}
      (injected atomically by the orchestrator right after the run lands —
      run_tier2 itself is untouched). Every analysis artifact embeds the
      manifest (paths + hashes) of the run files it consumed. An analyzer
      with zero discoverable reference runs emits VOID, never a curve verdict
      (rule inherited from stage1b_common).
  (c) End of campaign: Drive-side file count vs expected; gaps are flagged.

Naming: variant tag pstar_cyc<N>_<corruption>_lr<eta>_p<p> (source:
pstar_cyc<N>_<corruption>_lr<eta>_source). The cyc token makes every prior
discovery pattern (single-pass E1 `pstar_<corr>_lr`, continual `pstar_lr`,
anchor `_anchor_lam`, drive `_drive_`) unable to match these files and vice
versa (covered by the self-test).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts.analysis_common import stationary_rows, row_metric, safe_float

# ----------------------------------------------------------------------------
# Protocol constants (STEP 1 / STEP 3).
# ----------------------------------------------------------------------------

CYCLES = 15                       # locked protocol field, recorded everywhere
V2_ARCH = "wrn28_10"
V2_SEVERITY = 5
V2_ETAS: list[float] = [5e-4, 1e-3, 2e-3]
CALIBRATION_CORRUPTIONS = ["gaussian_noise", "elastic_transform"]  # fit R here
HELDOUT_CORRUPTIONS = ["impulse_noise", "contrast"]                # never fit
V2_CORRUPTIONS = CALIBRATION_CORRUPTIONS + HELDOUT_CORRUPTIONS

STATIONARY_LO, STATIONARY_HI = 50, 150   # the existing window, per cycle-block


# ----------------------------------------------------------------------------
# Run identity.
# ----------------------------------------------------------------------------

def variant_tag_v2(corruption: str, p: Optional[float] = None,
                   source: bool = False, eta: Optional[float] = None,
                   cycles: int = CYCLES) -> str:
    eta_part = f"lr{pc.format_p(eta)}_" if eta is not None else ""
    base = f"pstar_cyc{cycles}_{corruption}_{eta_part}"
    if source:
        return base + "source"
    if p is None:
        raise ValueError("variant_tag_v2 requires p when source=False")
    return base + f"p{pc.format_p(p)}"


def run_output_path_v2(results_dir, arch, corruption, severity, seed,
                       p=None, source=False, eta=None,
                       cycles: int = CYCLES) -> Path:
    tag = variant_tag_v2(corruption, p=p, source=source, eta=eta, cycles=cycles)
    return Path(results_dir) / f"p9_{arch}_{tag}_seed{seed}_sev{severity}.json"


def discover_p_values_v2(results_dir, arch, corruption, severity, seed, eta,
                         cycles: int = CYCLES) -> list[float]:
    prefix = f"p9_{arch}_pstar_cyc{cycles}_{corruption}_lr{pc.format_p(eta)}_p"
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(re.escape(prefix) + r"(.+?)" + re.escape(suffix) + r"$")
    out = []
    for path in Path(results_dir).glob(f"{prefix}*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
    return sorted(set(out))


def build_run_command_v2(*, run_tier2, arch, checkpoint, corruption, severity,
                         seed, results_dir, c10c_root, heat_lr,
                         batch_size=64, num_workers=2, p=None, source=False,
                         cycles: int = CYCLES, python=None) -> list[str]:
    """Cycled stream = the corruption repeated `cycles` times in --corruptions.
    Source runs use ONE cycle (no adaptation => every cycle is identical); the
    fingerprint records the actual cycle count."""
    python = python or sys.executable
    n_cyc = 1 if source else cycles
    cmd = [
        python, str(run_tier2), "--protocol", "p9", "--arch", arch,
        "--dataset", "cifar10", "--checkpoint", str(checkpoint),
        "--c10c-root", str(c10c_root), "--severity", str(severity),
        "--seed", str(seed), "--batch-size", str(batch_size),
        "--num-workers", str(num_workers), "--out-dir", str(results_dir),
        "--corruptions", *([str(corruption)] * n_cyc),
    ]
    if source:
        cmd += ["--methods", "source",
                "--variant-tag", variant_tag_v2(corruption, source=True,
                                                eta=heat_lr, cycles=cycles)]
    else:
        cmd += ["--methods", "heat",
                "--heat-lr", repr(float(heat_lr)),
                "--heat-restore-prob", pc.format_p(p if p is not None else 0.0),
                "--heat-diagnostic-snapshot",
                "--variant-tag", variant_tag_v2(corruption, p=p, eta=heat_lr,
                                                cycles=cycles)]
    return cmd


# ----------------------------------------------------------------------------
# STEP 2(a) — Drive-path guard + sentinel.
# ----------------------------------------------------------------------------

def require_drive_dir(results_dir, allow_ephemeral: bool = False) -> Path:
    resolved = Path(results_dir).resolve()
    as_posix = str(resolved).replace("\\", "/")
    on_drive = as_posix.startswith("/content/drive/")
    print(f"[guard a] RESULTS_DIR resolves to: {resolved}"
          f"  (drive-backed={on_drive})", flush=True)
    if not on_drive and not allow_ephemeral:
        raise SystemExit(
            "[ABORT — provenance guard] RESULTS_DIR does not resolve under "
            "/content/drive/... — results would be EPHEMERAL and lost on "
            "disconnect (the prior failure). Point it at a Drive path, or "
            "pass --allow-ephemeral to proceed deliberately.")
    resolved.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    sentinel = resolved / ".write_check"
    sentinel.write_text(token, encoding="utf-8")
    back = sentinel.read_text(encoding="utf-8")
    sentinel.unlink()
    if back != token:
        raise SystemExit("[ABORT] sentinel write+read mismatch on RESULTS_DIR.")
    print(f"[guard a] sentinel write+read OK on {resolved}", flush=True)
    return resolved


# ----------------------------------------------------------------------------
# STEP 2(b) — fingerprints and consumed-manifests.
# ----------------------------------------------------------------------------

def file_sha256(path, chars: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:chars]


def inject_fingerprint(json_path: Path, *, corruption, cycle_count, eta, p,
                       checkpoint_hash, seed, batch_size,
                       extra: Optional[dict] = None) -> dict:
    """Embed the protocol fingerprint into a run JSON (atomic rewrite).
    Idempotent: an existing fingerprint is kept verbatim. `extra` carries
    variant fields (e.g. anchor_lambda, drive)."""
    data = pc.load_json(json_path)
    if "protocol_fingerprint" in data:
        return data["protocol_fingerprint"]
    fp = {
        "n_steps": len(pc.stream_rows(data)),
        "corruption": corruption,
        "cycle_count": cycle_count,
        "eta": eta,
        "p": p,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "batch_size": batch_size,
        "protocol": f"single-corruption severity-5 split cycled x{cycle_count}",
        **(extra or {}),
    }
    data["protocol_fingerprint"] = fp
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, json_path)
    return fp


def consumed_manifest(paths: list[Path]) -> list[dict]:
    """Manifest of the run files an analysis consumed (STEP 2b)."""
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            n = len(pc.stream_rows(pc.load_json(p)))
        except Exception:
            n = None
        out.append({"path": str(p), "sha256_16": file_sha256(p), "n_steps": n})
    return out


def drive_count_check(results_dir, expected_names: set[str]) -> dict:
    """STEP 2(c): Drive-side file count vs expected; flag any gap."""
    present = {p.name for p in Path(results_dir).glob("p9_*_pstar_cyc*.json")}
    missing = sorted(expected_names - present)
    extra = sorted(present - expected_names)
    report = {"expected": len(expected_names), "present_on_drive": len(present),
              "missing": missing, "unexpected_extra": extra}
    print(f"[guard c] drive-side count: expected={len(expected_names)} "
          f"present={len(present)} missing={len(missing)}", flush=True)
    if missing:
        print("[guard c] MISSING FILES (gap flagged):", flush=True)
        for m in missing:
            print(f"    {m}", flush=True)
    return report


# ----------------------------------------------------------------------------
# Cycle-aware stream analysis. group by local_step reset (cycles share the
# corruption name so the stock per-corruption grouping would merge them).
# ----------------------------------------------------------------------------

def group_rows_by_cycle(rows: list[dict]) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda r: (r.get("step")
                                          if r.get("step") is not None else 1 << 60))
    cycles, cur, prev_local = [], [], None
    for r in ordered:
        local = r.get("local_step")
        if (prev_local is not None and local is not None
                and local < prev_local and cur):
            cycles.append(cur)
            cur = []
        cur.append(r)
        prev_local = local if local is not None else prev_local
    if cur:
        cycles.append(cur)
    return cycles


def _cycle_nonfinite(cyc_rows) -> bool:
    return pc.block_has_nonfinite(cyc_rows)


def grad_norm_gbar_cycled(data) -> Optional[tuple[float, int, int]]:
    """||g_bar|| for a cycled stream: mean grad_l2 over the stationary window
    (local 50-150) WITHIN each cycle-block, averaged across cycles; cycles
    containing any NaN/inf are excluded — mirroring the continual semantics
    byte-for-byte (per-block window mean, diverged blocks dropped)."""
    rows = pc.stream_rows(data)
    if not rows:
        return None
    means, n_div = [], 0
    for cyc in group_rows_by_cycle(rows):
        if _cycle_nonfinite(cyc):
            n_div += 1
            continue
        stat = stationary_rows(cyc, STATIONARY_LO, STATIONARY_HI)
        vals = [row_metric(r, "grad_l2") for r in stat]
        vals = [v for v in vals if v is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    if not means:
        return None
    return (sum(means) / len(means), len(means), n_div)


def drift_stationary_cycled(data) -> Optional[float]:
    rows = pc.stream_rows(data)
    if not rows:
        return None
    means = []
    for cyc in group_rows_by_cycle(rows):
        if _cycle_nonfinite(cyc):
            continue
        stat = stationary_rows(cyc, STATIONARY_LO, STATIONARY_HI)
        vals = [row_metric(r, "drift_l2") for r in stat]
        vals = [v for v in vals if v is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    return (sum(means) / len(means)) if means else None


# ----------------------------------------------------------------------------
# Read-only cell analysis (row schema identical to stage1_common.analyze_cell_sc
# so analyze_stage1's fits/plot/E4/E5/verdict machinery is reused unchanged).
# Classification reuses pstar_common.classify_run UNCHANGED (hard primary via
# hard_only=True; soft recorded by the caller when needed).
# ----------------------------------------------------------------------------

def load_run_v2(results_dir, arch, corruption, severity, seed, p=None,
                source=False, eta=None, cycles: int = CYCLES):
    path = run_output_path_v2(results_dir, arch, corruption, severity, seed,
                              p=p, source=source, eta=eta, cycles=cycles)
    if not path.exists():
        return None
    try:
        return pc.load_json(path)
    except Exception:
        return None


def source_acc_v2(results_dir, arch, corruption, severity, seed, eta,
                  cycles: int = CYCLES) -> Optional[float]:
    data = load_run_v2(results_dir, arch, corruption, severity, seed,
                       source=True, eta=eta, cycles=cycles)
    return pc.source_mean_acc(data) if data is not None else None


def derive_pref_v2(results_dir, arch, corruption, severity, seed, eta,
                   source_acc, cycles: int = CYCLES):
    """Stable reference p_ref (eta-scaled ladder; stability = hard +
    below-source, the continual reference policy). Returns (p_ref, gbar)."""
    last = (None, None)
    for cand in pc.scaled_pref_ladder(arch, eta):
        data = load_run_v2(results_dir, arch, corruption, severity, seed,
                           p=cand, eta=eta, cycles=cycles)
        if data is None:
            continue
        g = grad_norm_gbar_cycled(data)
        gbar = g[0] if g else None
        last = (cand, gbar)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False)
        if not v["collapsed"]:
            return cand, gbar
    return last


def analyze_cell_v2(results_dir, arch, corruption, severity, eta, seed,
                    hard_only: bool = False, cycles: int = CYCLES) -> dict:
    source_acc = source_acc_v2(results_dir, arch, corruption, severity, seed,
                               eta, cycles=cycles)
    p_ref, gbar = derive_pref_v2(results_dir, arch, corruption, severity,
                                 seed, eta, source_acc, cycles=cycles)
    p_values = discover_p_values_v2(results_dir, arch, corruption, severity,
                                    seed, eta, cycles=cycles)
    points, per_p = [], {}
    consumed = []
    for p in p_values:
        path = run_output_path_v2(results_dir, arch, corruption, severity,
                                  seed, p=p, eta=eta, cycles=cycles)
        data = load_run_v2(results_dir, arch, corruption, severity, seed,
                           p=p, eta=eta, cycles=cycles)
        if data is None:
            points.append({"p": p, "collapsed": False, "valid": False})
            per_p[pc.format_p(p)] = {"valid": False}
            continue
        consumed.append(path)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False, hard_only=hard_only)
        v_other = pc.classify_run(data, source_acc, pref_drift=None,
                                  use_drift_criterion=False,
                                  hard_only=not hard_only)
        points.append({"p": p, "collapsed": v["collapsed"], "valid": True})
        rows = pc.stream_rows(data)
        per_p[pc.format_p(p)] = {
            "valid": True, "collapsed": v["collapsed"],
            "criterion": v["criterion"], "mean_acc": v["mean_acc"],
            "other_family_criterion": v_other["criterion"],
            "first_nan_step": pc.first_nonfinite_step(rows),
            "n_steps": len(rows),
        }
    sel = pc.choose_pstar(points)
    blo, bhi = sel["bracket_low"], sel["bracket_high"]
    if blo is not None and per_p.get(pc.format_p(blo), {}).get("collapsed"):
        crit = per_p[pc.format_p(blo)].get("criterion", "unknown")
    elif sel["p_star"] == 0.0:
        crit = "none(p*~0)"
    else:
        crit = sel.get("note") or "none"
    return {
        "arch": arch, "corruption": corruption, "severity": severity,
        "seed": seed, "eta": eta, "cycles": cycles,
        "source_acc": source_acc, "p_ref_used": p_ref,
        "grad_norm_gbar": gbar,
        "eta_times_gbar": (gbar * eta) if gbar is not None else None,
        "p_star": sel["p_star"], "p_star_bracket_low": blo,
        "p_star_bracket_high": bhi, "collapse_criterion": crit,
        "monotone_bracket": sel["monotone"], "bracket_note": sel["note"],
        "anomalous_zero": (sel["p_star"] == 0.0 and sel["monotone"] is False),
        "p_values_run": p_values, "per_p": per_p,
        "criterion_family": "hard_only" if hard_only else "soft+hard",
        "stable_mean_acc": (per_p.get(pc.format_p(bhi), {}).get("mean_acc")
                            if bhi is not None else None),
        "consumed_files": [str(c) for c in consumed],
    }


# ----------------------------------------------------------------------------
# STEP 0 — recover-or-declare scan (read-only).
# ----------------------------------------------------------------------------

GENUINE_E1 = re.compile(
    r"^p9_(\w+?)_pstar_(gaussian_noise|elastic_transform|impulse_noise|"
    r"contrast)_lr([-0-9.eE+]+)_p([0-9.]+)_seed(\d+)_sev(\d+)\.json$")
GENUINE_FWD = re.compile(
    r"^p9_(\w+?)_pstar_(fog|shot_noise)_lr([-0-9.eE+]+)_p([0-9.]+)"
    r"_seed(\d+)_sev(\d+)\.json$")


def recover_scan(results_dir, max_samples: int = 5) -> dict:
    """STEP 0: find genuine single-corruption/forward run JSONs; print
    n_steps + recorded args for samples; declare if none."""
    rd = Path(results_dir)
    names = sorted(p.name for p in rd.glob("*.json"))
    e1 = [n for n in names if GENUINE_E1.match(n)]
    fwd = [n for n in names if GENUINE_FWD.match(n)]
    print(f"[step 0] genuine single-corruption runs: {len(e1)} | "
          f"forward runs: {len(fwd)}", flush=True)
    for group, label in ((e1, "E1"), (fwd, "forward")):
        for n in group[:max_samples]:
            try:
                d = pc.load_json(rd / n)
            except Exception:
                continue
            a = d.get("args", {})
            rows = pc.stream_rows(d)
            print(f"  [{label}] {n}: n_steps={len(rows)} "
                  f"first_nan={pc.first_nonfinite_step(rows)} "
                  f"eta={a.get('heat_lr')} p={a.get('heat_restore_prob')} "
                  f"corruptions={a.get('corruptions')} "
                  f"ckpt={a.get('checkpoint')}", flush=True)
    if not e1 and not fwd:
        print("[step 0] DECLARED: no recoverable E1 runs; restarting from "
              "scratch (per instruction, no further recovery effort).",
              flush=True)
    return {"n_e1": len(e1), "n_forward": len(fwd),
            "e1_files": e1, "forward_files": fwd}


__all__ = [
    "CYCLES", "V2_ARCH", "V2_SEVERITY", "V2_ETAS",
    "CALIBRATION_CORRUPTIONS", "HELDOUT_CORRUPTIONS", "V2_CORRUPTIONS",
    "variant_tag_v2", "run_output_path_v2", "discover_p_values_v2",
    "build_run_command_v2",
    "require_drive_dir", "file_sha256", "inject_fingerprint",
    "consumed_manifest", "drive_count_check",
    "group_rows_by_cycle", "grad_norm_gbar_cycled", "drift_stationary_cycled",
    "load_run_v2", "source_acc_v2", "derive_pref_v2", "analyze_cell_v2",
    "recover_scan", "GENUINE_E1", "GENUINE_FWD",
]
