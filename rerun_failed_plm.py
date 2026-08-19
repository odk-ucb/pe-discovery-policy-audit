#!/usr/bin/env python3
"""Re-run TEV/TrpB4 policies that failed under missing PLM caches.

Reconstructs coords from:
  - free_llm / full_compositional / sonnet pools (name prefix match)
  - RAND:draw_XX via Random(20260818) same as run_pools.py
  - SEED:* via PS.make patterns
"""
from __future__ import annotations
import copy, json, os, sys, time, random as _r
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import campaign as C
import policy_space as PS
import ssmula_landscape as SSM

LAND = os.environ.get("LANDSCAPE", "TEV")
_CASE = {"GB1": "GB1", "TEV": "TEV", "TRPB4": "TrpB4"}
LAND = _CASE.get(LAND.upper(), LAND)
BUDGET = int(os.environ.get("BUDGET", 96))
N_INIT = int(os.environ.get("N_INIT", 24))
BATCH = int(os.environ.get("BATCH", 24))
N_SEEDS = int(os.environ.get("N_SEEDS", 8))
NPROC = int(os.environ.get("NPROC", 4))
LIMIT = int(os.environ.get("LIMIT", 0)) or None

SRC = HERE / f"pool_results_{LAND}_s24_b96i24x24.json"
# Prefer matching seed count of source file if present
OUT = HERE / f"pool_results_{LAND}_plmfix.json"
REPORT = HERE / f"plmfix_report_{LAND}.json"


def rebuild_rand():
    RANDP = []
    _rr = _r.Random(20260818)
    for i in range(20):
        c = copy.deepcopy(PS.DEFAULT)
        for comp, kv in PS.COORDINATES.items():
            for k, vals in kv.items():
                c[comp][k] = _rr.choice(vals)
        RANDP.append(PS.Policy(name=f"RAND:draw_{i:02d}", channel="random_basis", coords=c))
    return {p.name: p for p in RANDP}


def rebuild_seeds():
    pols = [
        PS.make("SEED:base_onehot_ridge", "seed"),
        PS.make("SEED:esm_rf_greedy", "seed", representation={"encoding": "esm2_frozen", "pooling": "site"},
                surrogate={"family": "random_forest"}),
        PS.make("SEED:gp_ucb", "seed", surrogate={"family": "gaussian_process", "uncertainty": "gp_posterior"},
                acquisition={"rule": "ucb", "beta": "medium"}),
        PS.make("SEED:rf_thompson", "seed", surrogate={"family": "random_forest", "count": "ensemble_of_k",
                "uncertainty": "ensemble_disagreement"}, acquisition={"rule": "thompson"}),
    ]
    return {p.name: p for p in pols}


def load_named_pools():
    """Map result-style names FREE:Name / FULL:Name to Policy coords."""
    out = {}
    mapping = [
        ("pools/free_llm.json", "FREE", "free"),
        ("pools/full_compositional.json", "FULL", "perturb"),
        ("pools/replication_sonnet5_free.json", "FREE", "free_replication"),
        ("pools/replication_sonnet5_perturb.json", "FULL", "perturb_replication"),
    ]
    # Also try REPLICATION prefixes if used
    for fn, prefix, ch in mapping:
        path = HERE / fn
        if not path.exists():
            continue
        blob = json.load(open(path))
        pols = blob.get("policies", [])
        for rec in pols:
            nm = rec.get("name") or ""
            coords = copy.deepcopy(PS.DEFAULT)
            for comp, kv in (rec.get("coords") or {}).items():
                if comp in coords and isinstance(kv, dict):
                    coords[comp].update({k: v for k, v in kv.items() if k in coords[comp]})
            # result keys truncate names; store full and short
            full = f"{prefix}:{nm}"
            p = PS.Policy(name=full, channel=ch, coords=coords)
            out[full] = p
            # truncated ~44 chars as in run_pools print (but keys may be full)
            out[full[:44]] = p
            out[full[:46]] = p
    return out



def rebuild_cmc_esm():
    """Single known CMC fail on TEV: representation.encoding=esm2_frozen off perturb implied base."""
    import collections as _cc
    # implied base from perturb channel
    blob = json.load(open(HERE / "pools/full_compositional.json"))
    flats = []
    for rec in blob.get("policies", []):
        c = copy.deepcopy(PS.DEFAULT)
        for comp, kv in (rec.get("coords") or {}).items():
            if comp in c and isinstance(kv, dict):
                c[comp].update({k: v for k, v in kv.items() if k in c[comp]})
        flats.append({f"{comp}.{k}": v for comp, kv in c.items() for k, v in kv.items()})
    if not flats:
        return {}
    implied = {k: _cc.Counter(f[k] for f in flats).most_common(1)[0][0] for k in flats[0]}
    c = copy.deepcopy(PS.DEFAULT)
    for comp, kv in PS.COORDINATES.items():
        for k in kv:
            dk = f"{comp}.{k}"
            if dk in implied:
                c[comp][k] = implied[dk]
    c["representation"]["encoding"] = "esm2_frozen"
    p = PS.Policy(name="CMC:representation.encoding=esm2_frozen", channel="coord_matched", coords=c)
    return {p.name: p}


def match_policy(name, catalogs):

    if name in catalogs:
        return catalogs[name]
    # prefix match
    for k, p in catalogs.items():
        if k.startswith(name) or name.startswith(k) or name.rstrip() in k or k.startswith(name.rstrip()):
            return p
    # strip channel
    if ":" in name:
        body = name.split(":", 1)[1]
        for k, p in catalogs.items():
            if body and body in k:
                return p
    return None


def _one(args):
    coords, seed, budget, n_init, batch, landscape, channel = args
    L = None if landscape == "GB1" else SSM.SSMuLALandscape(landscape)
    h = C.Harness(budget=budget, n_init=n_init, batch=batch, landscape=L, landscape_name=landscape)
    p = PS.Policy("tmp", channel or "rerun", coords)
    r = h.run(p, seed)
    if not r.ok:
        return None
    return float(r.best)


def eval_policy(coords, seeds, channel):
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        rs = list(ex.map(_one, [(coords, s, BUDGET, N_INIT, BATCH, LAND, channel) for s in seeds]))
    ok = [b for b in rs if b is not None]
    if not ok:
        return dict(ok=False, fail=len(rs), seeds=None, best=None)
    return dict(ok=True, fail=len(rs) - len(ok), seeds=ok, best=float(np.mean(ok)))


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    d = json.load(open(SRC))
    old_build = d.pop("__build__", None)
    failed = {k: v for k, v in d.items()
              if v.get("best") is None or (isinstance(v.get("best"), float) and v["best"] != v["best"])
              or int(v.get("fail") or 0) >= 20}
    print(f"LAND={LAND} failed={len(failed)}/{len(d)} build={old_build}", flush=True)

    catalogs = {}
    catalogs.update(rebuild_rand())
    catalogs.update(rebuild_seeds())
    catalogs.update(load_named_pools())
    catalogs.update(rebuild_cmc_esm())

    # Detect seed count from a successful row
    n_src_seeds = 24
    for v in d.values():
        if isinstance(v.get("seeds"), list) and v["seeds"]:
            n_src_seeds = len(v["seeds"])
            break
    seeds = list(range(min(N_SEEDS, n_src_seeds) if N_SEEDS else n_src_seeds))
    if os.environ.get("N_SEEDS"):
        seeds = list(range(N_SEEDS))
    print(f"using {len(seeds)} seeds", flush=True)

    report = dict(landscape=LAND, n_failed=len(failed), now_ok=[], still_fail=[], skipped=[], rerun=[])
    fixed = {}
    n_done = 0
    for name, row in failed.items():
        if LIMIT and n_done >= LIMIT:
            break
        p = match_policy(name, catalogs)
        if p is None:
            report["skipped"].append(name)
            print(f"  SKIP {name}", flush=True)
            continue
        print(f"  RERUN {name} ...", flush=True)
        t0 = time.time()
        r = eval_policy(p.coords, seeds, p.channel)
        n_done += 1
        entry = dict(name=name, channel=p.channel, ok=r["ok"], best=r.get("best"), fail=r["fail"],
                     sec=round(time.time() - t0, 1))
        report["rerun"].append(entry)
        if r["ok"]:
            report["now_ok"].append(name)
            fixed[name] = dict(
                best=r["best"], seeds=r["seeds"], seed_ids=seeds, fail=r["fail"],
                channel=p.channel, sec=entry["sec"], plmfix=True,
            )
            print(f"    OK {r['best']:.4f} ({entry['sec']}s)", flush=True)
        else:
            report["still_fail"].append(name)
            print(f"    FAIL", flush=True)

    # merge into full table
    merged = dict(d)
    for k, v in fixed.items():
        merged[k] = v
    merged["__build__"] = f"plmfix_{LAND}_{int(time.time())}"
    merged["__plmfix_from__"] = old_build
    # strip non-policy keys for analyze compatibility - keep __ keys
    json.dump(merged, open(OUT, "w"), indent=2)
    report.update(n_now_ok=len(report["now_ok"]), n_still_fail=len(report["still_fail"]),
                  n_skipped=len(report["skipped"]), out=str(OUT))
    json.dump(report, open(REPORT, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ["n_failed", "n_now_ok", "n_still_fail", "n_skipped", "out"]}, indent=2))


if __name__ == "__main__":
    main()
