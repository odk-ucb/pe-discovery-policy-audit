#!/usr/bin/env python3
"""Powered paired evaluation: DEFAULT vs a fixed LLM one-shot policy (24 seeds)."""
from __future__ import annotations
import copy, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import campaign as C
import policy_space as PS

N_SEEDS = int(os.environ.get("N_SEEDS", "24"))
NPROC = int(os.environ.get("NPROC", "6"))
BUDGET, N_INIT, BATCH = 96, 24, 24
OUT = HERE / os.environ.get("OUT", "powered_llama_oneshot_vs_default.json")


def apply_partial(partial):
    coords = copy.deepcopy(PS.DEFAULT)
    for key, val in partial.items():
        if "." not in key:
            continue
        comp, k = key.split(".", 1)
        if comp in coords and k in coords[comp]:
            allowed = PS.COORDINATES.get(comp, {}).get(k)
            if allowed is None or val in allowed:
                coords[comp][k] = val
    return coords


def _one(args):
    coords, seed, name = args
    h = C.Harness(budget=BUDGET, n_init=N_INIT, batch=BATCH, landscape_name="GB1")
    r = h.run(PS.Policy(name, "powered", coords), seed)
    if not r.ok:
        return None
    return float(r.gain) if hasattr(r, "gain") and r.gain == r.gain else float(r.best)


def eval_all(coords, seeds, name):
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        return list(ex.map(_one, [(coords, s, name) for s in seeds]))


def main():
    # best llama oneshot partial from bridge
    partial = {
        "representation.encoding": "esm2_frozen",
        "surrogate.family": "gaussian_process",
        "generator.scope": "local_hamming_ball",
        "acquisition.rule": "ucb",
        "batch.selection": "diverse_top_k_by_distance",
    }
    if len(sys.argv) > 1:
        partial = json.loads(sys.argv[1])
    seeds = list(range(N_SEEDS))
    base_c = copy.deepcopy(PS.DEFAULT)
    llm_c = apply_partial(partial)
    print("eval DEFAULT...", flush=True)
    t0 = time.time()
    b = eval_all(base_c, seeds, "default")
    print("eval LLM...", flush=True)
    L = eval_all(llm_c, seeds, "llama_oneshot")
    # paired complete
    pairs = [(x, y) for x, y in zip(b, L) if x is not None and y is not None]
    bx = np.array([p[0] for p in pairs])
    lx = np.array([p[1] for p in pairs])
    d = lx - bx
    tstat, p = stats.ttest_rel(lx, bx) if len(d) > 1 else (np.nan, np.nan)
    # Wilcoxon
    try:
        wstat, wp = stats.wilcoxon(d) if len(d) > 0 and np.any(d != 0) else (np.nan, np.nan)
    except Exception:
        wstat, wp = np.nan, np.nan
    out = dict(
        n_seeds=N_SEEDS,
        n_paired=len(pairs),
        partial=partial,
        default_mean=float(np.nanmean([x for x in b if x is not None])),
        llm_mean=float(np.nanmean([x for x in L if x is not None])),
        paired_delta_mean=float(d.mean()) if len(d) else None,
        paired_delta_std=float(d.std(ddof=1)) if len(d) > 1 else None,
        ttest_rel_p=float(p) if p == p else None,
        wilcoxon_p=float(wp) if wp == wp else None,
        n_llm_better=int((d > 0).sum()) if len(d) else 0,
        default_seeds=b,
        llm_seeds=L,
        seconds=round(time.time() - t0, 1),
        note="Powered re-eval of llama-3.3-70b best one-shot partial vs DEFAULT on GB1",
    )
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({k: out[k] for k in out if k not in ("default_seeds", "llm_seeds", "partial")}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
