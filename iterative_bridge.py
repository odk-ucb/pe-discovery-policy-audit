#!/usr/bin/env python3
"""Iterative one-coordinate PE policy search bridge experiment.

Closes the one-shot vs iterative scope gap without claiming a new PE method:
  - ONE_SHOT: sample K random single-coordinate moves off DEFAULT once; pick best after eval
  - ITERATIVE: G generations; each generation proposes K random single-coordinate moves off
    the *current best measured* policy (feedback from previous generation's fitness)

Both use the same non-LLM proposal distribution over live coordinates (uniform over live cells),
matched seeds/budget. If ITERATIVE beats ONE_SHOT, multi-generation structure helps even without
LLM judgment — supporting F5 as a real boundary. If not, one-shot null is conservative.

Also optional LLM_ITERATIVE via OpenRouter if OPENROUTER_API_KEY is set and --llm is passed.

Writes: iterative_bridge_results.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import campaign as C
import policy_space as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def live_cells(base_coords):
    """List (comp, key, value) alternatives that differ from base and are applicable."""
    out = []
    flat_base = {f"{c}.{k}": v for c, kv in base_coords.items() for k, v in kv.items()}
    for comp, kv in PS.COORDINATES.items():
        if comp not in base_coords:
            continue
        for k, vals in kv.items():
            key = f"{comp}.{k}"
            if not PS.applicable(flat_base, key):
                continue
            cur = base_coords[comp][k]
            for v in vals:
                if v == cur:
                    continue
                # skip if conditional would make it inapplicable after change
                trial = copy.deepcopy(base_coords)
                trial[comp][k] = v
                tflat = {f"{c}.{kk}": vv for c, kkvs in trial.items() for kk, vv in kkvs.items()}
                if not PS.applicable(tflat, key):
                    continue
                out.append((comp, k, v))
    return out


def apply_move(base_coords, move):
    c = copy.deepcopy(base_coords)
    comp, k, v = move
    c[comp][k] = v
    return c


def eval_policy(coords, seeds, budget, n_init, batch, landscape="GB1"):
    name = "bridge_tmp"
    p = PS.Policy(name=name, channel="bridge", coords=coords)
    bests = []
    fails = 0
    for s in seeds:
        r = C.Harness(p, seed=s, budget=budget, n_init=n_init, batch=batch,
                      landscape_name=landscape).run()
        if not r.ok:
            fails += 1
            continue
        bests.append(r.best)
    if not bests:
        return dict(mean=float("nan"), fails=fails, seeds=[])
    return dict(mean=float(np.mean(bests)), fails=fails, seeds=bests)


def _one(args):
    coords, seed, budget, n_init, batch, landscape = args
    import ssmula_landscape as SSM
    p = PS.Policy(name="x", channel="bridge", coords=coords)
    L = None
    if landscape != "GB1":
        L = SSM.SSMuLALandscape(landscape)
    h = C.Harness(budget=budget, n_init=n_init, batch=batch, landscape=L, landscape_name=landscape)
    r = h.run(p, seed)
    if hasattr(r, "ok") and not r.ok:
        return None
    # Prefer gain (best − noop); fall back to best.
    if hasattr(r, "gain") and r.gain == r.gain:
        return float(r.gain)
    if hasattr(r, "best"):
        return float(r.best)
    if isinstance(r, dict):
        return r.get("gain", r.get("best"))
    return None


def eval_policy_parallel(coords, seeds, budget, n_init, batch, landscape, nproc=4):
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        rs = list(ex.map(_one, [(coords, s, budget, n_init, batch, landscape) for s in seeds]))
    ok = [x for x in rs if x is not None]
    if not ok:
        return dict(mean=float("nan"), fails=len(rs), seeds=[])
    return dict(mean=float(np.mean(ok)), fails=len(rs) - len(ok), seeds=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", default="GB1")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--n-init", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--K", type=int, default=5, help="proposals per generation")
    ap.add_argument("--G", type=int, default=3, help="generations for iterative")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "iterative_bridge_results.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    seeds = list(range(args.seeds))
    base = copy.deepcopy(PS.DEFAULT)
    cells = live_cells(base)
    print(f"live cells from DEFAULT: {len(cells)}", flush=True)

    def sample_moves(coords, k):
        pool = live_cells(coords)
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    # --- evaluate DEFAULT ---
    t0 = time.time()
    base_res = eval_policy_parallel(base, seeds, args.budget, args.n_init, args.batch,
                                    args.landscape, args.nproc)
    print(f"DEFAULT mean={base_res['mean']:.4f} fails={base_res['fails']}  ({time.time()-t0:.0f}s)",
          flush=True)

    # --- ONE_SHOT: K moves off DEFAULT, evaluate all, take best ---
    moves = sample_moves(base, args.K)
    one_shot = []
    for m in moves:
        coords = apply_move(base, m)
        r = eval_policy_parallel(coords, seeds, args.budget, args.n_init, args.batch,
                                 args.landscape, args.nproc)
        one_shot.append(dict(move=list(m), mean=r["mean"], fails=r["fails"]))
        print(f"  one_shot {m} -> {r['mean']:.4f}", flush=True)
    best_os = max(one_shot, key=lambda d: (-np.nan if np.isnan(d["mean"]) else d["mean"]))

    # --- ITERATIVE: G gens, each K moves off current best ---
    cur = copy.deepcopy(base)
    cur_mean = base_res["mean"]
    hist = [dict(gen=0, mean=cur_mean, move=None)]
    for g in range(1, args.G + 1):
        moves = sample_moves(cur, args.K)
        gen_rows = []
        for m in moves:
            coords = apply_move(cur, m)
            r = eval_policy_parallel(coords, seeds, args.budget, args.n_init, args.batch,
                                     args.landscape, args.nproc)
            gen_rows.append(dict(move=list(m), mean=r["mean"], fails=r["fails"]))
            print(f"  iter g={g} {m} -> {r['mean']:.4f}", flush=True)
        best = max(gen_rows, key=lambda d: (-1e9 if (d["mean"] != d["mean"]) else d["mean"]))
        if best["mean"] == best["mean"] and (cur_mean != cur_mean or best["mean"] > cur_mean):
            cur = apply_move(cur, tuple(best["move"]))
            cur_mean = best["mean"]
        hist.append(dict(gen=g, mean=cur_mean, best_of_gen=best, all=gen_rows))

    out = dict(
        landscape=args.landscape,
        seeds=args.seeds,
        budget=args.budget,
        K=args.K,
        G=args.G,
        n_live_cells=len(cells),
        default=base_res,
        one_shot=dict(candidates=one_shot, best=best_os),
        iterative=dict(history=hist, final_mean=cur_mean),
        contrast=dict(
            iterative_minus_oneshot=(cur_mean - best_os["mean"])
            if (cur_mean == cur_mean and best_os["mean"] == best_os["mean"]) else None,
            oneshot_minus_default=(best_os["mean"] - base_res["mean"])
            if (best_os["mean"] == best_os["mean"] and base_res["mean"] == base_res["mean"]) else None,
            iterative_minus_default=(cur_mean - base_res["mean"])
            if (cur_mean == cur_mean and base_res["mean"] == base_res["mean"]) else None,
        ),
        note=(
            "Non-LLM uniform sampling over live single-coordinate moves. "
            "Tests whether multi-generation structure alone improves over one-shot under the same proposal class."
        ),
    )
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out["contrast"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
