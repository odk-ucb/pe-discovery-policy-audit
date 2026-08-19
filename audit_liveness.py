"""Behavioural liveness: does setting a coordinate to an alternative value change what the campaign DOES?

Not part of the build hash -- it reads the compiler, it cannot change it.

The static check in run_pools.py greps for the coordinate's bare name. That is not liveness. It cannot
distinguish `stopping.rule` from `acquisition.rule` (21 unique bare names for 22 coordinates), and it cannot
see a value that is read and then discarded -- `batch.size_rule` is read and both branches assign
`nb = self.batch`. The previous "inertness audit" was a byte-identical duplicate of that same grep, so it was
provably always empty and audited nothing.

This does the real test: for every (coordinate, alternative value) cell, run the campaign from the same base
with only that cell changed, on the same seeds, and compare the FULL trace. Identical trace on every seed
means the cell cannot change any proposal -- it is INERT, a phantom degree of freedom that inflates the
apparent size of the design space without enlarging it.

A coordinate is LIVE if at least one of its alternative values is live.
"""
import copy
import json
import os
import sys

import numpy as np

import campaign as C
import policy_space as PS
import run_pools as R

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1]           # a cell that changes nothing on two independent seeds is inert for our purposes
BUDGET, N_INIT, BATCH = R.BUDGET, R.N_INIT, R.BATCH


def base_coords():
    """The perturb channel's own implied base, so the audit matches the base the controls perturb."""
    import collections
    d = json.load(open(os.path.join(HERE, "pools", "full_compositional.json")))
    per = [r for r in d["policies"] if r.get("channel") == "perturb"]
    fl = []
    for rec in per:
        c = copy.deepcopy(PS.DEFAULT)
        for comp, kv in (rec.get("coords") or {}).items():
            for k, v in kv.items():
                if comp in c and k in c[comp]:
                    c[comp][k] = v
        fl.append({f"{a}.{b}": v for a, cc in c.items() for b, v in cc.items()})
    if not fl:
        return copy.deepcopy(PS.DEFAULT)
    implied = {k: collections.Counter(x[k] for x in fl).most_common(1)[0][0] for k in fl[0]}
    c = copy.deepcopy(PS.DEFAULT)
    for comp, kv in PS.COORDINATES.items():
        for k in kv:
            dk = f"{comp}.{k}"
            if dk in implied:
                c[comp][k] = implied[dk]
    return c


def trace_of(H, coords, seed):
    r = H.run(PS.Policy(name="audit", channel="audit", coords=coords), seed)
    if not r.ok:
        return None
    return np.asarray(r.trace, float)


def main():
    H = C.Harness(budget=BUDGET, n_init=N_INIT, batch=BATCH, universe="all")
    base = base_coords()
    flat = {f"{a}.{b}": v for a, cc in base.items() for b, v in cc.items()}
    ref = {s: trace_of(H, base, s) for s in SEEDS}

    print("=" * 104)
    print("BEHAVIOURAL LIVENESS AUDIT — does changing this cell change any proposal?")
    print(f"base = perturb channel's implied base | seeds {SEEDS} | budget {BUDGET}")
    print("=" * 104)

    rows, live_coord, cells = [], {}, 0
    for comp, kv in PS.COORDINATES.items():
        for k, vals in kv.items():
            dk = f"{comp}.{k}"
            applicable = PS.applicable(flat, dk)
            for v in vals:
                if v == flat[dk]:
                    continue
                cells += 1
                c = copy.deepcopy(base)
                c[comp][k] = v
                verdict, detail = "INERT", ""
                if not applicable:
                    verdict, detail = "INAPPLICABLE", "excluded by a declared conditional"
                else:
                    for s in SEEDS:
                        t = trace_of(H, c, s)
                        if t is None:
                            verdict, detail = "FAILS", "policy does not execute"
                            break
                        if ref[s] is None or t.shape != ref[s].shape or not np.array_equal(t, ref[s]):
                            verdict, detail = "LIVE", f"trace differs on seed {s}"
                            break
                rows.append((dk, v, verdict, detail))
                if verdict == "LIVE":
                    live_coord[dk] = True
                live_coord.setdefault(dk, live_coord.get(dk, False))
                print(f"   {dk:34s} = {str(v)[:26]:28s} {verdict:12s} {detail}", flush=True)

    n_live = sum(1 for x in live_coord.values() if x)
    n_tot = len(live_coord)
    by = {}
    for _, _, verdict, _ in rows:
        by[verdict] = by.get(verdict, 0) + 1
    print("\n" + "=" * 104)
    print(f"CELLS: {cells} tested -> " + ", ".join(f"{v} {kk}" for kk, v in sorted(by.items())))
    print(f"COORDINATES: {n_live}/{n_tot} LIVE (at least one value changes behaviour); "
          f"{n_tot - n_live} DEAD")
    dead = sorted(k for k, v in live_coord.items() if not v)
    if dead:
        print("DEAD coordinates (no value changes any proposal — phantom degrees of freedom):")
        for d in dead:
            print(f"   {d}")
    json.dump([dict(coordinate=a, value=str(b), verdict=c, detail=d) for a, b, c, d in rows],
              open(os.path.join(HERE, "liveness_audit.json"), "w"), indent=1)
    print("\nsaved -> liveness_audit.json")


if __name__ == "__main__":
    main()
