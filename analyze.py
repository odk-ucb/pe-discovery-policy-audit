"""Analysis of record for the policy-generation experiment.

Deliberately NOT part of the build hash: this file reads results, it cannot change them.

Rewritten after a Phase-2 review found four defects in the previous version, each of which flattered the
headline:

1. THE MISSING CONTRAST. The pre-registered family compared `perturb` against `free`, `random_basis` and the
   matched arms, but never against the BASE POLICY the perturb channel perturbs -- which is behaviourally the
   harness default. Almost all of "perturb beats free" turns out to be "the default beats free". The family
   below now contains `perturb vs base` and `base vs free`, and the report leads with how many policies beat
   the default at all.
2. FALSE EXCLUSION REASON. The previous version dropped policies with incomplete seed sets, claiming seed
   indices were not recorded. They are: 107/109 rows carry `seed_ids`. Everything is now paired on the
   INTERSECTION of seed ids, so no policy is dropped for attrition.
3. ATTRITION HANDLED ONE WAY ONLY. `random_basis` loses far more cells to infeasibility than `free` does.
   Reporting only complete-case analysis silently treats infeasible policies as if they had never been
   proposed, which deletes the possibility that proposing FEASIBLE configurations is itself a contribution.
   All three handlings are now reported side by side.
4. NO EQUIVALENCE TEST. Two headline claims are nulls. A null is not equivalence without a margin, so TOST
   is reported against a declared margin, with the minimum detectable effect stated beside it.

A policy whose seed vector is bit-identical to the base is BEHAVIOURALLY INERT -- it is the base under
another name. Those are counted, and every contrast is additionally reported over live-only policies, because
an arm part-filled with copies of the base is biased toward the base.

On the pre-registration (FINDINGS iteration 77): it predicted value-over-coordinate. Note that [1] contains
two analyses that disagree with each other -- Fig 4(b)/8 puts Architectural change among the per-edit-helpful
categories, while Figs 5/6/9 show Hyperparameter tuning enriched among best-so-far updates. The
pre-registration read the second. Neither this file nor the paper should claim [1] speaks with one voice.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NAME = "SEED:base_onehot_ridge"

# Declared before this run's numbers were read: the largest difference in best-found fitness we are willing
# to call "no practical difference". 0.20 is ~3.5% of the base policy's mean. TOST at this margin is what
# converts a null into an equivalence claim; without a margin a null is only an absence of evidence.
EQUIV_MARGIN = 0.20

FAMILY = [
    ("perturb vs base            (does the LLM's move add to the default?)", "perturb", "base"),
    ("base vs free               (does the DEFAULT beat free proposal?)", "base", "free"),
    ("perturb vs free            (is the channel better at all?)", "perturb", "free"),
    ("perturb vs coord_matched   (did the LLM pick the COORDINATE?)", "perturb", "coord_matched"),
    ("perturb vs value_matched   (did the LLM pick the VALUE?)", "perturb", "value_matched"),
    ("free vs random_basis       (does LLM generation beat uniform?)", "free", "random_basis"),
    ("perturb vs random_basis    (does the channel beat uniform?)", "perturb", "random_basis"),
    ("project vs free            (do cross-domain donors help?)", "project", "free"),
    ("recombine vs free          (does combination help?)", "recombine", "free"),
    ("free_replication vs base   (does the SECOND generator's move beat the default?)", "free_replication", "base"),
    ("free_replication vs random_basis (does the SECOND generator beat uniform?)", "free_replication", "random_basis"),
    ("free vs free_replication    (do the two generators differ from each other?)", "free", "free_replication"),
    ("perturb_replication vs base (SECOND generator's perturb move vs default)", "perturb_replication", "base"),
    ("perturb vs perturb_replication (do the two generators' perturb channels differ?)", "perturb", "perturb_replication"),
]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load(path):
    d = json.load(open(path))
    build = d.pop("__build__", None)
    rows, failed = {}, {}
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        if v.get("seeds"):
            ids = v.get("seed_ids") or list(range(len(v["seeds"])))
            rows[k] = dict(channel=v["channel"], vals=dict(zip(ids, map(float, v["seeds"]))),
                           fail=int(v.get("fail", 0) or 0))
        else:
            failed[k] = dict(channel=v.get("channel", "?"), fail=int(v.get("fail", 0) or 0))
    return build, rows, failed


def paired(a, b):
    """Pair on the INTERSECTION of seed ids. No policy is dropped for attrition."""
    common = sorted(set(a) & set(b))
    return np.array([a[s] for s in common]), np.array([b[s] for s in common]), common


def is_inert(vals, base_vals):
    common = sorted(set(vals) & set(base_vals))
    return bool(common) and all(vals[s] == base_vals[s] for s in common)


def arm_series(rows, ch, base_vals, live_only=False):
    members = {k: v["vals"] for k, v in rows.items() if v["channel"] == ch}
    if live_only:
        members = {k: v for k, v in members.items() if not is_inert(v, base_vals)}
    if not members:
        return {}, 0
    common = set.intersection(*[set(v) for v in members.values()])
    return {s: float(np.mean([v[s] for v in members.values()])) for s in sorted(common)}, len(members)


def tost(a, b, margin):
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0, 0.0
    p = max(1 - stats.t.cdf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))
    return float(p), float(stats.t.ppf(0.975, n - 1) * se)


def main(path):
    build, rows, failed = load(path)
    if not rows:
        print(f"{path}: no completed policies")
        return
    base_vals = rows[BASE_NAME]["vals"]

    print("=" * 104)
    print(f"ANALYSIS OF RECORD   build {build}   {len(rows)} executed policies   paired on seed_ids")
    print("=" * 104)

    # ---- 0. does anything beat the hand-written default?
    print(f"\nDOES ANY POLICY BEAT THE DEFAULT ({BASE_NAME})?")
    pv, names = [], []
    for k, v in rows.items():
        if k == BASE_NAME:
            continue
        a, b, common = paired(v["vals"], base_vals)
        if len(common) < 3:
            continue
        pv.append(float(stats.ttest_rel(a, b).pvalue))
        names.append((k, float((a - b).mean())))
    adj = holm(np.array(pv))
    better = [(n, dd) for (n, dd), q in zip(names, adj) if q < 0.05 and dd > 0]
    worse = [(n, dd) for (n, dd), q in zip(names, adj) if q < 0.05 and dd < 0]
    print(f"   tested against the default: {len(names)}")
    print(f"   significantly BETTER after Holm: {len(better)}/{len(names)}")
    print(f"   significantly WORSE  after Holm: {len(worse)}/{len(names)}")

    # ---- 1. behaviourally inert policies
    inert, tot = {}, {}
    for k, v in rows.items():
        tot[v["channel"]] = tot.get(v["channel"], 0) + 1
        if k != BASE_NAME and is_inert(v["vals"], base_vals):
            inert.setdefault(v["channel"], []).append(k)
    n_inert = sum(len(v) for v in inert.values())
    print(f"\nBEHAVIOURALLY INERT POLICIES (seed vector bit-identical to base): {n_inert}/{len(rows)}")
    for ch in sorted(inert):
        print(f"   {ch:16s} {len(inert[ch]):2d}/{tot[ch]:2d} are the base under another name")

    # ---- 2. attrition
    print("\nATTRITION (cells lost to infeasibility), by arm:")
    for ch in sorted(tot):
        cells = sum(len(v["vals"]) for v in rows.values() if v["channel"] == ch)
        lost = sum(v["fail"] for v in rows.values() if v["channel"] == ch)
        deadp = [k for k, v in failed.items() if v["channel"] == ch]
        lost += sum(failed[k]["fail"] for k in deadp)
        denom = cells + lost
        print(f"   {ch:16s} {lost:3d}/{denom:4d} cells lost ({100*lost/max(denom,1):5.1f}%)"
              f"   whole policies lost: {len(deadp)}")

    # ---- 3. channel means
    print(f"\n{'channel':16s} {'n':>3s} {'live':>5s} {'mean best':>10s} {'sd':>7s}")
    series = {}
    for ch in sorted(tot):
        s, n = arm_series(rows, ch, base_vals)
        sl, nl = arm_series(rows, ch, base_vals, live_only=True)
        series[ch] = (s, sl)
        ms = np.array([float(np.mean(list(rows[k]["vals"].values())))
                       for k in rows if rows[k]["channel"] == ch])
        print(f"{ch:16s} {n:3d} {nl:5d} {ms.mean():10.3f} {ms.std(ddof=1) if len(ms) > 1 else 0:7.3f}")
    series["base"] = (dict(base_vals), dict(base_vals))

    # ---- 4. the family, all policies and live-only
    for tag, idx in (("ALL POLICIES", 0), ("LIVE POLICIES ONLY", 1)):
        print("\n" + "-" * 104)
        print(f"PRE-REGISTERED FAMILY — {tag}  (unit = seed; paired on seed ids; Holm across the family)")
        print("-" * 104)
        res = []
        for label, x, y in FAMILY:
            sx, sy = series.get(x, ({}, {}))[idx], series.get(y, ({}, {}))[idx]
            if not sx or not sy:
                res.append((label, None))
                continue
            a, b, common = paired(sx, sy)
            if len(common) < 3:
                res.append((label, None))
                continue
            t = stats.ttest_rel(a, b)
            pe, mde = tost(a, b, EQUIV_MARGIN)
            res.append((label, dict(d=float((a - b).mean()), p=float(t.pvalue), n=len(common),
                                    wins=int((a > b).sum()), tost=pe, mde=mde)))
        ok = [r for _, r in res if r]
        adjf = holm([r["p"] for r in ok]) if ok else []
        it = iter(adjf)
        for label, r in res:
            if r is None:
                print(f"   {label:62s}  arm absent")
                continue
            q = next(it)
            star = "*" if q < 0.05 else " "
            eq = "EQUIV" if r["tost"] < 0.05 else "     "
            print(f"   {label:62s} {r['d']:+7.3f} p={r['p']:.4f} holm={q:.4f}{star} "
                  f"{r['wins']:2d}/{r['n']:2d} TOST={r['tost']:.3f} {eq} MDE={r['mde']:.3f}")
        print(f"   * survives Holm.  EQUIV = equivalent within +/-{EQUIV_MARGIN} (TOST p<0.05), "
              f"not merely non-significant.")

    # ---- 5. attrition sensitivity on the headline null
    print("\n" + "-" * 104)
    print("ATTRITION SENSITIVITY — free vs random_basis under three handlings")
    print("-" * 104)
    fs, _ = arm_series(rows, "free", base_vals)
    rs, _ = arm_series(rows, "random_basis", base_vals)
    a, b, common = paired(fs, rs)
    print(f"   complete-case                                 {(a-b).mean():+7.3f}  "
          f"p={stats.ttest_rel(a,b).pvalue:.4f}  {int((a>b).sum())}/{len(common)}")
    anchor = min(min(v["vals"].values()) for v in rows.values())
    itt = {}
    for ch in ("free", "random_basis"):
        mem = {k: v["vals"] for k, v in rows.items() if v["channel"] == ch}
        for k, v in failed.items():
            if v["channel"] == ch:
                mem[k] = {}
        allseeds = sorted(set().union(*[set(v) for v in mem.values() if v]))
        itt[ch] = {s: float(np.mean([v.get(s, anchor) for v in mem.values()])) for s in allseeds}
    a2, b2, c2 = paired(itt["free"], itt["random_basis"])
    print(f"   intention-to-treat (infeasible = anchor)      {(a2-b2).mean():+7.3f}  "
          f"p={stats.ttest_rel(a2,b2).pvalue:.4f}  {int((a2>b2).sum())}/{len(c2)}   anchor={anchor:.3f}")
    print("   -> if these disagree, FEASIBILITY of the proposal is part of what the generator supplies,")
    print("      and complete-case analysis deletes exactly that contribution.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pool_results_s24.json"))
