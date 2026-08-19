"""Ground, execute and score the generated policy pools — the decisive FREE_LLM vs FULL_COMPOSITIONAL comparison.

THE QUESTION (governing review §7.1): on one objective landscape, does the full concept-decompose-perturb-project-
compose generator produce a better executable discovery policy than the same high-reasoning model generating freely,
under the same candidate and rollout budget?

WHAT IS HELD IDENTICAL between the two conditions, so the comparison is about the generation representation and nothing
else: the same frozen campaign state, the same executable action space, the same candidate budget (20 policies), the
same landscape, the same initial observations per seed, the same assay budget per policy, the same seeds, and the same
scoring code. What differs is *only* whether the generator was given the source library and the
component-coordinate decomposition.

GROUNDING IS AUDITED, NOT TRUSTED. Every proposed coordinate assignment is validated against the declared basis. A value
outside it is `INVALID` **with the offending coordinate named**, never mapped to the nearest legal value — a previous
experiment in this project measured that a substituting grounder reported a clean 20/20 while silently deleting an
operator family that was second-best on the real task. `missing_coordinate` fields are reported verbatim: a named gap in
the action space is an output of this experiment, not a nuisance.

THREE COMPARATORS, because two of them are insufficient on their own:
    NULL     uniform random selection at equal budget — the matched-resource control.
    NO-OP    the best fitness already present in the initial observations. A policy that does not beat this has
             discovered nothing, and a previous experiment issued a false GO at margin 0.996 for want of this anchor.
    SEEDS    the four literature-shaped policies; `FIXED_LIBRARY_ORACLE` is the best of them chosen in hindsight, which
             is a ceiling and not a baseline.

E-CLASSES USE COMPLETE LINKAGE. "Within tolerance" is not transitive; single linkage previously chained the best and
worst operator in a menu into one class.
"""
import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker; seeds are parallel at process level
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import rankdata

import campaign as C
import ssmula_landscape as SSM
import policy_space as PS

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- Single-writer discipline and atomic result writes.
# Four instances of this script were once found running at once, all truncating the same result JSON with
# open(path,"w"). Concurrent truncating writers interleave partial states, so a reader can observe a file
# that never corresponded to any single run. Two fixes: an exclusive lock so a second instance refuses to
# start, and atomic replace so a crash mid-write cannot leave a truncated file.
def assert_every_coordinate_is_read():
    """Report which declared coordinate NAMES appear in the compiler, and flag names that collide.

    This check previously claimed to prove that all 22 coordinates are read. It cannot. It greps for the BARE
    coordinate name, and bare names collide: `stopping.rule` and `acquisition.rule` share the name `rule`, so
    `stopping.rule` passed on the strength of a match belonging to a different coordinate. 21 unique bare names
    cover 22 coordinates. A reviewer also found `batch.size_rule` read and then discarded (`nb = self.batch` in
    both branches), which no static check can catch.

    So this function no longer asserts liveness -- it asserts only that a name is present, names its own
    blind spots, and aborts on a name that is entirely absent. REAL liveness is behavioural and is measured by
    audit_liveness.py, which varies each coordinate and checks whether any proposal changes."""
    src = open(os.path.join(HERE, "campaign.py")).read()
    bare = {}
    for comp, kv in PS.COORDINATES.items():
        for k in kv:
            bare.setdefault(k, []).append(f"{comp}.{k}")
    absent = [d for k, ds in bare.items() for d in ds
              if f'"{k}"' not in src and f"'{k}'" not in src]
    collide = {k: ds for k, ds in bare.items() if len(ds) > 1}
    if absent:
        raise SystemExit("run_pools.py: coordinate names never present in campaign.py: " + ", ".join(absent))
    if collide:
        print(f"   liveness check: NAME COLLISIONS, presence is not proof of use for {collide} "
              f"-- see audit_liveness.py for the behavioural test")
    return len(bare), sum(len(v) for v in bare.values())


def _build_hash():
    """Hash of every file that can change a number, stored inside each results file.

    A reviewer found campaign.py edited mid-run, changing 23/40 policies' scored pool, with no way to tell
    from any results file which build produced which number. Results carrying no build identity cannot be
    compared across time, so the hash travels with the data."""
    import hashlib
    h = hashlib.sha256()
    for f in ("campaign.py", "policy_space.py", "landscape.py", "run_pools.py"):
        try:
            h.update(open(os.path.join(HERE, f), "rb").read())
        except OSError:
            h.update(b"MISSING")
    return h.hexdigest()[:16]


# Captured once, at import. Computing it lazily at write time would re-read the sources and silently record a
# DIFFERENT hash if any file were edited mid-run — which is precisely the failure the hash exists to detect.
_BUILD = _build_hash()


def _atomic_dump(obj, path):
    import tempfile
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)                     # atomic on POSIX: readers see old or new, never partial
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _claim_writer_lock():
    import fcntl, atexit
    fh = open(os.path.join(HERE, ".run_pools.lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("run_pools.py: another instance holds the result lock; refusing to start.\n"
                         "   Concurrent writers corrupt pool_results_*.json. Wait for it, or kill it first.")
    fh.write(f"{os.getpid()}\n"); fh.flush()
    atexit.register(lambda: (fcntl.flock(fh, fcntl.LOCK_UN), fh.close()))
    return fh
POOLS = os.path.join(HERE, "pools")
# TWO-STAGE DESIGN, declared here before any pool result was read. Stage 1 screens all 44 policies at 8 seeds to
# identify finalists and the per-channel pattern; stage 2 re-runs the finalists and every comparator at the full 24
# seeds, and only stage-2 numbers are reported as the comparison of record. Screening then confirming is standard and
# costs nothing in validity provided the confirmation set is chosen by stage-1 RANK and the stage-2 numbers are the
# ones quoted — both of which hold here. The reason is purely computational: the expensive policies request
# exhaustive pools under 1,920- and 2,480-dimensional encodings.
# Operating point. Overridable from the environment so the budget regime can be varied without editing a
# build-hashed source mid-experiment. The default 96/24/24 is FAR below the protein-subfield norm: ALDE 480,
# CLADE 480, ftMLDE 480, SSMuLA 480 (range 120-2,016), BO-EVO 1,536. BO-EVO's own sweep on this same GB1
# four-site landscape reports that batch sizes below 384 do not reliably reach the global optimum within four
# rounds, and its smallest swept batch is 24 -- exactly ours. So any null measured at 96/24/24 may be a
# property of the budget regime rather than of the generator, and must be re-measured at 480 before it is
# claimed. BUDGET=480, N_INIT=96, BATCH=96 reproduces ALDE's and CLADE's geometry exactly (96 + 4x96).
# Landscape selector. Default GB1 preserves every prior number exactly. TEV and TrpB4 are the two additional
# four-site SSMuLA landscapes acquired to test whether the coordinate-choice equivalence and "nothing beats
# the default" findings are properties of GB1 specifically or of the method generally. Coordinates requiring
# a per-landscape PLM cache (representation.encoding=esm2_frozen, generator.zero_shot_prior) are not yet
# built for TEV/TrpB4 and raise a named RuntimeError on those landscapes -- policies asserting them are
# recorded as infeasible FOR THAT LANDSCAPE, not silently scored against GB1's cache.
_LANDSCAPE_RAW = os.environ.get("LANDSCAPE", "GB1")
# Case-normalised: SSMuLA file names are mixed-case (TrpB4.csv), but env vars are naturally typed in any
# case. A bare .upper() looked for TRPB4.csv and failed with FileNotFoundError -- caught before any TrpB4
# campaign was scored on the wrong assumption.
_CASE = {"GB1": "GB1", "TEV": "TEV", "TRPB4": "TrpB4"}
LANDSCAPE = _CASE.get(_LANDSCAPE_RAW.upper(), _LANDSCAPE_RAW)


def _make_landscape():
    if LANDSCAPE == "GB1":
        return None, "GB1"          # None -> Harness builds its own Landscape(); identical to prior behaviour
    return SSM.SSMuLALandscape(LANDSCAPE), LANDSCAPE


BUDGET = int(os.environ.get("BUDGET", 96))
N_INIT = int(os.environ.get("N_INIT", 24))
BATCH = int(os.environ.get("BATCH", 24))
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
# Bounded batches with resume: background work does not survive session boundaries here, so each invocation executes
# at most LIMIT new policies and exits. Every (policy, seed) is deterministic, so this changes no number.
LIMIT = int(os.environ.get("LIMIT", "0")) or None
SEEDS = list(range(N_SEEDS))


def ground(rec, idx):
    """Validate a proposed coordinate assignment. Returns (Policy | None, status, detail)."""
    coords = rec.get("coords")
    if not isinstance(coords, dict):
        return None, "INVALID", "no coords object"
    import copy
    out = copy.deepcopy(PS.DEFAULT)
    problems = []
    for comp, kv in coords.items():
        if comp not in PS.COORDINATES:
            problems.append(f"unknown component '{comp}'"); continue
        if not isinstance(kv, dict): problems.append(f"component '{comp}' is not an object"); continue
        for k, v in kv.items():
            if k not in PS.COORDINATES[comp]:
                problems.append(f"unknown coordinate '{comp}.{k}'"); continue
            if v not in PS.COORDINATES[comp][k]:
                problems.append(f"'{comp}.{k}' = '{v}' outside the declared basis {PS.COORDINATES[comp][k]}")
                continue
            out[comp][k] = v
    missing = [f"{c}.{k}" for c in PS.COORDINATES for k in PS.COORDINATES[c]
               if k not in coords.get(c, {})]
    if problems:
        return None, "INVALID", "; ".join(problems[:3])
    name = rec.get("name", f"p{idx}")[:40]
    ch = rec.get("channel", "free")
    p = PS.Policy(name=name, channel=ch, coords=out,
                  provenance=dict(rationale=rec.get("rationale", "")[:300],
                                  derivation=rec.get("derivation", "")[:300],
                                  missing_coordinate=rec.get("missing_coordinate", "")))
    return p, ("OK" if not missing else "OK_DEFAULTED"), (f"defaulted {len(missing)}" if missing else "")


def load(fn, condition):
    f = os.path.join(POOLS, fn)
    if not os.path.exists(f): return None
    d = json.load(open(f))
    recs = d.get("policies", [])
    pols, audit = [], []
    for i, r in enumerate(recs):
        p, st, detail = ground(r, i)
        audit.append(dict(name=r.get("name", f"p{i}"), channel=r.get("channel", "free"), status=st, detail=detail,
                          missing_coordinate=r.get("missing_coordinate", "")))
        if p is not None:
            p.name = f"{condition[:4]}:{p.name}"
            pols.append(p)
    return dict(condition=condition, policies=pols, audit=audit, n_proposed=len(recs))


_H = None


def _one(args):
    """Run one (policy, seed) in a worker. The harness is built once per worker and reused."""
    global _H
    pol, seed, budget, n_init, batch = args
    if _H is None:
        _lsc, _lname = _make_landscape()
        _H = C.Harness(budget=budget, n_init=n_init, batch=batch, universe="all",
                       landscape=_lsc, landscape_name=_lname)
    return _H.run(pol, seed)


def eclasses(names, vals, tol):
    """Complete linkage on the mean outcome: every member within `tol` of the class minimum."""
    order = sorted(range(len(names)), key=lambda i: vals[i])
    cls, cur = [], [order[0]]
    for i in order[1:]:
        if vals[i] - vals[cur[0]] <= tol: cur.append(i)
        else: cls.append(cur); cur = [i]
    cls.append(cur)
    return [[names[i] for i in c] for c in cls]


def main():
    _lsc, _lname = _make_landscape()
    H = C.Harness(budget=BUDGET, n_init=N_INIT, batch=BATCH, universe="all",
                  landscape=_lsc, landscape_name=_lname)
    noop = float(np.mean([max(H.initial(s).values()) for s in SEEDS]))
    print("=" * 118)
    print(f"landscape {LANDSCAPE}   operating point declared before generation: budget {BUDGET}, "
          f"init {N_INIT}, batch {BATCH}, {N_SEEDS} seeds, universe {len(H.universe):,}")
    print(f"true optimum {H.true_best_f:.4f}   no-op anchor {noop:.4f}")
    print("=" * 118)

    # REPLICATION_SONNET5: a second, independently-versioned generator (claude-sonnet-5), generated cold
    # under the same protocol as the original pools, to test whether the central finding is a property of one
    # unversioned generator or generalises across generators. Loaded like any other pool; its policies carry
    # distinct channels (free_replication / perturb_replication) so they are reported separately, not merged
    # into the original free/perturb channels which would silently conflate two different generators.
    conds = [c for c in (load("free_llm.json", "FREE_LLM"),
                         load("full_compositional.json", "FULL_COMPOSITIONAL"),
                         load("replication_sonnet5_free.json", "REPLICATION_SONNET5_FREE"),
                         load("replication_sonnet5_perturb.json", "REPLICATION_SONNET5_PERTURB")) if c]
    if not conds:
        print("no pools found in pools/ — the generators have not written yet"); return

    SEEDPOL = [
        PS.make("SEED:base_onehot_ridge", "seed"),
        PS.make("SEED:esm_rf_greedy", "seed", representation={"encoding": "esm2_frozen", "pooling": "site"},
                surrogate={"family": "random_forest"}),
        PS.make("SEED:gp_ucb", "seed", surrogate={"family": "gaussian_process", "uncertainty": "gp_posterior"},
                acquisition={"rule": "ucb", "beta": "medium"}),
        PS.make("SEED:rf_thompson", "seed", surrogate={"family": "random_forest", "count": "ensemble_of_k",
                "uncertainty": "ensemble_disagreement"}, acquisition={"rule": "thompson"}),
    ]
    NULL = PS.Policy(name="NULL:random", channel="null", coords=PS.DEFAULT)

    # ---- TWO NON-LLM CONTROLS, both added on an auditor's finding that neither LLM condition beat uniform sampling
    # of the same design space. Without these the experiment cannot attribute anything to the LLM at all.
    import copy, random as _r
    RANDP = []
    _rr = _r.Random(20260818)
    for i in range(20):                     # RANDOM_POLICY: uniform draws from the declared basis, no LLM, no papers
        c = copy.deepcopy(PS.DEFAULT)
        for comp, kv in PS.COORDINATES.items():
            for k, vals in kv.items():
                c[comp][k] = _rr.choice(vals)
        RANDP.append(PS.Policy(name=f"RAND:draw_{i:02d}", channel="random_basis", coords=c))
    # ---- BASE-MATCHED controls. COORD_ONLY below perturbs PS.DEFAULT, but the LLM's perturb channel
    # perturbs its OWN base (pool_size=exhaustive, pooling=mean). Comparing them directly repeats exactly the
    # base-mismatch error that made the channel audit report 5/5 false failures. These two arms perturb the
    # SAME implied base the LLM used, which splits the LLM's contribution into two separable questions:
    #   CMATCH_COORD — random coordinate, random value: did the LLM pick the right COORDINATE to move?
    #   CMATCH_VALUE — the coordinates the LLM chose, every OTHER value: did it pick the right VALUE?
    # If CMATCH_COORD matches the perturb channel, the coordinate system is doing the work, not the model.
    import collections as _cc
    CMC, CMV = [], []
    _per = [pp for c in conds for pp in c["policies"] if pp.channel == "perturb"]
    if _per:
        _fl = [pp.flat() for pp in _per]
        _implied = {k: _cc.Counter(f[k] for f in _fl).most_common(1)[0][0] for k in _fl[0]}
        # flat() keys are dotted ("acquisition.rule"); PS.COORDINATES keys are bare ("rule") and bare names
        # COLLIDE — stopping.rule and acquisition.rule share one bare name, 21 unique for 22 coordinates.
        # Everything below is keyed on the dotted form for that reason.
        def _tocoords(flatd):
            c = copy.deepcopy(PS.DEFAULT)
            for comp, kv in PS.COORDINATES.items():
                for k in kv:
                    dk = f"{comp}.{k}"
                    if dk in flatd: c[comp][k] = flatd[dk]
            return c
        # CMC and CMV must be DISJOINT FROM THE TREATMENT and built over BEHAVIOURALLY LIVE cells.
        # Two defects fixed here, both of which biased the contrast toward zero:
        #  (1) Disjointness. Enumerating "every other value" of a chosen coordinate re-creates the LLM's own
        #      policy; a random (coordinate, value) draw can land on one. Measured: 5/13 value_matched and
        #      5/20 coord_matched were BIT-IDENTICAL to a perturb policy.
        #  (2) Liveness. Selection used PS.applicable -- a DECLARED conditional -- not measured liveness. So
        #      8/16 coord_matched, 2/8 value_matched and 10/20 coord_only turned out to be the base under
        #      another name (bit-identical 24-seed vectors). An arm part-filled with copies of the base is
        #      biased toward the base, which is exactly the comparison being made.
        # `liveness_audit.json` is produced by audit_liveness.py, which varies one cell and compares full
        # traces. Only cells it marks LIVE are eligible. This caps the arms: at this base only 24 cells are
        # live across 14 coordinates, so coord_matched and value_matched cannot reach n=20 by construction.
        # That cap is a property of the space and is reported rather than worked around.
        _LIVE = set()
        _lpath = os.path.join(HERE, "liveness_audit.json")
        if os.path.exists(_lpath):
            for _rec in json.load(open(_lpath)):        # not _r: that name is `random`, imported above
                if _rec["verdict"] == "LIVE":
                    _LIVE.add((_rec["coordinate"], str(_rec["value"])))
        else:
            print("   WARNING: liveness_audit.json absent; arms fall back to declared applicability")

        def _is_live(dk, v):
            return (not _LIVE) or (dk, str(v)) in _LIVE

        _live_cells = [(comp, k, f"{comp}.{k}") for comp, kv in PS.COORDINATES.items() for k in kv
                       if PS.applicable(_implied, f"{comp}.{k}")]
        _chosen = sorted({k for f in _fl for k in f if f[k] != _implied[k]})
        _treat = {tuple(sorted(f.items())) for f in _fl}

        def _add(bucket, f, name, channel, seen):
            sig = tuple(sorted(f.items()))
            if sig in _treat or sig in seen:
                return
            seen.add(sig)
            bucket.append(PS.Policy(name=name[:44], channel=channel, coords=_tocoords(f)))

        # COORDINATE-CHOICE control: every LIVE cell on a coordinate the LLM did NOT choose.
        _seen = set()
        for comp, k, dk in _live_cells:
            if dk in _chosen:
                continue
            for v in PS.COORDINATES[comp][k]:
                if v == _implied[dk] or not _is_live(dk, v):
                    continue
                f = dict(_implied); f[dk] = v
                _add(CMC, f, f"CMC:{dk}={v}", "coord_matched", _seen)

        # VALUE-CHOICE control: every LIVE cell on a coordinate the LLM DID choose, except the LLM's value.
        _seenv = set()
        for dk in _chosen:
            comp, k = dk.split(".", 1)
            for v in PS.COORDINATES[comp][k]:
                if v == _implied[dk] or not _is_live(dk, v):
                    continue
                f = dict(_implied); f[dk] = v
                _add(CMV, f, f"CMV:{dk}={v}", "value_matched", _seenv)

    COORD = []                              # COORD_ONLY: every single-coordinate perturbation of the base, no LLM
    allc = [(comp, k, v) for comp, kv in PS.COORDINATES.items() for k, vals in kv.items()
            for v in vals if v != PS.DEFAULT[comp][k]]
    _rr2 = _r.Random(7)
    for i, (comp, k, v) in enumerate(_rr2.sample(allc, 20)):
        c = copy.deepcopy(PS.DEFAULT); c[comp][k] = v
        COORD.append(PS.Policy(name=f"COORD:{comp}.{k}={v}"[:44], channel="coord_only", coords=c))

    # ---------------- grounding audit (Gate B)
    print("\nGROUNDING AUDIT — a named gap is an output, a silent substitution is not")
    for c in conds:
        ok = sum(a["status"].startswith("OK") for a in c["audit"])
        inv = [a for a in c["audit"] if a["status"] == "INVALID"]
        gaps = [a for a in c["audit"] if a.get("missing_coordinate")]
        print(f"   {c['condition']:20s} proposed {c['n_proposed']:2d}  grounded {ok:2d}  INVALID {len(inv):2d}  "
              f"named-gaps {len(gaps):2d}")
        for a in inv[:4]: print(f"      INVALID  {a['name'][:34]:36s} {a['detail'][:70]}")
        for a in gaps[:6]: print(f"      GAP      {a['name'][:34]:36s} {a['missing_coordinate'][:70]}")

    # ---------------- CHANNEL-CONSTRAINT AUDIT, against the RIGHT reference base
    # An auditor reported that all 5/5 "one-coordinate perturbations" change three coordinates. That is true against
    # this harness's DEFAULT and it is the wrong comparison. Measured properly, each perturbation changes exactly ONE
    # coordinate relative to the *pool's own implied base* — the modal value of each coordinate across the channel —
    # and of the two extra differences from DEFAULT one (`representation.pooling`) is INERT for every one of them,
    # since they all use `encoding=onehot`. The channel is internally valid; the audit was not. Comparing against a
    # base the generator never adopted, and counting inapplicable coordinates as differences, both inflate the count.
    print("\nCHANNEL-CONSTRAINT AUDIT — one-coordinate perturbation, judged against the pool's own implied base")
    import collections as _co
    for c in conds:
        per = [p for p in c["policies"] if p.channel == "perturb"]
        if not per: continue
        flats = [p.flat() for p in per]
        implied = {k: _co.Counter(f[k] for f in flats).most_common(1)[0][0] for k in flats[0]}
        drift = {k: v for k, v in implied.items() if v != PS.make("b", "b").flat()[k]}
        ok = 0
        for p, f in zip(per, flats):
            d = [k for k in f if f[k] != implied[k] and PS.applicable(f, k)]
            ok += (len(d) == 1)
        print(f"   {c['condition']:20s} exactly one LIVE coordinate vs the implied base: {ok}/{len(per)}")
        print(f"      implied base drifts from harness DEFAULT at: "
              f"{', '.join(f'{k}={v}' for k, v in drift.items()) or 'nothing'}")
        inert_drift = [k for k in drift if not PS.applicable(flats[0], k)]
        if inert_drift:
            print(f"      of which INERT for every policy in the channel: {', '.join(inert_drift)}")

    # ---------------- inert-coordinate audit: does an asserted value actually change any proposal?
    INERT_FOR_ALL = []                       # coordinates the harness never reads at all
    src = open(os.path.join(HERE, "campaign.py")).read()
    for comp, kv in PS.COORDINATES.items():
        for k in kv:
            if f'"{k}"' not in src and f"'{k}'" not in src:
                INERT_FOR_ALL.append(f"{comp}.{k}")
    print("\nINERT-COORDINATE AUDIT — an asserted value that cannot change a proposal is a phantom difference")
    print(f"   coordinates never read by the harness: {INERT_FOR_ALL if INERT_FOR_ALL else 'none'}")
    for c in conds:
        n = 0
        for p in c["policies"]:
            f = p.flat()
            if any(f.get(k) != dict(PS.make('x','y').flat()).get(k) for k in INERT_FOR_ALL): n += 1
        print(f"   {c['condition']:20s} policies asserting an unread coordinate: {n}/{len(c['policies'])}")

    # ---------------- coverage
    print("\nCOORDINATE COVERAGE — twenty texts can be four decisions")
    for c in conds:
        r = PS.coverage_report(c["policies"])
        print(f"   {c['condition']:20s} {r['n_policies']:2d} policies, {r['distinct_signatures']:2d} distinct "
              f"assignments, {len(r['varied'])}/{r['n_coordinates']} coordinates varied")
        print(f"      never varied: {', '.join(r['untouched'][:8])}{' …' if len(r['untouched'])>8 else ''}")
        print(f"      by channel: {r['by_channel']}")

    # ---------------- execute
    allp = [p for c in conds for p in c["policies"]] + RANDP + COORD + CMC + CMV + SEEDPOL + [NULL]

    # ARMS / PER_ARM: run a decisive subset. At budget 480 a single policy costs ~140-205 s/seed, so the full
    # 105-policy roster would take over 100 hours -- the budget sweep has to be scoped or it cannot be run at
    # all. ARMS selects channels; PER_ARM caps how many policies are taken from each (deterministically, by
    # the roster order the generator produced, so the subset is reproducible and not cherry-picked).
    _arms = os.environ.get("ARMS")
    if _arms:
        keep = {a.strip() for a in _arms.split(",")}
        allp = [p for p in allp if p.channel in keep or p.name == "SEED:base_onehot_ridge"]
    _cap = int(os.environ.get("PER_ARM", 0))
    if _cap:
        seen_ct, capped = {}, []
        for p in allp:
            if p.name == "SEED:base_onehot_ridge":
                capped.append(p); continue
            n = seen_ct.get(p.channel, 0)
            if n < _cap:
                capped.append(p); seen_ct[p.channel] = n + 1
        allp = capped

    if os.environ.get("DRY_RUN"):
        # Build the policy list and report its structure without executing a single campaign. Added after an
        # invented AUDIT_ONLY flag silently launched a full 85-policy run instead of the audit it named.
        import collections as _c2
        print(f"DRY_RUN — {len(allp)} policies, no campaigns executed")
        for ch, n in sorted(_c2.Counter(p.channel for p in allp).items()):
            print(f"   {ch:16s} {n:3d}")
        _pr = [p for p in allp if p.channel == "perturb"]
        if _pr:
            _f = [p.flat() for p in _pr]
            _im = {k: _c2.Counter(x[k] for x in _f).most_common(1)[0][0] for k in _f[0]}
            for ch in ("coord_matched", "value_matched"):
                arm = [p for p in allp if p.channel == ch]
                if not arm: continue
                ds = [[k for k in p.flat() if p.flat()[k] != _im[k] and PS.applicable(p.flat(), k)] for p in arm]
                print(f"   {ch}: {sum(1 for d in ds if len(d)==1)}/{len(arm)} exactly one live coordinate "
                      f"from the perturb channel's own base")
        raise SystemExit(0)
    print(f"\nEXECUTING {len(allp)} policies x {N_SEEDS} seeds at matched budget ...")
    # PARALLEL OVER SEEDS. This is a compute detail and changes no declared design parameter: the same policies, the
    # same seeds, the same budget, the same scoring. Several policies request an exhaustive pool with a 2,480- or
    # 1,920-dimensional encoding and a Gaussian process, which costs minutes per seed serially; that cost is a real
    # property of those policies and is reported, but paying it serially would take days.
    # RESUMABLE. Background work does not survive a session boundary, and this run is ~1 hour of wall clock, so
    # completed policies are reloaded rather than recomputed. Every (policy, seed) is deterministic given the seed, so
    # resuming changes no number; it only avoids repeating work.
    RES = {}
    # Results are cached per policy NAME, so a resume is only valid if the cached rows were produced by the
    # SAME BUILD AND THE SAME OPERATING POINT. They were not, once: a budget-480 run resumed 97 rows from a
    # budget-96 cache and stamped the result with the 480 build hash. The hash recorded identity and nothing
    # checked it, which is worse than not recording it -- the file looked authoritative and was mixed.
    # The cache key now includes the operating point, and the stored build must match exactly.
    inc = os.path.join(HERE, f"pool_results_{LANDSCAPE}_s{N_SEEDS}_b{BUDGET}i{N_INIT}x{BATCH}.json")
    DONE = {}
    if os.path.exists(inc):
        try:
            _c = json.load(open(inc))
            _cb = _c.pop("__build__", None)
            if _cb == _BUILD:
                DONE = _c
                print(f"   (resuming: {len(DONE)} policies from build {_cb})", flush=True)
            else:
                print(f"   (cache at build {_cb} != current {_BUILD}; starting fresh, nothing reused)",
                      flush=True)
        except Exception:
            DONE = {}
    nproc = int(os.environ.get("NPROC", "4"))     # 4, not 8: each worker holds a ~320 MB encoded pool
    print(f"   (executing seeds in parallel across {nproc} processes; design unchanged)", flush=True)
    _newly_run = [0]
    ex = ProcessPoolExecutor(max_workers=nproc)
    for pi, p in enumerate(allp):
        if p.name in DONE and DONE[p.name].get("best") is not None:
            d = DONE[p.name]
            RES[p.name] = dict(policy=p, fail=d.get("fail", 0), best=np.array(d["seeds"]) if d.get("seeds")
                               else np.full(N_SEEDS, d["best"]), auc=d.get("auc", np.nan),
                               hits=d.get("hits", np.nan), topq=d.get("topq", np.nan),
                               sec=d.get("sec", 0.0), capped=d.get("capped", 0))
            print(f"   [{pi+1}/{len(allp)}] {p.name[:44]:46s} {d['best']:7.3f} (cached)", flush=True)
            continue
        if LIMIT is not None and _newly_run[0] >= LIMIT:
            print(f"   [{pi+1}/{len(allp)}] {p.name[:44]:46s} (deferred to next batch)", flush=True)
            continue
        _newly_run[0] += 1
        rs = list(ex.map(_one, [(p, s, BUDGET, N_INIT, BATCH) for s in SEEDS]))
        ok = [r for r in rs if r.ok]
        if not ok:
            RES[p.name] = dict(policy=p, fail=len(rs), best=np.nan)
            print(f"   [{pi+1}/{len(allp)}] {p.name[:44]:46s} ALL-FAILED  {rs[0].reason[:60]}", flush=True)
            continue
        RES[p.name] = dict(policy=p, fail=len(rs) - len(ok),
                           best=np.array([r.best for r in ok]), auc=np.mean([r.auc for r in ok]),
                           hits=np.mean([r.hits for r in ok]), topq=np.mean([r.top_q_found for r in ok]),
                           gain=float(np.mean([r.gain for r in ok])),
                           seed_ids=[int(r.seed) for r in ok],
                           sec=np.mean([r.seconds for r in ok]),
                           capped=sum(1 for r in ok if r.reason))
        _atomic_dump({k: dict(best=float(np.mean(v["best"])) if isinstance(v.get("best"), np.ndarray) else None,
                           seeds=[float(x) for x in v["best"]] if isinstance(v.get("best"), np.ndarray) else None,
                           seed_ids=v.get("seed_ids"),
                           auc=float(v.get("auc", np.nan)), hits=float(v.get("hits", np.nan)),
                           topq=float(v.get("topq", np.nan)), gain=float(v.get("gain", np.nan)),
                           channel=v["policy"].channel, fail=int(v.get("fail", 0)),
                           capped=int(v.get("capped", 0)), sec=float(v.get("sec", 0)))
                   for k, v in RES.items()} | {"__build__": _BUILD}, inc)
        print(f"   [{pi+1}/{len(allp)}] {p.name[:44]:46s} "
              f"{np.mean(RES[p.name]['best']) if isinstance(RES[p.name].get('best'), np.ndarray) else float('nan'):7.3f} "
              f"({RES[p.name].get('sec',0):.1f}s/seed)", flush=True)
    ex.shutdown(wait=True)
    pending = [p.name for p in allp if p.name not in RES or not isinstance(RES[p.name].get("best"), np.ndarray)]
    if pending:
        print(f"\n   {len(allp)-len(pending)}/{len(allp)} complete; {len(pending)} pending. "
              f"Re-run to continue.", flush=True)
        return
    base = RES["SEED:base_onehot_ridge"]["best"]

    # Every policy is run on the SAME seeds, so every comparison here is PAIRED. The unpaired two-sample SE
    # previously used, sqrt(var(a)/n + var(b)/n), ignores the seed-to-seed correlation it was designed around
    # and is the wrong denominator: it understates power when the arms are positively correlated across seeds
    # and overstates it when they are not. Paired SE is sd(a-b)/sqrt(n). A separate one-sample form is needed
    # for comparisons against a SCALAR (the library oracle), where the earlier code wrongly substituted the SE
    # of a different pair entirely.
    def se_paired(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = min(len(a), len(b))
        return float(np.std(a[:n] - b[:n], ddof=1) / np.sqrt(n))

    def se_scalar(a):
        a = np.asarray(a, float)
        return float(np.std(a, ddof=1) / np.sqrt(len(a)))

    def holm(pvals):
        """Holm-Bonferroni adjusted p-values; the table below runs one comparison per policy."""
        order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
        return adj

    print(f"\n{'policy':46s} {'channel':11s} {'best':>7s} {'sd':>6s} {'vs BASE':>8s} {'1.96SE':>7s} {'AUC':>6s} {'fail':>4s}")
    rows = sorted([v for v in RES.values() if isinstance(v.get("best"), np.ndarray)],
                  key=lambda v: -v["best"].mean())
    from scipy import stats as _st
    _pv = []
    for v in rows:
        n = min(len(v["best"]), len(base))
        _pv.append(float(_st.ttest_rel(v["best"][:n], base[:n]).pvalue) if n > 1 else 1.0)
    _adj = holm(np.array(_pv))
    for v, pr, pa in zip(rows, _pv, _adj):
        d = v["best"].mean() - base.mean()
        mark = "*" if pa < 0.05 and d > 0 else ("." if pr < 0.05 and d > 0 else " ")
        print(f"{v['policy'].name[:44]:46s} {v['policy'].channel:11s} {v['best'].mean():7.3f} "
              f"{v['best'].std(ddof=1):6.3f} {d:+8.3f}{mark} {1.96*se_paired(v['best'],base):7.3f} "
              f"{v['auc']:6.3f} {v['fail']:4d}")
    print(f"   * survives Holm across {len(rows)} paired comparisons;  . uncorrected p<0.05 only")
    print(f"   separable after Holm: {int((_adj < 0.05).sum())}/{len(rows)}   "
          f"uncorrected: {int((np.array(_pv) < 0.05).sum())}/{len(rows)} "
          f"(expected by chance at 0.05: {0.05*len(rows):.1f})")

    # ---------------- the comparison of record
    print("\n" + "=" * 118)
    print("THE COMPARISON OF RECORD")
    print("=" * 118)
    oracle = max((RES[p.name]["best"].mean() for p in SEEDPOL))
    print(f"   FIXED_LIBRARY_ORACLE (best seed in hindsight, a ceiling not a baseline): {oracle:.4f}")
    print(f"   NULL at equal budget: {RES['NULL:random']['best'].mean():.4f}   no-op anchor: {noop:.4f}")
    for c in conds:
        bs = [RES[p.name]["best"] for p in c["policies"] if isinstance(RES[p.name].get("best"), np.ndarray)]
        if not bs: continue
        means = np.array([b.mean() for b in bs])
        best_i = int(np.argmax(means))
        bestp = [p for p in c["policies"] if isinstance(RES[p.name].get("best"), np.ndarray)][best_i]
        # vs a SCALAR oracle: one-sample SE of that policy, not the SE of (policy, base)
        beat = sum(1 for b in bs if b.mean() - oracle > 1.96 * se_scalar(b))
        print(f"\n   {c['condition']}")
        print(f"      best policy        {bestp.name[:44]}  {means[best_i]:.4f} "
              f"(channel {bestp.channel})")
        print(f"      vs library oracle  {means[best_i]-oracle:+.4f}")
        print(f"      policies beating the oracle separably: {beat}/{len(bs)}")
        print(f"      pool mean {means.mean():.4f}, median {np.median(means):.4f}, "
              f"worst {means.min():.4f}")
        bych = {}
        for p, b in zip([p for p in c["policies"] if isinstance(RES[p.name].get("best"), np.ndarray)], bs):
            bych.setdefault(p.channel, []).append(b.mean())
        for ch, vv in sorted(bych.items()):
            print(f"         channel {ch:11s} n={len(vv):2d}  best {max(vv):.4f}  mean {np.mean(vv):.4f}")

    # ---------------- behavioural e-classes
    print("\n" + "=" * 118)
    print("BEHAVIOURAL E-CLASSES (complete linkage; tolerance = the instrument's own detectable difference)")
    print("=" * 118)
    nm = [v["policy"].name for v in rows]; mv = [v["best"].mean() for v in rows]
    tol = 1.96 * float(np.mean([np.sqrt(np.var(v["best"], ddof=1) / N_SEEDS) for v in rows])) * np.sqrt(2)
    cls = eclasses(nm, mv, tol)
    print(f"   {len(nm)} executed policies -> {len(cls)} classes at tolerance {tol:.4f}")
    for c in cls:
        vals = [mv[nm.index(x)] for x in c]
        print(f"      [{min(vals):.3f}–{max(vals):.3f}]  {', '.join(x[:30] for x in c[:6])}"
              f"{' …' if len(c) > 6 else ''}")

    _atomic_dump({k: (dict(best=float(v["best"].mean()), sd=float(v["best"].std(ddof=1)),
                        channel=v["policy"].channel, fail=int(v["fail"]))
                   if isinstance(v.get("best"), np.ndarray) else dict(fail=int(v.get("fail", 0))))
               for k, v in RES.items()} | {"__build__": _BUILD},
              os.path.join(HERE, f"pool_results_{LANDSCAPE}.json"))
    print(f"\nsaved -> pool_results_{LANDSCAPE}.json")


if __name__ == "__main__":
    _claim_writer_lock()      # refuse to start alongside another writer
    _n_names, _n_coords = assert_every_coordinate_is_read()
    main()
