"""Executable discovery policies, and the equal-budget campaign harness that scores them.

WHAT A POLICY IS, OPERATIONALLY. A coordinate assignment over the seven components in `policy_space.py` compiles to a
function `propose(observed) -> batch`. Nothing else about a policy is allowed to matter: two assignments that compile to
the same proposals are the same policy, which is what makes behavioural equivalence measurable rather than rhetorical.

THE FIVE DISCIPLINES CARRIED OVER, each learned by getting it wrong in this project's previous experiment:

  1. **The budget is a counter, not a convention.** Every oracle call is counted and asserted equal across policies.
     Ten matched-budget accounting errors in the previous experiment came from budgets that were nominally equal.
  2. **A matched-resource null AND a no-op anchor.** `RANDOM` selects uniformly at the same budget; the no-op anchor is
     the best fitness already present in the initial observations. "Beats random" is necessary and insufficient: on a
     structureless artifact a magnitude rule beat random at margin 0.996 while beating nothing at all.
  3. **The instrument's noise before any comparison.** `noise_floor()` runs the SAME policy over many seeds. No
     difference smaller than that spread is a finding.
  4. **Identical initial observations per seed**, shared across all policies, so a policy is never rewarded for a lucky
     start.
  5. **Failures are outcomes.** A policy that raises, returns duplicates, or proposes unmeasured variants is recorded
     `INVALID` with the reason; it is never silently repaired.

THE ORACLE IS NEVER VISIBLE TO A POLICY. `propose` receives only the variants it has already paid for and their
fitnesses. Scoring uses ground truth; policies do not.
"""
import json, os, time
import numpy as np
from dataclasses import dataclass, field

from landscape import Landscape, WT, AAS
import policy_space as PS

SITES = 4
MAX_ENCODED_ELEMENTS = 40_000_000        # ~320 MB float64; declared before any pool was generated
# A uniform cap on the number of candidates SCORED per round, applied identically to every policy in both conditions.
# Without it, `pool_size: exhaustive` under a cheap encoding scores all 149,361 candidates, and a Gaussian process
# doing that costs 218 s per seed. The cap is declared, applied equally, and REPORTED whenever it binds, so a policy
# that wanted an unbounded pool is recorded as having been capped rather than quietly rescoped. A real campaign is
# also unable to score an unbounded pool, so `exhaustive` means "the largest pool this harness will score".
MAX_POOL = 20_000


# ----------------------------------------------------------------------------------------------------------------
# representation
# ----------------------------------------------------------------------------------------------------------------
_AAI = {a: i for i, a in enumerate(AAS)}
# five standard physicochemical scales (Kyte-Doolittle hydropathy, volume, charge, polarity, aromaticity)
_PHYS = {
    "A": (1.8, 88.6, 0, 0, 0), "C": (2.5, 108.5, 0, 0, 0), "D": (-3.5, 111.1, -1, 1, 0),
    "E": (-3.5, 138.4, -1, 1, 0), "F": (2.8, 189.9, 0, 0, 1), "G": (-0.4, 60.1, 0, 0, 0),
    "H": (-3.2, 153.2, 0.1, 1, 1), "I": (4.5, 166.7, 0, 0, 0), "K": (-3.9, 168.6, 1, 1, 0),
    "L": (3.8, 166.7, 0, 0, 0), "M": (1.9, 162.9, 0, 0, 0), "N": (-3.5, 114.1, 0, 1, 0),
    "P": (-1.6, 112.7, 0, 0, 0), "Q": (-3.5, 143.8, 0, 1, 0), "R": (-4.5, 173.4, 1, 1, 0),
    "S": (-0.8, 89.0, 0, 1, 0), "T": (-0.7, 116.1, 0, 1, 0), "V": (4.2, 140.0, 0, 0, 0),
    "W": (-0.9, 227.8, 0, 1, 1), "Y": (-1.3, 193.6, 0, 1, 1),
}


_ESM = {}
_ZS = {}
_LANDSCAPE_NAME = "GB1"          # set by Harness.__init__; keys the PLM caches below


def _zs():
    """Cached ESM-2 masked-marginal zero-shot score for the current landscape's universe.

    Keyed by landscape name, not a bare module global -- a bare global would silently hand TEV or TrpB4
    campaigns GB1's zero-shot scores under variant strings that happen to collide or simply be wrong for that
    landscape, which is a correctness bug worse than a missing feature: it produces a number instead of an
    error. Only landscapes with a built zeroshot cache load; others raise a named, catchable error.
    """
    global _ZS
    if _LANDSCAPE_NAME not in _ZS:
        zs_files = {
            "GB1": "gb1_zeroshot_esm2.npz",
            "TEV": "tev_zeroshot_esm2.npz",
            "TrpB4": "trpb4_zeroshot_esm2.npz",
        }
        fn = zs_files.get(_LANDSCAPE_NAME)
        if not fn:
            raise RuntimeError(f"zero_shot_prior requires a cache not built for landscape={_LANDSCAPE_NAME}")
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", fn)
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"zero-shot cache missing for {_LANDSCAPE_NAME}: {f}; run zeroshot_multi.py {_LANDSCAPE_NAME}"
            )
        z = np.load(f)
        _ZS[_LANDSCAPE_NAME] = {str(v): float(x) for v, x in zip(z["variants"], z["zs"])}
    return _ZS[_LANDSCAPE_NAME]


def _esm():
    """Load the frozen ESM-2 cache for the current landscape. Keyed by name for the same reason as `_zs`."""
    global _ESM
    if _LANDSCAPE_NAME not in _ESM:
        esm_files = {
            "GB1": "gb1_esm2_35M.npz",
            "TEV": "tev_esm2_35M.npz",
            "TrpB4": "trpb4_esm2_35M.npz",
        }
        fn = esm_files.get(_LANDSCAPE_NAME)
        if not fn:
            raise RuntimeError(
                f"representation.encoding=esm2_frozen requires a cache not built for "
                f"landscape={_LANDSCAPE_NAME}"
            )
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", fn)
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"frozen ESM-2 cache missing for {_LANDSCAPE_NAME}: {f}; "
                f"run `python3 embed.py` (GB1) or `python3 embed_multi.py {_LANDSCAPE_NAME}`"
            )
        z = np.load(f)
        _ESM[_LANDSCAPE_NAME] = dict(mean=z["mean"], site=z["site"],
                                     ix={str(v): i for i, v in enumerate(z["variants"])})
    return _ESM[_LANDSCAPE_NAME]


def encode(variants, coords, pca=None):
    enc = coords["representation"]["encoding"]
    if enc == "onehot":
        X = np.zeros((len(variants), SITES * 20))
        for i, v in enumerate(variants):
            for s, a in enumerate(v): X[i, s * 20 + _AAI[a]] = 1.0
    elif enc == "physicochem":
        X = np.array([[c for a in v for c in _PHYS[a]] for v in variants], float)
    elif enc == "onehot_plus_pairs":
        base = encode(variants, {"representation": {"encoding": "onehot"}})
        extra = np.zeros((len(variants), 6 * 400))
        for i, v in enumerate(variants):
            k = 0
            for s1 in range(SITES):
                for s2 in range(s1 + 1, SITES):
                    extra[i, k * 400 + _AAI[v[s1]] * 20 + _AAI[v[s2]]] = 1.0
                    k += 1
        X = np.hstack([base, extra])
    elif enc == "esm2_frozen":
        # a real frozen protein-language-model embedding, precomputed for the whole universe by embed.py. Pooling is a
        # coordinate: "mean" over residues (480 dims) or the four mutated sites concatenated (1920 dims).
        E = _esm()
        pool = coords["representation"].get("pooling", "site")
        M = E["site"] if pool == "site" else E["mean"]
        X = np.array([M[E["ix"][v]] for v in variants], dtype=np.float64)
    else:
        raise ValueError(f"unimplemented encoding '{enc}'")
    # PIPELINE ORDER MATTERS AND WAS WRONG. The PCA branch used to return BEFORE the pairwise augmentation, while the
    # PCA was fitted on a path that applied it — so a policy combining `low_rank_pca` with `pairwise_sites` fitted the
    # projection on 4,320 features and then applied it to 1,920, raising ValueError on every seed. Augment first, then
    # reduce; both the fitting and the scoring path now traverse the same order.
    if coords["representation"].get("site_coupling") == "pairwise_sites" and enc != "onehot_plus_pairs":
        X = np.hstack([X, encode(variants, {"representation": {"encoding": "onehot_plus_pairs"}})[:, SITES * 20:]])
    if coords["representation"].get("rank") == "low_rank_pca" and pca is not None:
        X = pca.transform(X)
    return X


# ----------------------------------------------------------------------------------------------------------------
# surrogate
# ----------------------------------------------------------------------------------------------------------------
def _make_model(family, seed):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    if family == "ridge": return Ridge(alpha=1.0, random_state=None) if False else Ridge(alpha=1.0)
    if family == "random_forest": return RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=1)
    if family == "boosted_trees": return GradientBoostingRegressor(random_state=seed)
    if family == "knn": return KNeighborsRegressor(n_neighbors=5)
    if family == "gaussian_process":
        k = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed, alpha=1e-6)
    raise ValueError(f"unimplemented surrogate family '{family}'")


class Surrogate:
    """Fits one or many models; exposes mean and an uncertainty channel selected by coordinate."""

    def __init__(self, coords, seed, prev=None):
        self.c = coords
        self.seed = seed
        self.models, self.clf = [], None
        # WARM_START (was declared but never read — audit finding). Only tree ensembles can genuinely warm-start by
        # growing more estimators; for ridge/GP/kNN there is nothing to carry, so the coordinate is INERT for those
        # families and `self.warm_inert` records that instead of pretending otherwise.
        fam = coords["surrogate"]["family"]
        self.want_warm = coords["update"].get("refit") == "warm_start"
        self.warm_inert = self.want_warm and fam not in ("random_forest", "boosted_trees")
        self.prev = prev if (self.want_warm and not self.warm_inert) else None

    def fit(self, X, y, w=None):
        sc = self.c["surrogate"]
        k = 5 if sc["count"] == "ensemble_of_k" else 1
        self.models = []
        rng = np.random.default_rng(self.seed)
        for j in range(k):
            if self.prev is not None and j < len(self.prev.models):
                m = self.prev.models[j]
                try:
                    m.set_params(warm_start=True, n_estimators=m.get_params()["n_estimators"] + 20)
                except Exception:
                    m = _make_model(sc["family"], self.seed + j)
            else:
                m = _make_model(sc["family"], self.seed + j)
            if k > 1:                                   # bootstrap resample so the ensemble actually disagrees
                ix = rng.integers(0, len(y), len(y))
                Xi, yi = X[ix], y[ix]
                wi = None if w is None else w[ix]
            else:
                Xi, yi, wi = X, y, w
            try:
                m.fit(Xi, yi, sample_weight=wi) if wi is not None else m.fit(Xi, yi)
            except TypeError:
                m.fit(Xi, yi)
            self.models.append(m)
        if sc["heads"] == "separated_activity_and_fitness":
            # DECOUPLE SELECTION FROM EVALUATION, projected in from the Double-DQN mechanism: one model decides which
            # variants are plausibly active, an independently fitted model scores fitness among them.
            from sklearn.linear_model import LogisticRegression
            act = (y > np.median(y)).astype(int)
            if act.min() != act.max():
                self.clf = LogisticRegression(max_iter=400).fit(X, act)

    def predict(self, X):
        P = np.stack([m.predict(X) for m in self.models])
        mu = P.mean(0)
        u = self.c["surrogate"]["uncertainty"]
        if u == "ensemble_disagreement" and len(self.models) > 1:
            sd = P.std(0)
        elif u == "gp_posterior" and self.c["surrogate"]["family"] == "gaussian_process":
            _, sd = self.models[0].predict(X, return_std=True)
        elif u == "bootstrap" and len(self.models) > 1:
            sd = P.std(0)
        else:
            sd = np.zeros_like(mu)
        if self.clf is not None:
            mu = mu * self.clf.predict_proba(X)[:, 1]
        return mu, sd


# ----------------------------------------------------------------------------------------------------------------
# generator, acquisition, batch
# ----------------------------------------------------------------------------------------------------------------
def _neighbours(seqs, radius, universe):
    out = set()
    for s in seqs:
        for i in range(SITES):
            for a in AAS:
                if a == s[i]: continue
                c = s[:i] + a + s[i + 1:]
                if c in universe: out.add(c)
    if radius >= 2:
        base = list(out)
        for s in base[:400]:
            for i in range(SITES):
                for a in AAS:
                    c = s[:i] + a + s[i + 1:]
                    if c in universe: out.add(c)
    return out


# ---- Independent RNG streams, one per subsystem.
# A reviewer demonstrated that a `thompson` twin with sd identically zero — mathematically the same policy as
# greedy — scored -0.4844 against BASE over 24 seeds and differed on 22/24. Cause: every subsystem drew from a
# single Generator, so thompson's standard_normal() draw advanced the stream that pool subsampling and batch
# selection read next. Policies differing in one coordinate therefore saw different pools, and the difference
# was scored as that coordinate's effect. Each subsystem now gets a stream derived from (seed, subsystem, round),
# so its draws cannot depend on how many numbers any other subsystem consumed.
_STREAM_ID = {"candidates": 1, "recombine": 2, "poolcap": 3, "acquisition": 4,
              "pick": 5, "nullbatch": 6, "initial": 7}


def stream(seed, name, round_idx=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), _STREAM_ID[name], int(round_idx)]))


def candidates(coords, observed, universe, uset, rng, round_idx, n_rounds, batch_needed=24, seed=0):
    # two independent streams: recombination draws must not shift the pool-capping draw
    rng_recomb = stream(seed, "recombine", round_idx)
    rng_cap = stream(seed, "poolcap", round_idx)
    g = coords["generator"]
    seen = set(observed)
    if g["scope"] == "global_all_variants":
        pool = universe
    elif g["scope"] == "local_hamming_ball":
        rad = 1 if g["radius"] == "r1" else 2 if g["radius"] == "r2" else (1 if round_idx < n_rounds / 2 else 2)
        # ball_centre is now explicit (induced coordinate): wild-type-centred provably cannot reach an optimum at
        # Hamming distance 4, elite-centred can walk there over rounds.
        if g.get("ball_centre", "observed_elites") == "wild_type":
            centres = [WT]
        else:
            centres = [v for v, _ in sorted(observed.items(), key=lambda kv: -kv[1])[:20]]
        pool = list(_neighbours(centres, rad, uset))
    elif g["scope"] == "recombine_observed_elites":
        elites = [v for v, _ in sorted(observed.items(), key=lambda kv: -kv[1])[:24]]
        out = set()
        for _ in range(4000):
            a, b = rng_recomb.choice(len(elites), 2, replace=True)
            c = "".join(elites[a][i] if rng_recomb.random() < 0.5 else elites[b][i] for i in range(SITES))
            if c in uset: out.add(c)
        pool = list(out)
    else:
        raise ValueError(f"unimplemented generator scope '{g['scope']}'")
    pool = [v for v in pool if v not in seen]
    # ZERO_SHOT_PRIOR (was declared but never read — audit finding). Focus the pool on the top half by ESM-2
    # masked-marginal score. Measured on GB1 this prior is near-useless (rho +0.096) and ranks the optimum
    # 27,495th of 149,361, so it is expected to HURT here; implementing it makes that a measured outcome rather than an
    # unexercised coordinate.
    if g.get("zero_shot_prior") == "esm_masked_marginal_focus" and pool:
        z = _zs()
        pool = sorted(pool, key=lambda v: -z.get(v, -1e9))[:max(batch_needed, len(pool) // 2)]
    cap = {"small": 2000, "medium": 20000, "exhaustive": len(universe)}[g["pool_size"]]
    # A DECLARED COMPUTATIONAL CAP, reported rather than silent. An exhaustive pool under a 1920-dim ESM encoding is
    # 149,361 x 1920 = 287M floats (2.3 GB) per round, which is a real cost of that policy and not a harness defect.
    # The cap binds on the ENCODED size, so cheap encodings keep their full pool; when it binds it is recorded, because
    # computational cost is a reported endpoint and a silently truncated pool would misattribute the policy's
    # behaviour to its acquisition rule.
    dim = {"onehot": 80, "physicochem": 20, "onehot_plus_pairs": 2480,
           "esm2_frozen": (1920 if coords["representation"].get("pooling", "site") == "site" else 480)}[
        coords["representation"]["encoding"]]
    hard = max(2000, int(MAX_ENCODED_ELEMENTS // dim))
    eff = min(cap, hard, MAX_POOL)
    capped = len(pool) > eff
    if capped:
        pool = [pool[i] for i in rng_cap.choice(len(pool), eff, replace=False)]
    return pool, capped


_BETA = {"low": 0.5, "medium": 1.5, "high": 3.0}


def score(coords, mu, sd, best_obs, rng, round_idx, n_rounds):
    a = coords["acquisition"]
    b = a["beta"]
    beta = (_BETA[b] if b in _BETA else 3.0 * (1.0 - round_idx / max(n_rounds - 1, 1)))  # state_adaptive_beta: anneal
    r = a["rule"]
    if r == "greedy_mean": return mu
    if r == "ucb": return mu + beta * sd
    if r == "pure_explore": return sd + 1e-9 * mu
    if r == "thompson": return mu + sd * rng.standard_normal(len(mu))
    if r == "expected_improvement":
        from scipy.stats import norm
        s = np.maximum(sd, 1e-9); z = (mu - best_obs) / s
        return (mu - best_obs) * norm.cdf(z) + s * norm.pdf(z)
    raise ValueError(f"unimplemented acquisition rule '{r}'")


def apply_lookahead(coords, base_score, pool, mu, uset):
    """TWO_STEP_MYOPIC_ROLLOUT (was declared but never read — audit finding).

    A one-step score asks "how good is this variant". A two-step score asks "and what does observing it open up".
    Implemented as a discounted best-neighbour term: each candidate's score gains gamma times the best predicted
    value among its Hamming-1 neighbours that are themselves in the pool. That is a genuine lookahead over the
    search graph and costs one dictionary pass, not a fantasised refit.
    """
    if coords["acquisition"].get("lookahead") != "two_step_myopic_rollout":
        return base_score
    gamma = 0.5
    pos = {v: i for i, v in enumerate(pool)}
    out = np.array(base_score, dtype=float, copy=True)
    for i, v in enumerate(pool):
        best = 0.0
        for k in range(SITES):
            for a in AAS:
                if a == v[k]: continue
                c = v[:k] + a + v[k + 1:]
                j = pos.get(c)
                if j is not None and mu[j] > best: best = mu[j]
        out[i] += gamma * best
    return out


def pick(coords, pool, sc, n, rng, X=None):
    b = coords["batch"]["selection"]
    order = np.argsort(-sc)
    if b == "top_k_by_score":
        return [pool[i] for i in order[:n]]
    if b == "epsilon_mixed":
        k = max(1, int(0.75 * n)); chosen = [pool[i] for i in order[:k]]
        rest = [pool[i] for i in order[k:]]
        if rest: chosen += [rest[i] for i in rng.choice(len(rest), min(n - k, len(rest)), replace=False)]
        return chosen
    if b == "diverse_top_k_by_distance":
        chosen = []
        for i in order:
            v = pool[i]
            if all(sum(1 for x, y in zip(v, c) if x != y) >= 2 for c in chosen):
                chosen.append(v)
            if len(chosen) == n: break
        j = 0
        while len(chosen) < n and j < len(order):
            v = pool[order[j]]
            if v not in chosen: chosen.append(v)
            j += 1
        return chosen
    if b == "score_then_cluster":
        from sklearn.cluster import KMeans
        top = order[:max(n * 8, n)]
        Xt = X[top]
        kk = min(n, len(top))
        # EXACT ISOMETRY, not an approximation. Clustering 192 points in 2,480 dimensions took 23 s per call — 66% of
        # a whole campaign — because sklearn's k-means++ init is pathological for d >> n here. Euclidean distances
        # among m points are *exactly* preserved by projecting onto their own row space (rank <= m-1), so an SVD to
        # that rank leaves every pairwise distance, and therefore the clustering, unchanged to numerical precision.
        # This is a compute fix with no semantic content; the policy still clusters in its own representation.
        if Xt.shape[1] > Xt.shape[0]:
            Xc = Xt - Xt.mean(0, keepdims=True)
            U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
            Xt = U * sv                                  # coordinates in the data's own span; distances identical
        lab = KMeans(n_clusters=kk, n_init=4, random_state=0).fit_predict(Xt)
        chosen = []
        for c in range(kk):
            m = [t for t, l in zip(top, lab) if l == c]
            if m: chosen.append(pool[max(m, key=lambda t: sc[t])])
        return chosen[:n]
    raise ValueError(f"unimplemented batch selection '{b}'")


# ----------------------------------------------------------------------------------------------------------------
# the harness
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    policy: str
    channel: str
    seed: int
    ok: bool
    reason: str = ""
    best: float = float("nan")
    best_reliable: float = float("nan")
    noop: float = float("nan")
    best_discovered: float = float("nan")   # best among QUERIED variants only, excludes the free anchor
    gain: float = float("nan")              # best - noop: what the budget actually bought
    auc: float = float("nan")
    regret: float = float("nan")
    hits: int = 0
    top_q_found: bool = False
    calls: int = 0
    seconds: float = 0.0
    trace: list = field(default_factory=list)


class Harness:
    def __init__(self, budget=96, n_init=24, batch=24, universe="all", q=0.99, landscape=None,
                 landscape_name="GB1"):
        # Defaults track the declared operating point. They previously read budget=192/48/48 and
        # universe="reliable" — the retired configuration, whose reliable subset holds 98.6% of the
        # above-WT variants in 22.7% of the space, so any caller relying on defaults searched a
        # far easier landscape than the one reported.
        #
        # `landscape` accepts any object presenting the same surface as `Landscape` (see
        # ssmula_landscape.SSMuLALandscape). `landscape_name` sets the global that keys the PLM caches in
        # `_zs`/`_esm`, so a non-GB1 landscape raises a clear error on those coordinates instead of silently
        # reusing GB1's cache.
        global _LANDSCAPE_NAME
        _LANDSCAPE_NAME = landscape_name
        self.L = landscape if landscape is not None else Landscape()
        self.budget, self.n_init, self.batch = budget, n_init, batch
        assert budget % batch == 0, "budget must be a whole number of batches"
        self.n_rounds = budget // batch
        self.universe = self.L.reliable_variants() if universe == "reliable" else self.L.all_variants()
        self.uset = set(self.universe)
        self.thr = self.L.hit_threshold(q, over=universe)
        self.thr_frac = self.L.hit_fraction(self.thr, over=universe)   # true selectivity, for reporting
        self.true_best_v, self.true_best_f = self.L.true_best(True)
        self._pca = None

    def initial(self, seed):
        """Identical initial observations for every policy at a given seed."""
        rng = np.random.default_rng(10_000 + seed)
        ix = rng.choice(len(self.universe), self.n_init, replace=False)
        vs = [self.universe[i] for i in ix]
        return {v: float(self.L.f[self.L.idx[v]]) for v in vs}

    def _pca_for(self, coords):
        if coords["representation"]["rank"] != "low_rank_pca": return None
        # the cache key must name every coordinate that changes the FEATURE DIMENSION, or two policies sharing an
        # encoding but differing in site_coupling silently share a projection fitted for the wrong width
        key = (coords["representation"]["encoding"], coords["representation"].get("pooling"),
               coords["representation"].get("site_coupling"))
        if getattr(self, "_pca_key", None) != key:
            from sklearn.decomposition import PCA
            sub = self.universe[::40]
            Xs = encode(sub, coords)
            # Clamp to what the encoding can supply. A hardcoded rank of 24 crashed on any encoding with
            # fewer than 24 features, which in the 118-policy screen struck only the random_basis control --
            # so the control lost members to a harness limitation rather than to policy quality, flattering
            # every LLM arm it was there to test. min() returns 24 whenever 24 components were obtainable,
            # so no policy that already succeeded can change.
            ncomp = int(min(24, Xs.shape[1], Xs.shape[0]))
            self._pca = PCA(n_components=ncomp, random_state=0).fit(Xs)
            self._pca_key = key
        return self._pca

    def run(self, pol, seed):
        t0 = time.time()
        obs = dict(self.initial(seed))
        noop = max(obs.values())
        rng = np.random.default_rng(seed)
        calls, trace, capped_rounds = 0, [max(obs.values())], 0
        prev_surrogate = None
        try:
            pca = self._pca_for(pol.coords)
            for r in range(self.n_rounds):
                if pol.channel == "null":                       # matched-resource null: uniform at equal budget
                    pool = [v for v in self.universe if v not in obs]
                    chosen = [pool[i] for i in stream(seed, "nullbatch", r)
                              .choice(len(pool), self.batch, replace=False)]
                else:
                    pool, was_capped = candidates(pol.coords, obs, self.universe, self.uset,
                                                  stream(seed, "candidates", r), r, self.n_rounds,
                                                  self.batch, seed=seed)
                    capped_rounds += int(was_capped)
                    if len(pool) < self.batch:
                        raise RuntimeError(f"generator produced {len(pool)} < batch {self.batch} at round {r}")
                    ov = list(obs); oy = np.array([obs[v] for v in ov])
                    Xo = encode(ov, pol.coords, pca)
                    w = None
                    wt = pol.coords["update"]["weighting"]
                    if wt == "recency_weighted":
                        w = np.linspace(0.5, 1.5, len(oy))
                    elif wt == "elite_weighted":
                        w = 1.0 + 2.0 * (oy >= np.quantile(oy, 0.75))
                    # target transform (induced coordinate). Applied to the RESPONSE only; the oracle is untouched
                    # and scoring always uses raw fitness. Ranking is monotone-invariant, so this can only change
                    # what the SURROGATE learns, which is the point.
                    tt = pol.coords["update"].get("target_transform", "identity")
                    if tt == "log1p":
                        yfit = np.log1p(np.maximum(oy, 0.0))
                    elif tt == "rank":
                        from scipy.stats import rankdata
                        yfit = rankdata(oy) / len(oy)
                    else:
                        yfit = oy
                    S = Surrogate(pol.coords, seed, prev=prev_surrogate)
                    S.fit(Xo, yfit, w)
                    prev_surrogate = S
                    Xp = encode(pool, pol.coords, pca)
                    mu, sd = S.predict(Xp)
                    sc = score(pol.coords, mu, sd, float(yfit.max()),
                               stream(seed, "acquisition", r), r, self.n_rounds)
                    sc = apply_lookahead(pol.coords, sc, pool, mu, self.uset)
                    nb = self.batch
                    if pol.coords["batch"]["size_rule"] == "state_adaptive_size":
                        nb = self.batch          # size must stay matched; adaptivity is spent WITHIN the round
                    chosen = pick(pol.coords, pool, sc, nb, stream(seed, "pick", r), Xp)
                if len(set(chosen)) != self.batch:
                    raise RuntimeError(f"batch of {len(set(chosen))} distinct, expected {self.batch}")
                for v in chosen:
                    if v in obs: raise RuntimeError("policy re-queried an already observed variant")
                    fv = self.L.query(v)
                    if fv is None: raise RuntimeError(f"proposed unmeasured variant {v}")
                    obs[v] = fv; calls += 1
                    trace.append(max(trace[-1], fv))
            assert calls == self.budget, f"budget violation: {calls} != {self.budget}"
        except Exception as e:
            return Result(pol.name, pol.channel, seed, False, f"{type(e).__name__}: {str(e)[:120]}",
                          calls=calls, seconds=round(time.time() - t0, 2))
        queried = [v for v in obs if v not in self.initial(seed)]
        fq = np.array([obs[v] for v in queried])
        rel = np.array([self.L.reliable[self.L.idx[v]] for v in queried])
        best = float(max(trace))
        # `best` is the best-so-far curve endpoint and therefore INCLUDES the free no-op anchor: a policy that
        # discovers nothing still reports max(initial). The review measured NULL discovering nothing on 6/24
        # seeds while still contributing 35.9% of its own mean that way. `gain` is what the policy actually
        # bought with its budget, and it is the quantity comparisons should use.
        best_discovered = float(fq.max()) if len(fq) else float("nan")
        gain = best - noop
        best_rel = float(fq[rel].max()) if rel.any() else float("nan")
        auc = float(np.mean(trace))
        # Keyword arguments throughout: this call was positional, so inserting two fields into the dataclass
        # silently shifted auc into best_discovered and regret into gain. Positional construction of a growing
        # record is a standing trap; keywords make field order irrelevant.
        res = Result(policy=pol.name, channel=pol.channel, seed=seed, ok=True, reason="",
                     best=best, best_reliable=best_rel, noop=noop,
                     best_discovered=best_discovered, gain=gain, auc=auc,
                     regret=self.true_best_f - best, hits=int((fq >= self.thr).sum()),
                     top_q_found=bool(best >= self.thr), calls=calls,
                     seconds=round(time.time() - t0, 2), trace=trace)
        res.reason = f"pool capped in {capped_rounds}/{self.n_rounds} rounds" if capped_rounds else ""
        return res

    def noise_floor(self, pol, seeds):
        """The instrument's own spread: the SAME policy over many seeds. Run before comparing any two policies."""
        rs = [self.run(pol, s) for s in seeds]
        ok = [r for r in rs if r.ok]
        b = np.array([r.best for r in ok])
        return dict(n=len(ok), failed=len(rs) - len(ok), mean=float(b.mean()), sd=float(b.std(ddof=1)),
                    lo=float(b.min()), hi=float(b.max()),
                    detectable=float(1.96 * b.std(ddof=1) * np.sqrt(2)),
                    noop_mean=float(np.mean([r.noop for r in ok])))


if __name__ == "__main__":
    H = Harness()
    print("=" * 104)
    print(f"harness: budget {H.budget} assays, {H.n_rounds} rounds x {H.batch}, initial sample {H.n_init}")
    print(f"   candidate universe {len(H.universe):,} reliable variants; true best {H.true_best_v} "
          f"at {H.true_best_f:.4f}; active-hit threshold {H.thr:.4f}")
    base = PS.make("BASE_default", "baseline")
    null = PS.Policy(name="RANDOM_null", channel="null", coords=PS.DEFAULT)
    seeds = list(range(8))
    print("\nNOISE FLOOR FIRST — the same policy over 8 seeds, before any comparison")
    for p in (base, null):
        nf = H.noise_floor(p, seeds)
        print(f"   {p.name:14s} best {nf['mean']:.4f} +- {nf['sd']:.4f}  range {nf['lo']:.3f}-{nf['hi']:.3f}  "
              f"detectable diff {nf['detectable']:.4f}  no-op anchor {nf['noop_mean']:.4f}  failed {nf['failed']}")
    print("\n   No difference between two policies smaller than 'detectable diff' is a finding.")
    print("   No policy that fails to beat the no-op anchor has discovered anything.")
