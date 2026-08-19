"""The component-coordinate space for a protein DISCOVERY POLICY, and its executable substrate.

WHAT IS BEING INVENTED HERE. Not a protein method. The object under study is a *discovery policy* — the algorithm that
decides which variants to assay next — and the contribution is a machine that INVENTS such policies by decomposing
published methods into components, probing each component's coordinates, projecting transformations in from other
domains, and recombining. The protein landscape is the objective environment that says whether an invented policy works.

THE DECOMPOSITION. A discovery policy has seven components. Each has coordinates: the axes along which it can vary.

    representation   how a variant becomes a vector
    surrogate        what predicts fitness from that vector
    generator        which candidate variants are even considered
    acquisition      how predicted value and uncertainty become a score
    batch            how a batch is chosen given per-candidate scores
    update           how the surrogate is refit between rounds
    stopping         when the campaign stops or changes regime

THE COORDINATE SYSTEM IS OPEN, AND THAT IS A DESIGN REQUIREMENT RATHER THAN A CONVENIENCE. This project has already
measured what a closed operator vocabulary costs: a fixed set of transformation verbs recovered the historical move in
**0 of 5** cases, and in the algorithmic-move experiment a closed 14-operator space forced a strong grounder to refuse
**9 of 20** proposals — every refusal naming a real capability the space lacked. Widening the space on those refusals
took operator coverage from 4/14 to 10/20 and admitted a composite operator that turned out to be second-best on the
real end task.

So the values enumerated below are a STARTING BASIS, not the space. A proposal that does not fit is recorded `INVALID`
with the missing coordinate named, and the basis is widened from those refusals. `coverage_report()` exists to make the
gap visible rather than absorbed.

WHAT I AM CARRYING OVER FROM THE ALGORITHMIC-MOVE EXPERIMENT, because each was learned by getting it wrong:
  - **Matched budget is exact.** Policies get identical initial observations, identical seeds, and identical numbers of
    true-fitness queries. Ten separate accounting errors in that experiment came from comparing at nominally equal
    budgets that were not equal; here the budget is a counter of oracle calls and is asserted, not assumed.
  - **A matched-resource null AND a no-op comparator.** "Better than random at equal budget" is necessary and NOT
    sufficient: on a structureless artifact a magnitude-based rule beat random at margin 0.996 while doing nothing at
    all. Every campaign therefore reports the best fitness already present in its initial observations.
  - **Measure the instrument's noise before comparing anything.** Seed-to-seed spread of the SAME policy, first.
  - **Complete linkage for behavioural equivalence**, since "within tolerance" is not transitive.
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# COORDINATES. The dict value is the enumerated starting basis for that coordinate; the LLM may propose others and a
# proposal outside the basis is INVALID-with-a-named-gap, never silently mapped to the nearest entry.
# ----------------------------------------------------------------------------------------------------------------
COORDINATES = {
    "representation": {
        # `esm2_frozen` is a real frozen protein-language-model embedding (ESM-2 35M, precomputed for the entire
        # candidate universe by embed.py). It replaces an earlier `learned_embedding_frozen` value that was PCA of a
        # one-hot — not a learned embedding at all, and a placeholder that would have made any finding about this
        # coordinate meaningless. Encoding is one of the axes the protein-engineering literature actually varies.
        "encoding": ["onehot", "physicochem", "onehot_plus_pairs", "esm2_frozen"],
        # CONDITIONAL COORDINATE: pooling has meaning only when encoding == "esm2_frozen". A design space with
        # conditional coordinates is the normal case, not a defect, and it is recorded as conditional rather than
        # silently ignored — an inapplicable coordinate must not be counted as "varied" in coverage_report().
        "pooling": ["mean", "site"],
        "rank": ["full", "low_rank_pca"],
        "site_coupling": ["independent_sites", "pairwise_sites"],
    },
    "surrogate": {
        "family": ["ridge", "random_forest", "gaussian_process", "boosted_trees", "knn"],
        "count": ["single", "ensemble_of_k"],
        "heads": ["joint_fitness", "separated_activity_and_fitness"],
        "uncertainty": ["none", "ensemble_disagreement", "gp_posterior", "bootstrap"],
    },
    "generator": {
        "scope": ["global_all_variants", "local_hamming_ball", "recombine_observed_elites"],
        # INDUCED FROM THE GENERATORS, round 1. Both conditions named the ball's CENTRE as unstateable, and the
        # compositional one noted it "decides whether the optimum is reachable at all" on this state — correctly, since
        # the optimum sits at Hamming distance 4 of 4 from the wild type. It was implicitly elite-centred and is now
        # explicit, with the wild-type-centred value available so the reachability claim can be tested rather than
        # asserted.
        "ball_centre": ["observed_elites", "wild_type"],
        "radius": ["r1", "r2", "adaptive_radius"],
        "pool_size": ["small", "medium", "exhaustive"],
        # INDUCED FROM THE LITERATURE, iteration 72, which is the coordinate system working as intended: this axis did
        # not exist until decomposing "focused training with zero-shot predictors" (Cell Systems 2025) named it.
        # Implemented as ESM-2 masked marginals over the four sites (4 forward passes for the whole universe).
        # MEASURED INERT-TO-HARMFUL ON GB1: rho with true fitness is +0.096 (35M) and +0.126 (650M), and either filter
        # excludes the optimum from its top 10,000. Retained because the coordinate is real and its value is
        # state-dependent, not because it helps here.
        "zero_shot_prior": ["none", "esm_masked_marginal_focus"],
    },
    "acquisition": {
        "rule": ["greedy_mean", "ucb", "expected_improvement", "thompson", "pure_explore"],
        "beta": ["low", "medium", "high", "state_adaptive_beta"],
        "lookahead": ["one_step", "two_step_myopic_rollout"],
    },
    "batch": {
        "selection": ["top_k_by_score", "diverse_top_k_by_distance", "score_then_cluster", "epsilon_mixed"],
        "size_rule": ["fixed", "state_adaptive_size"],
    },
    "update": {
        "refit": ["refit_from_scratch", "warm_start"],
        "weighting": ["uniform", "recency_weighted", "elite_weighted"],
        # INDUCED FROM THE GENERATORS, round 1. BOTH conditions independently named this gap: the response is
        # zero-inflated and heavy-tailed (median ~0.2, max 8.76), and `weighting` reweights ROWS but cannot rescale
        # the TARGET. Predictions are made on the transformed scale and ranked there; the oracle is untouched.
        "target_transform": ["identity", "log1p", "rank"],
    },
    "stopping": {
        "rule": ["fixed_rounds", "plateau_detect", "budget_exhausted"],
    },
}

# The DEFAULT policy is a deliberate, literature-standard baseline: the configuration a competent practitioner would
# reach for, so that a perturbation is a perturbation OF something real rather than of a straw man.
DEFAULT = {
    "representation": {"encoding": "onehot", "pooling": "site", "rank": "full",
                       "site_coupling": "independent_sites"},
    "surrogate": {"family": "ridge", "count": "single", "heads": "joint_fitness", "uncertainty": "none"},
    "generator": {"scope": "global_all_variants", "ball_centre": "observed_elites", "radius": "r1",
                  "pool_size": "medium", "zero_shot_prior": "none"},
    "acquisition": {"rule": "greedy_mean", "beta": "low", "lookahead": "one_step"},
    "batch": {"selection": "top_k_by_score", "size_rule": "fixed"},
    "update": {"refit": "refit_from_scratch", "weighting": "uniform", "target_transform": "identity"},
    "stopping": {"rule": "fixed_rounds"},
}


def n_coordinates():
    return sum(len(v) for v in COORDINATES.values())


def n_configurations():
    n = 1
    for comp in COORDINATES.values():
        for vals in comp.values():
            n *= len(vals)
    return n


@dataclass
class Policy:
    """An executable discovery policy: a full assignment over the coordinate system."""
    name: str
    channel: str                       # free | perturb | project | recombine | baseline | null
    coords: dict
    provenance: dict = field(default_factory=dict)

    def flat(self):
        return {f"{c}.{k}": v for c, comp in self.coords.items() for k, v in comp.items()}

    def signature(self):
        return tuple(sorted(self.flat().items()))

    def differs_from(self, other):
        a, b = self.flat(), other.flat()
        return {k: (a[k], b[k]) for k in a if a.get(k) != b.get(k)}


def make(name, channel, **overrides):
    """Build a policy as DEFAULT plus explicit coordinate overrides, e.g. make('x','perturb', surrogate={'count':'ensemble_of_k'})."""
    import copy
    c = copy.deepcopy(DEFAULT)
    for comp, kv in overrides.items():
        if comp not in c:
            raise KeyError(f"unknown component '{comp}'; components are {list(c)}")
        for k, v in kv.items():
            if k not in c[comp]:
                raise KeyError(f"unknown coordinate '{comp}.{k}'")
            if v not in COORDINATES[comp][k]:
                raise ValueError(f"'{v}' is outside the enumerated basis for {comp}.{k} = "
                                 f"{COORDINATES[comp][k]}. Widen the basis explicitly rather than substituting.")
            c[comp][k] = v
    return Policy(name=name, channel=channel, coords=c)


# coordinates that only have meaning when another coordinate takes a particular value
CONDITIONAL = {"representation.pooling": ("representation.encoding", {"esm2_frozen"})}


def applicable(flat, key):
    """Is `key` a live coordinate for this policy, or inert given its other coordinates?"""
    if key not in CONDITIONAL: return True
    dep, ok = CONDITIONAL[key]
    return flat.get(dep) in ok


def coverage_report(policies):
    """Which coordinates the pool actually varies, and which it never touches.

    This is the report that mattered most in the algorithmic-move experiment: a pool can look like twenty candidates
    and be four decisions. Text-level diversity and coordinate-level diversity are different objects and both are
    reported; behavioural equivalence is a third and is measured later, on outcomes.
    """
    touched, values = {}, {}
    for p in policies:
        for k, v in p.flat().items():
            values.setdefault(k, set()).add(v)
    all_k = [f"{c}.{k}" for c, comp in COORDINATES.items() for k in comp]
    # a coordinate counts as VARIED only among the policies where it is applicable; otherwise a pool could look
    # diverse by varying a coordinate that does nothing in every policy that carries it.
    live = {}
    for p in policies:
        f = p.flat()
        for k, v in f.items():
            if applicable(f, k): live.setdefault(k, set()).add(v)
    varied = [k for k in all_k if len(live.get(k, set())) > 1]
    untouched = [k for k in all_k if len(live.get(k, set())) <= 1]
    return dict(n_policies=len(policies), n_coordinates=len(all_k), varied=varied, untouched=untouched,
                distinct_signatures=len({p.signature() for p in policies}),
                by_channel={ch: sum(1 for p in policies if p.channel == ch)
                            for ch in sorted({p.channel for p in policies})})


if __name__ == "__main__":
    print(f"components            {len(COORDINATES)}")
    print(f"coordinates           {n_coordinates()}")
    print(f"enumerated basis size {n_configurations():,} full configurations")
    print()
    for c, comp in COORDINATES.items():
        print(f"  {c:16s} " + ", ".join(f"{k}({len(v)})" for k, v in comp.items()))
    print("\nDEFAULT policy (the literature-standard baseline a perturbation perturbs):")
    for k, v in make("default", "baseline").flat().items():
        print(f"   {k:34s} {v}")
    print("\nthe basis is OPEN: a proposal outside it is INVALID with the missing coordinate named, never substituted.")
