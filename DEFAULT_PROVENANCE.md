# Default policy provenance — `SEED:base_onehot_ridge`

**Status:** factual record for the paper and release. Not a post-hoc story.

## What the default is

Code constant `DEFAULT` in `policy_space.py` (compiled as `SEED:base_onehot_ridge`):

| Component | Value |
|---|---|
| representation.encoding | `onehot` |
| representation.pooling | `site` (inert for one-hot) |
| representation.rank | `full` |
| representation.site_coupling | `independent_sites` |
| surrogate.family | `ridge` |
| surrogate.count | `single` |
| surrogate.heads | `joint_fitness` |
| surrogate.uncertainty | `none` |
| generator.scope | `global_all_variants` |
| generator.zero_shot_prior | `none` |
| acquisition.rule | `greedy_mean` |
| batch.selection | `top_k_by_score` |
| update.refit | `refit_from_scratch` |
| update.target_transform | `identity` |
| stopping.rule | `fixed_rounds` |

Code comment (frozen with the constant):

> “The DEFAULT policy is a deliberate, literature-standard baseline: the configuration a competent practitioner would reach for, so that a perturbation is a perturbation OF something real rather than of a straw man.”

## How it was chosen (honest)

1. **Role:** fixed **harness default** and base for `coord_only` / literature-standard comparison — not an LLM proposal and not the winner of a search over LLM outputs.
2. **Selection rule:** classical MLDE-style campaign: one-hot (or site-local) encoding + linear/ridge surrogate + greedy exploitation + global candidate pool. This is the simplest widely used PE campaign skeleton (cf. early MLDE practice and many factorial PE studies that treat one-hot + simple regressor as a baseline cell).
3. **Not done:** we did **not** choose the default by ranking free-form LLM policies on GB1 and keeping the best. The constant is the base of `make()` for every seed policy and control construction.
4. **Landscape knowledge:** the *form* of the default is literature-standard for combinatorial PE, not a GB1-specific hyperparameter sweep reported in this project. We **do not** claim the numerical values of every coordinate were locked in a third-party registry before any landscape was loaded; we claim they are a pre-declared code constant used uniformly across GB1/TEV/TrpB4 and not fit to LLM outcomes.
5. **Implication for the central null:** “0 of N beat the default” means “no tested policy beat this fixed classical baseline under matched seeds/budget,” not “no policy beat the oracle-best configuration in hindsight.” Literature-shaped recipes (EVOLVEpro-/ALDE-style) are also beaten by or tied below this default at matched budget in our harness — which argues the default is strong as a classical PE baseline, not a straw man.

## What would further strengthen this claim

- External time-stamped freeze (git tag / OSF) before any GB1 LLM pool execution.
- Side-by-side with original published ALDE/EVOLVEpro *code*, not only recipes.

Until those exist, the paper must use the wording above and must not say “pre-registered before any GB1 data were seen” unless a dated external lock is added.
