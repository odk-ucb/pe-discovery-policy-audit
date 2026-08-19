# Source library and frozen campaign state

Frozen before any candidate is generated. Nothing here contains a measured outcome from this experiment.

## The frozen campaign state `S`

| field | value |
|---|---|
| **O** objects | A protein variant library over **four mutable sites** of a 56-residue binding domain. Each site takes any of 20 amino acids, so the design space is 20⁴. A wild-type sequence is known. Each variant has a scalar fitness obtainable only by assay. |
| **R** relations | Fitness is epistatic across the four sites: it is not a sum of per-site effects. A surrogate fitted to observed variants can predict unobserved ones with unknown reliability. Assays are run in batches. |
| **B** bottleneck | The assay budget. Only **96** variants may be assayed in total, in **4 rounds of 24**, starting from **24** randomly chosen observed variants. |
| **A** assumptions | The candidate universe is fixed and enumerable. The oracle is deterministic. No variant may be assayed twice. Every policy receives the identical initial 24 observations at a given seed. |
| **C** constraints | A policy may use only variants it has already paid for, plus any computation on the unlabelled candidate universe. It may not query the oracle outside its batch, and it may not use fitness values it has not paid for. |
| **G** goal | Maximise the **highest fitness discovered** within the budget. Secondary: area under the best-so-far curve, and probability of finding a top-1% variant. |
| **E** evidence available | Sequences of all candidates; a frozen protein-language-model embedding of every candidate (two poolings); a frozen zero-shot likelihood score for every candidate; the fitnesses of variants already assayed. |

**Why policy design is non-trivial on this state.** The global optimum sits at Hamming distance **4 of 4** from the wild
type — the maximum possible. Only 2.4% of variants exceed wild-type fitness, and the best single mutant reaches under
half the optimum. A generator restricted to the wild type's immediate neighbourhood therefore cannot reach the optimum.

## The base policy, which perturbations perturb

A literature-standard configuration: one-hot encoding, ridge surrogate, no uncertainty, global candidate pool, greedy
acquisition on the predicted mean, top-k batch selection, refit from scratch each round, fixed batch size.

## Direct sources — protein-engineering discovery policies

**S1 · ALDE — Active learning-assisted directed evolution** (Nature Communications 2025, doi:10.1038/s41467-025-55987-8)
- π: alternate wet-lab assay with surrogate refit; use **uncertainty quantification** to prioritise the next batch.
- C: encoding · surrogate · acquisition · uncertainty quantification · batch.
- M: epistasis defeats single-step greedy directed evolution; quantified uncertainty lets the campaign leave a local
  optimum instead of hill-climbing into it.
- R: three wet-lab rounds took a non-native cyclopropanation yield from 12% to 93%; validated by simulation on two
  **combinatorially complete** landscapes. Compares three acquisitions: greedy, UCB, Thompson sampling.

**S2 · EVOLVEpro**
- π: embed variants with a frozen protein language model; fit a **random-forest** surrogate; iterate, taking the
  top-predicted variants each round.
- C: representation (frozen PLM) · surrogate (random forest) · acquisition (greedy) · update (iterate).
- M: a pretrained sequence model supplies features that generalise from very few labels, so a low-capacity surrogate
  suffices.

**S3 · Evaluation of MLDE across diverse combinatorial landscapes** (Cell Systems 2025)
- π: **focused training** — choose the training set using zero-shot predictors (evolutionary, structural, stability
  knowledge) rather than sampling at random, then fit and select.
- C: training-set selection · zero-shot prior · surrogate · acquisition.
- M: a prior that concentrates labels on plausibly functional variants raises the value of a small label budget.
- R: focused training with zero-shot predictors consistently beat random sampling; MLDE's advantage is larger on
  landscapes that are harder for directed evolution.

**S4 · ProSpero — active learning beyond wild-type neighbourhoods** (arXiv 2505.22494)
- π: propose candidates outside the wild type's local neighbourhood while retaining robustness.
- C: candidate generator scope · robustness constraint · acquisition.
- M: local search cannot reach optima that require several simultaneous substitutions; distant proposals must be
  filtered for plausibility rather than accepted freely.

**S5 · Protein language models with an automated biofoundry** (Nature Communications 2025, doi:10.1038/s41467-025-56751-8)
- π: couple model-guided selection to automated experimentation, so batch size and round count are set by platform
  throughput rather than by intuition.
- C: batch size · round schedule · representation · update cadence.
- M: when assays are cheap and parallel, many small rounds beat few large ones because each refit uses more information.

## Cross-domain donors — HAND-SPECIFIED, NOT RETRIEVED

> **Provenance correction (2026-08-18).** This section previously read "retrieved only against a named
> bottleneck". That claim is false and is withdrawn. **No retrieval was performed and no retrieval code
> exists in this directory.** D1–D5 below are the five examples named in the governing review
> (`Reviews/2026-08-17_220325_..._Phase2_GPT.md`, line 528: "cost-aware BO, safe/constrained BO, active
> search, bandit portfolio allocation or decoupled estimation") reproduced **in that same order**. The
> bottleneck sentences attached to each were written afterwards, by me, to fit donors I had already been
> handed.
>
> What the `project` channel can therefore support: *given* a cross-domain donor, it can be expressed as a
> point in the coordinate system and executed. What it cannot support: any claim that the method
> **finds** distant donors. Retrieval is the part of "cross-domain projection" that would make it a
> method rather than a transcription, and it is the part that is missing.

**D1 · Cost-aware Bayesian optimisation.** Acquisition divided by the cost of evaluation, so the search prefers
information per unit cost rather than information alone. *Bottleneck matched:* a fixed budget where evaluations differ in
value.

**D2 · Safe / constrained Bayesian optimisation.** Maintain a feasible set and only expand into regions the surrogate
believes satisfy a constraint. *Bottleneck matched:* most of the space is non-functional, and sampling it wastes budget.

**D3 · Active search (as distinct from active learning).** The objective is to find as many members of a rare positive
class as possible, not to fit an accurate model. *Bottleneck matched:* the goal here is the best variant found, not
predictive accuracy.

**D4 · Bandit portfolio allocation.** Run several strategies concurrently and shift budget toward whichever is paying
off, instead of committing to one. *Bottleneck matched:* which policy is best is unknown at the campaign's start.

**D5 · Decoupled estimation — separating selection from evaluation.** One estimator proposes, an independently fitted
estimator scores, so the optimiser cannot exploit the errors of the estimator that chose. *Bottleneck matched:* greedy
selection on a surrogate's own predictions compounds that surrogate's errors. (Structural donor: the double-estimator
correction for maximisation bias in value estimation.)
