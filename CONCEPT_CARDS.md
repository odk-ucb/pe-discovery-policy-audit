# Paper concept cards — Gate A

Format per the governing review §5.3. **Fields are filled only from material actually held in `SOURCES.md`
and the cited papers. Anything not grounded is written `UNKNOWN`, never estimated** — an invented field would
defeat the purpose of the blinded reconstruction audit these cards exist to support.

`evidence_locations` is `UNKNOWN` throughout: section- and figure-level locators were not retained when the
sources were first read. Recovering them requires re-reading the five papers and is listed as outstanding in
the audit status at the foot of this file.

---

```yaml
paper_id: S1 · ALDE — active learning-assisted directed evolution (Nat. Commun. 2025, 10.1038/s41467-025-55987-8)
problem_state: A combinatorial site-saturation library over a small number of positions, with a wet-lab assay
  that can be run for a few rounds of modest batch size.
binding_bottleneck: Epistasis. Single-step greedy directed evolution hill-climbs into a local optimum because
  each step is judged only by immediate improvement.
goal: Reach a high-fitness combination within a few assay rounds.
method_in_two_sentences: Alternate wet-lab assay rounds with refitting a surrogate over the library, using
  quantified uncertainty to choose the next batch. Because the surrogate reports uncertainty rather than only
  a point prediction, the campaign can spend part of its batch on informative rather than merely promising
  variants, and so escape a local optimum.
load_bearing_mechanism: Quantified uncertainty converts the batch choice from exploitation into an
  explore/exploit trade, which is what allows escape from the local optimum epistasis creates.
components:
  - name: encoding
    role: map a variant sequence to features
    inputs: variant sequence
    outputs: feature vector
    dependencies: none
  - name: surrogate
    role: predict fitness with a calibrated uncertainty
    inputs: encoded observed variants and their measured fitness
    outputs: predictive mean and standard deviation over candidates
    dependencies: encoding
  - name: acquisition
    role: rank candidates from mean and uncertainty
    inputs: predictive mean, predictive sd, round index
    outputs: score per candidate
    dependencies: surrogate
  - name: batch
    role: select the round's assay set
    inputs: candidate scores
    outputs: batch of variants
    dependencies: acquisition
changeable_coordinates:
  - component: acquisition
    coordinate: rule
    source_value: compared greedy, UCB, and Thompson sampling
    credible_alternatives: expected improvement; pure exploration
    invariants: must consume only surrogate outputs available before the round is run
    expected_effect: governs the explore/exploit balance; the paper's own comparison is evidence this
      coordinate is live rather than assumed
  - component: surrogate
    coordinate: uncertainty
    source_value: quantified uncertainty (the load-bearing choice)
    credible_alternatives: ensemble spread; GP posterior; bootstrap
    invariants: the acquisition rule must be able to read an sd
    expected_effect: removing it collapses the method to greedy directed evolution
assumptions: The assay is reproducible enough that surrogate refits are meaningful across rounds; the library
  is combinatorially enumerable.
constraints: A small number of wet-lab rounds; modest batch size per round.
measured_outcomes: Three wet-lab rounds took a non-native cyclopropanation yield from 12% to 93%. Validated in
  simulation on two combinatorially complete landscapes.
failure_modes: UNKNOWN
transferable_transition: "Replace a greedy step rule with an uncertainty-aware one when the objective is known
  to be epistatic." This is a policy-level transition, not a model-level one.
evidence_locations: UNKNOWN
```

---

```yaml
paper_id: S2 · EVOLVEpro
problem_state: Few labelled variants, a large candidate space, and a pretrained protein language model
  available.
binding_bottleneck: Label scarcity. A high-capacity surrogate cannot be fitted from the number of labels a
  campaign can afford.
goal: Improve an activity with very few labelled rounds.
method_in_two_sentences: Embed variants with a frozen protein language model and fit a random-forest surrogate
  on those embeddings. Each round, assay the top-predicted variants and refit.
load_bearing_mechanism: The pretrained representation supplies features that generalise from very few labels,
  so a low-capacity surrogate is sufficient and the label budget goes further.
components:
  - name: representation
    role: encode variants using a frozen pretrained model
    inputs: variant sequence
    outputs: embedding
    dependencies: none (model is frozen; no fine-tuning)
  - name: surrogate
    role: predict fitness
    inputs: embeddings and labels
    outputs: predicted fitness
    dependencies: representation
  - name: acquisition
    role: rank candidates
    inputs: predicted fitness
    outputs: score
    dependencies: surrogate
  - name: update
    role: refit and iterate
    inputs: accumulated observations
    outputs: refitted surrogate
    dependencies: surrogate
changeable_coordinates:
  - component: representation
    coordinate: encoding
    source_value: frozen protein language model embeddings
    credible_alternatives: one-hot; physicochemical descriptors; fine-tuned embeddings
    invariants: must be computable for every candidate before assay
    expected_effect: the paper's central claim; substituting one-hot tests whether pretraining is load-bearing
  - component: surrogate
    coordinate: family
    source_value: random forest
    credible_alternatives: ridge; GP; boosted ensemble
    invariants: must fit from tens of labels
    expected_effect: the claim is that a LOW-capacity surrogate suffices given the representation, so raising
      capacity should not help and may hurt
  - component: acquisition
    coordinate: rule
    source_value: greedy (top-predicted)
    credible_alternatives: UCB; Thompson; expected improvement
    invariants: none
    expected_effect: greedy is the deliberate choice here, in contrast to S1
assumptions: The pretrained model's representation is informative for the specific property being optimised.
constraints: Very few labels.
measured_outcomes: UNKNOWN (not retained in SOURCES.md at the level of specific figures)
failure_modes: UNKNOWN
transferable_transition: "Move surrogate capacity out of the model and into the representation when labels are
  the binding constraint."
evidence_locations: UNKNOWN
```

---

```yaml
paper_id: S3 · Evaluation of MLDE across diverse combinatorial landscapes (Cell Systems 2025)
problem_state: A fixed label budget to spend across a combinatorial landscape, with zero-shot predictors
  available before any label is collected.
binding_bottleneck: Where the labels are spent. Random sampling spends most of a small budget on
  non-functional variants.
goal: Get more out of a fixed label budget.
method_in_two_sentences: Choose the training set using zero-shot predictors — evolutionary, structural or
  stability-based — rather than sampling at random, then fit a surrogate and select. The prior concentrates
  labels on plausibly functional variants, so the same budget buys a more informative training set.
load_bearing_mechanism: A prior that concentrates labels on plausibly functional variants raises the value of
  each label, independently of the surrogate that is fitted afterwards.
components:
  - name: training-set selection
    role: choose which variants to label first
    inputs: zero-shot scores over the library
    outputs: initial labelled set
    dependencies: zero-shot prior
  - name: zero-shot prior
    role: score variants before any label exists
    inputs: sequence, structure, or stability model
    outputs: score per variant
    dependencies: none
  - name: surrogate
    role: predict fitness from the focused training set
    inputs: labelled set
    outputs: predictions
    dependencies: training-set selection
  - name: acquisition
    role: select final candidates
    inputs: predictions
    outputs: ranked candidates
    dependencies: surrogate
changeable_coordinates:
  - component: generator
    coordinate: zero_shot_prior
    source_value: focused training using an evolutionary/structural/stability zero-shot predictor
    credible_alternatives: none (random sampling); a different prior family
    invariants: the prior must be computable without labels
    expected_effect: the paper's central claim; its value depends on the prior actually correlating with
      fitness on the target landscape, which is landscape-specific and therefore testable
assumptions: The zero-shot predictor is informative for the target property.
constraints: Fixed label budget.
measured_outcomes: Focused training with zero-shot predictors consistently beat random sampling. MLDE's
  advantage over directed evolution is larger on landscapes that are harder for directed evolution.
failure_modes: Implied but not enumerated — a prior uninformative for the target property should remove the
  benefit. This is directly testable on GB1, where the ESM-2 masked-marginal prior ranks the true optimum
  27,495th of 149,361 (Spearman rho +0.096), so the mechanism is predicted to be near-useless or harmful here.
transferable_transition: "Spend the prior, not the budget: use a label-free predictor to decide where labels
  go."
evidence_locations: UNKNOWN
```

---

```yaml
paper_id: S4 · ProSpero — active learning beyond wild-type neighbourhoods (arXiv:2505.22494)
problem_state: A campaign whose candidate generator proposes variants near the wild type.
binding_bottleneck: Reachability. Local search cannot reach optima requiring several simultaneous
  substitutions.
goal: Reach distant optima without wasting budget on implausible sequences.
method_in_two_sentences: Propose candidates outside the wild type's local neighbourhood while retaining a
  robustness or plausibility constraint. Distant proposals are filtered for plausibility rather than accepted
  freely, so the search gains reach without paying for nonsense.
load_bearing_mechanism: Decoupling *reach* from *acceptance* — the generator widens, and a separate constraint
  does the filtering that proximity used to do implicitly.
components:
  - name: candidate generator
    role: propose variants
    inputs: observed set, wild type
    outputs: candidate pool
    dependencies: none
  - name: robustness constraint
    role: filter implausible proposals
    inputs: candidate pool
    outputs: filtered pool
    dependencies: candidate generator
  - name: acquisition
    role: rank the filtered pool
    inputs: surrogate predictions
    outputs: ranked candidates
    dependencies: robustness constraint
changeable_coordinates:
  - component: generator
    coordinate: scope
    source_value: beyond the wild-type neighbourhood
    credible_alternatives: local ball of radius r; global; recombination of observed elites
    invariants: the pool must remain enumerable at the harness's cap
    expected_effect: directly controls reachability; on GB1 the optimum is at Hamming distance 4 from the
      wild type, so a scope restricted to a small local ball provably cannot reach it
  - component: generator
    coordinate: ball_centre
    source_value: not the wild type
    credible_alternatives: wild type; best observed; top-k observed
    invariants: centre must be a sequence in the space
    expected_effect: interacts with scope; an induced coordinate added because this source requires it
assumptions: A plausibility signal exists that is cheaper than the assay.
constraints: UNKNOWN
measured_outcomes: UNKNOWN
failure_modes: UNKNOWN
transferable_transition: "When the optimum is out of local reach, widen the generator and add an explicit
  plausibility filter rather than relying on proximity as an implicit one."
evidence_locations: UNKNOWN
```

---

```yaml
paper_id: S5 · Protein language models with an automated biofoundry (Nat. Commun. 2025, 10.1038/s41467-025-56751-8)
problem_state: Assays are cheap and highly parallel because experimentation is automated.
binding_bottleneck: Round schedule. Batch size and round count are usually set by intuition or by wet-lab
  convenience rather than by information return.
goal: Exploit platform throughput rather than inherit manual-campaign habits.
method_in_two_sentences: Couple model-guided selection directly to automated experimentation, so batch size
  and round count follow platform throughput. Many small rounds are preferred to few large ones because each
  refit then uses strictly more information than the one before.
load_bearing_mechanism: Refit cadence. Holding total budget fixed, more rounds means each selection is made by
  a surrogate trained on more data — the gain comes from the schedule, not from the model.
components:
  - name: batch size
    role: how many variants per round
    inputs: platform throughput
    outputs: batch size
    dependencies: none
  - name: round schedule
    role: how many rounds the budget is split into
    inputs: total budget, batch size
    outputs: number of rounds
    dependencies: batch size
  - name: representation
    role: encode variants
    inputs: sequence
    outputs: features
    dependencies: none
  - name: update cadence
    role: when the surrogate is refitted
    inputs: round boundaries
    outputs: refitted surrogate
    dependencies: round schedule
changeable_coordinates:
  - component: batch
    coordinate: size_rule
    source_value: set by platform throughput; many small rounds
    credible_alternatives: fixed large batch; annealed batch size
    invariants: total budget fixed, so batch size and round count trade off exactly
    expected_effect: the paper's claim, and it is a pure schedule effect — testable at fixed total budget with
      no change to representation, surrogate, or acquisition
  - component: update
    coordinate: refit
    source_value: refit every round
    credible_alternatives: refit every k rounds; warm-started refit
    invariants: refit must precede the next selection
    expected_effect: the mechanism by which more rounds pays
assumptions: Assay cost per variant is roughly independent of batch size — the claim depends on this and it
  fails when there is a large fixed cost per round.
constraints: Platform throughput.
measured_outcomes: UNKNOWN
failure_modes: A large per-round fixed cost inverts the conclusion.
transferable_transition: "Treat the round schedule as a decision variable rather than a fixed property of the
  setting."
evidence_locations: UNKNOWN
```

---

## Audit status

**Gate A is not passed by the existence of these cards.** It requires a *blinded reconstruction audit*: a
reader who sees only a card, and not the paper, reconstructs the method, and the reconstruction is scored
against the source. Recording the protocol and the denominator here so the gate has a definition to be
measured against rather than an impression.

**Protocol.**
1. For each card, a reader with no access to the source paper writes the method as executable coordinate
   assignments in the 22-coordinate space.
2. The reconstruction is scored against the assignment derived from the paper itself, coordinate by
   coordinate, over the coordinates the card lists under `changeable_coordinates`.
3. Denominator = number of scored coordinates across all cards: **10** (S1 · 2, S2 · 3, S3 · 1, S4 · 2,
   S5 · 2). An earlier version of this file said 11; that was an arithmetic error in the sum of its own
   per-card counts, caught by recounting the `- component:` blocks. Score = coordinates reconstructed to the
   correct value.
4. A card whose `load_bearing_mechanism` is reconstructed incorrectly fails as a whole regardless of
   coordinate score, since the mechanism is what makes the transition transferable.

**Status: RUN — see `gate_a_result.md`. Denominator 10.** No blinded reconstruction has been performed, so Gate A currently has a
denominator and no numerator — which is still an improvement on having neither.

**Known gaps in the cards themselves**, which the audit must not be allowed to paper over:
- `evidence_locations` is UNKNOWN for all five; section/figure locators were not retained.
- `measured_outcomes` is UNKNOWN for S2, S4, S5.
- `failure_modes` is UNKNOWN for S1, S2, S4.
- S3's failure mode is the only one derived rather than reported, and it is derived from a measurement on
  *this* landscape, not from the source.
