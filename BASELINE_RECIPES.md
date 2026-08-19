# Literature-shaped recipe baselines (honest scope)

These are **not** original ALDE / EVOLVEpro / BO-EVO codebases.
They are first-class policies in our coordinate space whose coordinate assignments
are chosen to **resemble** published recipes, then executed in our harness under
matched seeds/budget.

| Recipe label | Intended literature peer | Coordinate sketch | What we do **not** claim |
|---|---|---|---|
| `SEED:base_onehot_ridge` | Classical MLDE-style default | one-hot + ridge + greedy-mean + global pool | Not a third-party timestamped default |
| `SEED:esm_rf_greedy` | PLM + tree surrogate campaigns | esm2_frozen site + RF + greedy | Not EVOLVEpro original code |
| `SEED:gp_ucb` | BO-style UCB | GP posterior + UCB medium β | Not BO-EVO original code |
| `SEED:rf_thompson` | Ensemble Thompson | RF ensemble disagreement + Thompson | Not ALDE original code |
| ALDE-like / EVOLVEpro-like free proposals | Named in free pool rationales | Generator-chosen coords | Recipes, not paper re-implementations |

**Implication for reviews:** beating or losing to these recipes is evidence about
**our harness + classical PE skeleton**, not a claim that we re-ran published
repositories bit-for-bit. Closing that gap would require vendoring upstream code
under their licenses and matched splits—future work.

Sources: `SOURCES.md` (ALDE Nat Comm 2025; EVOLVEpro; SSMuLA; BO-EVO; ftMLDE).
