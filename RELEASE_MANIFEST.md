# Release manifest

This lists exactly what is prepared for release with `PAPER_DISCOVERY_POLICY.md`, under `LICENSE` (MIT for
code; see that file's note on SSMuLA data provenance). This directory is release-ready as of 2026-08-18; it
has not yet been pushed to a public host, and the paper should not claim it has until it is.

## Code (all original, MIT)

| file | role |
|---|---|
| `policy_space.py` | the 22-coordinate (20 after phantom removal) declared space, conditional-applicability rules |
| `landscape.py` | GB1 landscape loader, reliability filter, hit-threshold/fraction |
| `ssmula_landscape.py` | TEV/TrpB4 loader over SSMuLA's published fitness tables (data only, no SSMuLA code) |
| `campaign.py` | the executable harness: encoders, surrogates, acquisition rules, generators, the per-subsystem RNG streams, the writer lock and atomic-write discipline |
| `run_pools.py` | orchestrates a full sweep: grounding/liveness/channel-constraint audits, arm construction (including the disjoint, behaviourally-live `coord_matched`/`value_matched` construction), execution, the statistics block (Holm, TOST) |
| `analyze.py` | the analysis of record: paired-on-seed-id contrasts, attrition handling (complete-case / intention-to-treat), the pre-registered family |
| `audit_liveness.py` | the standalone behavioural liveness audit (42-cell sweep) |
| `embed.py` | frozen ESM-2 embedding computation for GB1 (mean and site pooling) |
| `make_figures.py` | generates the three released figures from the results/audit files below |

## Data artifacts

| file | contents |
|---|---|
| `pools/free_llm.json` | 20 free-form LLM-generated policies (original generator, provenance disclosed in §5) |
| `pools/full_compositional.json` | 20 structured policies across perturb/project/recombine/free channels |
| `pools/replication_sonnet5.json` | independent replication pool, `claude-sonnet-5`, generated cold under the same protocol (§5) |
| `liveness_audit.json` | the 42-cell behavioural liveness audit result |
| `pool_results_s24.json` | GB1 run of record, 24 seeds, build `255f291a55b1ac42` |
| `pool_results_TEV_s24_b96i24x24.json` | TEV run, 24 seeds, standard operating point |
| `pool_results_TrpB4_s24_b96i24x24.json` | TrpB4 run, 24 seeds, standard operating point |
| `pool_results_s8_b480i96x96.json`, `pool_results_s24_b480i96x96.json` | the budget-480 confirmation sweeps |
| `frozen_build_*/` | source snapshots for every build hash quoted in the paper, so any number can be traced to the exact code that produced it |

## Not released (and why)

- `data/gb1_esm2_35M.npz` (686 MB) — the frozen ESM-2 embedding cache. Regenerable in ~7 minutes from
  `embed.py` on the published eLife GB1 dataset; too large to bundle, not required to verify results, only to
  regenerate the embedding coordinate from scratch.
- `data/gb1_elife.xlsx` — the source GB1 dataset, itself a public supplement of the cited eLife paper. Point to
  the original publication rather than redistribute a copy.
- `contaminated_build_archive/` — deliberately excluded from any release framing as data; kept locally as a
  record of what must never be quoted (see `PAPER_DISCOVERY_POLICY.md` §6.1).

## Reproducing a headline number

```
cd protein/
python3 analyze.py pool_results_s24.json        # §4.1-4.5, GB1
python3 analyze.py pool_results_TEV_s24_b96i24x24.json
python3 analyze.py pool_results_TrpB4_s24_b96i24x24.json
python3 audit_liveness.py                        # the 42-cell liveness table (§3.1)
python3 make_figures.py                          # regenerates figures/f1-f3
```

Re-running the campaign itself from a frozen build (rather than only re-analysing its stored results):

```
cd protein/frozen_build_255f291a55b1ac42/
ln -s ../data data && ln -s ../pools pools     # or copy
N_SEEDS=24 python3 run_pools.py
```

RNG streams are seeded deterministically by `(seed, subsystem, round)`, not by wall-clock, so this reproduces
the stored results bit-for-bit.
