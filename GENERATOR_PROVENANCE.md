# Generator provenance (what is logged and what is not)

## Original free / compositional pools

| Field | Value |
|---|---|
| Files | `pools/free_llm.json` (20 free), `pools/full_compositional.json` (20 structured: perturb/project/recombine/free) |
| Project records | “Opus subagents” / Claude Opus family for both original pools |
| API model string | **Not retained** in the JSON (no `model_id`, temperature, or prompt hash fields) |
| Recoverable content | policy `name`, `rationale`/`derivation`, full `coords`, channel labels |

**Honest paper wording:** main generator = one incompletely versioned Claude Opus checkpoint; prompts/temperature not retained. Findings about “the model” are about this generator instance class, not LMs in general.

## Independent replication pool

| Field | Value |
|---|---|
| File | `pools/replication_sonnet5.json` |
| `generator_identity` | `claude-sonnet-5` |
| `generation_date` | `2026-08-18` |
| Protocol | cold: coordinate menu + problem framing only; no access to project results |
| Contents | 20 free + 5 perturb off its stated base (+ recorded `base_policy`) |

## Implications

- Free-proposal null replicates across the two generator *families* as logged.
- Perturb-channel divergence is reported, not smoothed (small n=5 policies/channel).
- Re-issuing original pools through a fully logged API client remains the fix for confirmatory generation reproducibility.

## Contaminated archive

`contaminated_build_archive/` is excluded from release framing; do not quote its numbers.
