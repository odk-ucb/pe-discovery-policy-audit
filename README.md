# Anonymous release bundle (ICBINB-BIO companion)

Prepared for double-blind review. MIT code (see LICENSE). Landscape source data are public third-party tables (GB1 eLife; SSMuLA TEV/TrpB4) — not redistributed here.

## Reproduce headline analyses
```bash
python3 analyze.py pool_results_GB1_s24_b96i24x24.json
python3 analyze.py pool_results_TEV_s24_b96i24x24.json
python3 analyze.py pool_results_TrpB4_s24_b96i24x24.json
python3 audit_liveness.py
python3 make_figures.py
```

See RELEASE_MANIFEST.md, DEFAULT_PROVENANCE.md, GENERATOR_PROVENANCE.md.

Frozen full build trees and ESM caches are large; pointers listed under frozen_build_pointers/. Contaminated RNG archive is intentionally omitted.
