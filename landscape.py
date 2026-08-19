"""The objective environment: the GB1 four-site combinatorially (near-)complete fitness landscape.

WHY THIS LANDSCAPE. A discovery policy decides which variants to assay next, so evaluating one honestly requires an
environment in which ANY variant the policy asks for can be answered exactly. That rules out held-out-split benchmarks
and rules in a combinatorially complete landscape. Wu, Dai, Olson, Lloyd-Smith & Sun (eLife 2016,
doi:10.7554/eLife.16965) measured fitness for 149,361 of the 20^4 = 160,000 variants at sites V39, D40, G41, V54 of
protein G domain B1 — an epistatic region, which is the point: a landscape without epistasis would make every policy
look alike. It is the standard substrate for machine-learning-assisted directed evolution, including the ALDE study the
pivot brief cites.

THE ORACLE IS A LOOKUP, WHICH IS THE WHOLE ADVANTAGE. No surrogate stands between a policy and its reward, so a
campaign is simulated exactly and the only stochasticity is the policy's own seed. Every oracle call is counted, and the
budget is asserted rather than assumed — ten separate matched-budget accounting errors in this project's previous
experiment all came from comparing at budgets that were nominally but not actually equal.

MEASURE THE INSTRUMENT BEFORE USING IT. Two distinct noise sources, and conflating them is how a previous state in this
project produced a menu whose every option sat inside its own error bars:
  1. **Assay noise.** Fitness here is a ratio of selected to input read counts, so a variant seen a handful of times has
     an unreliable value. `noise_report()` quantifies this and reports how much of the apparent top of the landscape is
     low-count. The primary metric is defined on a count-filtered oracle; the unfiltered version is reported beside it.
  2. **Campaign noise.** Seed-to-seed spread of the SAME policy. That belongs to the harness, not here, and must be
     measured before any two policies are compared.

NO-OP ANCHOR. Every campaign reports the best fitness already present in its initial observations. A policy that does
not beat that has discovered nothing, and the absence of this anchor is exactly what let an earlier state's admission
failure pass unnoticed.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
XLSX = os.path.join(DATA, "gb1_elife.xlsx")
CACHE = os.path.join(DATA, "gb1_landscape.npz")
SITES = ("V39", "D40", "G41", "V54")
WT = "VDGV"
AAS = "ACDEFGHIKLMNPQRSTVWY"
# RELIABILITY. The first version of this filter used `input count >= 10` and passed all 149,361 variants — the
# published dataset is ALREADY filtered on input count, so that was not a filter at all. The noise is driven by the
# SELECTED count: 29,485 variants have 0 selected reads and 19,574 have exactly 1, giving a median relative standard
# error of 0.507 across the landscape. Reliability is therefore defined on the count ratio's own relative SE.
MAX_RSE = 0.25          # declared before any policy is run; see noise_report()


def _build():
    import pandas as pd
    df = pd.ExcelFile(XLSX).parse("Supplementary Dataset 1")
    df = df.rename(columns={"Variants": "v", "HD": "hd", "Count input": "cin",
                            "Count selected": "csel", "Fitness": "f"})
    df = df[df["v"].astype(str).str.len() == 4].copy()
    np.savez_compressed(CACHE, v=df["v"].to_numpy().astype("U4"), hd=df["hd"].to_numpy().astype(np.int8),
                        cin=df["cin"].to_numpy().astype(np.int64), csel=df["csel"].to_numpy().astype(np.int64),
                        f=df["f"].to_numpy().astype(np.float64))
    return CACHE


class Landscape:
    def __init__(self, max_rse=MAX_RSE):
        if not os.path.exists(CACHE):
            _build()
        z = np.load(CACHE, allow_pickle=False)
        self.v = [str(x) for x in z["v"]]
        self.hd, self.cin, self.csel, self.f = z["hd"], z["cin"], z["csel"], z["f"]
        self.idx = {s: i for i, s in enumerate(self.v)}
        self.max_rse = max_rse
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rse = np.sqrt(1.0 / np.maximum(self.cin, 1) + 1.0 / np.maximum(self.csel, 1))
        self.reliable = (self.rse <= max_rse)
        self.calls = 0
        self._called = set()

    # ---------------- the oracle
    def query(self, variant):
        """One assay. Counted. Returns fitness, or None if this variant was never measured."""
        i = self.idx.get(variant)
        if i is None: return None
        self.calls += 1
        self._called.add(variant)
        return float(self.f[i])

    def query_batch(self, variants):
        return [self.query(v) for v in variants]

    def reset_counter(self):
        self.calls = 0
        self._called = set()

    # ---------------- ground truth, for scoring only — never visible to a policy
    def true_best(self, reliable_only=True):
        m = self.reliable if reliable_only else np.ones(len(self.f), bool)
        j = int(np.argmax(np.where(m, self.f, -np.inf)))
        return self.v[j], float(self.f[j])

    def all_variants(self):
        return list(self.v)

    def reliable_variants(self):
        return [s for s, r in zip(self.v, self.reliable) if r]

    def hit_threshold(self, q=0.99, over="reliable"):
        """'Active hit' = fitness above the q-quantile of the universe ACTUALLY SEARCHED. Declared, not tuned.

        This previously always used the reliable subset while campaigns searched all variants, so a threshold
        reported as "top-1%" selected the top 0.23% of the executed universe. The quantile must be taken over
        the same set the campaign draws from or the metric's name is false. `over` must match the harness's
        universe argument."""
        f = self.f[self.reliable] if over == "reliable" else self.f
        return float(np.quantile(f, q))

    def hit_fraction(self, thr, over="reliable"):
        """Realised fraction of the executed universe at or above thr — the metric's true selectivity."""
        f = self.f[self.reliable] if over == "reliable" else self.f
        return float((f >= thr).mean())

    # ---------------- the instrument's own noise
    def noise_report(self):
        n = len(self.f)
        cov = n / (20 ** 4)
        top = np.argsort(-self.f)[:200]
        top_low = int((self.rse[top] > self.max_rse).sum())
        rse = self.rse
        rel = self.reliable
        return dict(
            n_measured=n, coverage=cov, n_reliable=int(rel.sum()),
            max_rse=self.max_rse, median_rse_top200=float(np.median(rse[top])),
            zero_selected=int((self.csel == 0).sum()), one_selected=int((self.csel == 1).sum()),
            wt_fitness=float(self.f[self.idx[WT]]),
            best_unfiltered=self.true_best(False), best_reliable=self.true_best(True),
            median_rse_all=float(np.median(rse)), median_rse_reliable=float(np.median(rse[rel])),
            top200_low_count=top_low,
            hit_threshold_q99=self.hit_threshold(0.99),
            frac_above_wt=float((self.f > self.f[self.idx[WT]]).mean()),
            frac_above_wt_reliable=float((self.f[rel] > self.f[self.idx[WT]]).mean()),
        )


if __name__ == "__main__":
    L = Landscape()
    r = L.noise_report()
    print("=" * 96)
    print("GB1 four-site landscape — the objective environment")
    print("=" * 96)
    print(f"   sites {', '.join(SITES)}; wild type {WT} (fitness {r['wt_fitness']:.4f})")
    print(f"   measured variants        {r['n_measured']:,} of 160,000  ({r['coverage']:.1%} of the space)")
    print(f"   reliable (relative SE <= {r['max_rse']})   {r['n_reliable']:,}  ({r['n_reliable']/r['n_measured']:.1%})")
    print()
    print("   ASSAY NOISE, measured before any policy is compared")
    print(f"      median relative SE, all variants      {r['median_rse_all']:.4f}")
    print(f"      median relative SE, reliable subset   {r['median_rse_reliable']:.4f}")
    print(f"      of the 200 highest-fitness variants, {r['top200_low_count']} are BELOW the count threshold")
    print(f"      -> that is why the primary oracle is count-filtered; an unfiltered 'best found' rewards read-count")
    print(f"         flukes, and a policy that chases them would look good for no biological reason.")
    print()
    print(f"   best variant, unfiltered   {r['best_unfiltered'][0]} at {r['best_unfiltered'][1]:.4f}")
    print(f"   best variant, reliable     {r['best_reliable'][0]} at {r['best_reliable'][1]:.4f}")
    print(f"   fraction fitter than WT    {r['frac_above_wt']:.4f} all / {r['frac_above_wt_reliable']:.4f} reliable")
    print(f"   active-hit threshold (99th pct of reliable)  {r['hit_threshold_q99']:.4f}")
    json.dump(r, open(os.path.join(DATA, "landscape_report.json"), "w"), indent=1, default=str)
    print("\nsaved -> data/landscape_report.json")
