"""Additional four-site combinatorial landscapes, loaded from SSMuLA.

The experiment ran on GB1 alone. Every close comparator runs on more: SILO 8, SSMuLA 16 (12 in main text),
ALDE 2, CLADE 2, BO-EVO 2 empirical + 4 NK, EVOLVEpro 12 DMS. One landscape cannot distinguish "the design
space does the work" from "the design space does the work ON GB1", and the coordinate-choice equivalence
result is exactly the kind of claim that could be landscape-specific.

Source: SSMuLA, Li et al., *Evaluation of Machine Learning-Assisted Directed Evolution Across Diverse
Combinatorial Landscapes*, Cell Systems 2025. Data: Zenodo doi:10.5281/zenodo.13910506, `data.zip`.
Code there is GPL-3.0; we take only the processed fitness tables and write our own loader, so no GPL code
enters this implementation.

Three landscapes share GB1's four-site structure exactly, so the existing harness applies unchanged:

    GB1    149,361 measured variants   sites V39 D40 G41 V54
    TrpB4  193,170 rows                sites V183 F184 V227 S228
    TEV    159,132 rows                sites T146 D148 H167 S170

SSMuLA's own GB1 table is used to cross-check our independently-built GB1 landscape; the two must agree on
ranking or one of them is wrong.

This class presents the same surface `campaign.Harness` uses: `f`, `idx`, `reliable`, `query`, `all_variants`,
`reliable_variants`, `true_best`, `hit_threshold`, `hit_fraction`.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SSMULA = os.path.join(HERE, "data", "ssmula")
SITES = 4

# Stop codons are not assayable protein variants; they are excluded rather than scored as dead, because a
# campaign cannot order them. TrpB4 is the only table here that contains them.
STOP = "*"


class SSMuLALandscape:
    def __init__(self, name, drop_stops=True, scale_to_wt=True):
        path = os.path.join(SSMULA, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} -- extract data.zip from Zenodo 13910506 into data/ssmula/")
        df = pd.read_csv(path)
        if drop_stops:
            df = df[~df["AAs"].str.contains(r"\*", regex=True, na=False)]
        df = df.dropna(subset=["fitness"]).reset_index(drop=True)

        self.name = name
        self.v = [str(x) for x in df["AAs"].values]
        self.idx = {s: i for i, s in enumerate(self.v)}
        f = df["fitness"].values.astype(float)

        # SSMuLA ships `scale2max` tables (max fitness = 1). Our GB1 convention is fitness relative to
        # wild type, so rescale to WT when a WT row is identifiable, and record which convention is in use.
        self.wt = None
        if "muts" in df.columns:
            wt_rows = df.index[df["muts"].astype(str) == "WT"]
            if len(wt_rows):
                self.wt = self.v[int(wt_rows[0])]
        if scale_to_wt and self.wt is not None and f[self.idx[self.wt]] > 0:
            f = f / f[self.idx[self.wt]]
            self.scale = "relative_to_wild_type"
        else:
            self.scale = "as_published_scale2max"
        self.f = f

        # SSMuLA publishes an `active` flag rather than per-variant read counts, so the RSE-based reliability
        # filter used for GB1 has no analogue here. Everything measured is treated as reliable and this is
        # recorded, not hidden: it means `reliable_only` comparisons are not equivalent across landscapes.
        self.reliable = np.ones(len(self.f), dtype=bool)
        self.active = (df["active"].values.astype(bool) if "active" in df.columns
                       else np.ones(len(self.f), dtype=bool))
        self.reliability_basis = "none_available_all_measured_treated_as_reliable"

        self.calls = 0
        self._called = set()

    # ---- the surface campaign.Harness uses
    def query(self, variant):
        i = self.idx.get(variant)
        if i is None:
            raise KeyError(variant)
        if variant not in self._called:
            self._called.add(variant)
            self.calls += 1
        return float(self.f[i])

    def query_batch(self, variants):
        return [self.query(v) for v in variants]

    def reset_counter(self):
        self.calls = 0
        self._called = set()

    def true_best(self, reliable_only=True):
        i = int(np.argmax(self.f))
        return self.v[i], float(self.f[i])

    def all_variants(self):
        return list(self.v)

    def reliable_variants(self):
        return list(self.v)

    def hit_threshold(self, q=0.99, over="all"):
        return float(np.quantile(self.f, q))

    def hit_fraction(self, thr, over="all"):
        return float((self.f >= thr).mean())

    def report(self):
        b, bf = self.true_best()
        return dict(name=self.name, n=len(self.f), scale=self.scale, wild_type=self.wt,
                    wt_fitness=float(self.f[self.idx[self.wt]]) if self.wt else None,
                    optimum=b, optimum_fitness=round(bf, 4),
                    frac_above_wt=round(float((self.f > 1.0).mean()), 5) if self.scale.startswith("relative")
                    else None,
                    frac_active=round(float(self.active.mean()), 5),
                    reliability_basis=self.reliability_basis)


def available():
    if not os.path.isdir(SSMULA):
        return []
    return sorted(x[:-4] for x in os.listdir(SSMULA) if x.endswith(".csv"))


if __name__ == "__main__":
    import json
    for n in available():
        try:
            print(json.dumps(SSMuLALandscape(n).report()))
        except Exception as e:
            print(f"{n}: {type(e).__name__}: {e}")
