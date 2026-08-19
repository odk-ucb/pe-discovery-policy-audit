"""Frozen ESM-2 embeddings for TEV and TrpB4, generalising `embed.py`'s GB1 pipeline.

Why this exists: `campaign.py`'s `_esm()`/`_zs()` previously raised a named error for any landscape other
than GB1, because building a PLM cache for TEV/TrpB4 required their wild-type backbone sequences, which were
not on hand (discarded during an earlier disk-space emergency, iteration 88). Re-acquired from SSMuLA's own
FASTA files (Zenodo doi:10.5281/zenodo.13910506) and verified against the SSMuLA CSVs before use: the wild-type
residue at every declared site position must match the CSV's own wild-type row, or the sequence is wrong.

    TEV   236 aa, sites 146/148/167/170 (1-indexed) = T/D/H/S  -- verified
    TrpB4 390 aa, sites 183/184/227/228 (1-indexed) = V/F/V/S  -- verified

Same two poolings as GB1 (mean over the sequence; concatenation of the four mutated sites), same model
(facebook/esm2_t12_35M_UR50D), same "no GPU needed" reasoning: ~160K sequences at the GB1-measured
378 seq/s rate is under 8 minutes per landscape on local MPS.
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = "facebook/esm2_t12_35M_UR50D"
BATCH = 256

LANDSCAPES = {
    "TEV": dict(
        wtseq="GESLFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSLHGVFKVKNTTTLQQHLIDGRDMIIIRMPKDFPPFPQKL"
              "KFREPQREERICLVTTNFQTKSMSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFTSVPKNFMELLTNQEAQ"
              "QWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQLMN",
        pos_1idx=[146, 148, 167, 170],
        out="tev_esm2_35M.npz",
    ),
    "TrpB4": dict(
        wtseq="MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMG"
              "KTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPII"
              "VRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHS"
              "VSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRL",
        pos_1idx=[183, 184, 227, 228],
        out="trpb4_esm2_35M.npz",
    ),
}


def full_sequences(variants, wtseq, pos0):
    out = []
    for v in variants:
        s = list(wtseq)
        for k, p in enumerate(pos0):
            s[p] = v[k]
        out.append("".join(s))
    return out


def run(name):
    import ssmula_landscape as SSM
    cfg = LANDSCAPES[name]
    pos0 = [p - 1 for p in cfg["pos_1idx"]]
    wtseq = cfg["wtseq"]

    lsc = SSM.SSMuLALandscape(name)
    wt_ok = "".join(wtseq[p] for p in pos0) == lsc.wt if lsc.wt else None
    print(f"{name}: {len(lsc.v):,} variants; wild-type site check "
          f"{'PASS' if wt_ok else ('no WT row in CSV -- skipped' if wt_ok is None else 'FAIL')}")
    if wt_ok is False:
        raise SystemExit(f"{name}: wild-type residues at declared positions do not match the CSV's WT row")

    vs = lsc.v
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()
    D = mdl.config.hidden_size
    print(f"embedding {len(vs):,} {name} variants with {MODEL} on {dev}; hidden size {D}")
    mean = np.zeros((len(vs), D), dtype=np.float16)
    site = np.zeros((len(vs), 4 * D), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(vs), BATCH):
            chunk = vs[i:i + BATCH]
            b = tok(full_sequences(chunk, wtseq, pos0), return_tensors="pt", padding=True)
            ii = b["input_ids"].to(dev)
            am = b["attention_mask"].to(dev)
            h = mdl(input_ids=ii, attention_mask=am).last_hidden_state
            m = am.clone(); m[:, 0] = 0
            last = am.sum(1) - 1
            for r in range(len(chunk)):
                m[r, last[r]] = 0
            mm = m.unsqueeze(-1).float()
            mean[i:i + len(chunk)] = ((h * mm).sum(1) / mm.sum(1)).cpu().numpy().astype(np.float16)
            sidx = torch.tensor([p + 1 for p in pos0], device=dev)
            site[i:i + len(chunk)] = h[:, sidx, :].reshape(len(chunk), -1).cpu().numpy().astype(np.float16)
            if (i // BATCH) % 60 == 0:
                done = i + len(chunk)
                rate = done / (time.time() - t0 + 1e-9)
                print(f"   {done:>7,}/{len(vs):,}  {rate:7.1f} seq/s  eta {(len(vs)-done)/max(rate,1)/60:5.1f} min")

    out_path = os.path.join(DATA, cfg["out"])
    np.savez(out_path, variants=np.array(vs, dtype="U4"), mean=mean, site=site, model=np.array([MODEL]))
    print(f"saved -> {out_path}  ({os.path.getsize(out_path)/1e6:.0f} MB) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(LANDSCAPES)
    for t in targets:
        run(t)
