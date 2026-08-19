#!/usr/bin/env python3
"""ESM-2 masked-marginal zero-shot scores for TEV / TrpB4 (and GB1 if needed).

Matches campaign._zs() cache format: npz with variants (U4) and zs (float).
Efficient: 4 masked WT forward passes (one per site) → score every combinatorial variant
as sum of log-probs of its amino acids at the four sites.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL = "facebook/esm2_t12_35M_UR50D"
AA = "ACDEFGHIKLMNPQRSTVWY"

LANDSCAPES = {
    "GB1": dict(
        wtseq="MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        pos_1idx=[39, 40, 41, 54],  # 1-indexed GB1 numbering used in embed.py as 0-idx 38..
        # embed.py uses POS = [38,39,40,53] 0-indexed
        pos0=[38, 39, 40, 53],
        out="gb1_zeroshot_esm2.npz",
        loader="gb1",
    ),
    "TEV": dict(
        wtseq=("GESLFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSLHGVFKVKNTTTLQQHLIDGRDMIIIRMPKDFPPFPQKL"
               "KFREPQREERICLVTTNFQTKSMSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFTSVPKNFMELLTNQEAQ"
               "QWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQLMN"),
        pos_1idx=[146, 148, 167, 170],
        out="tev_zeroshot_esm2.npz",
        loader="ssmula",
    ),
    "TrpB4": dict(
        wtseq=("MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMG"
               "KTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPII"
               "VRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHS"
               "VSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRL"),
        pos_1idx=[183, 184, 227, 228],
        out="trpb4_zeroshot_esm2.npz",
        loader="ssmula",
    ),
}


def variants_for(name, cfg):
    if cfg["loader"] == "gb1":
        from landscape import Landscape
        return list(Landscape().v)
    import ssmula_landscape as SSM
    return list(SSM.SSMuLALandscape(name).v)


def run(name: str):
    cfg = LANDSCAPES[name]
    pos0 = cfg.get("pos0") or [p - 1 for p in cfg["pos_1idx"]]
    wt = cfg["wtseq"]
    vs = variants_for(name, cfg)
    # WT check
    wt4 = "".join(wt[p] for p in pos0)
    print(f"{name}: n={len(vs):,} wt_sites={wt4} pos0={pos0} len_wt={len(wt)}")

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    # map aa -> id
    aa_id = {a: tok.convert_tokens_to_ids(a) for a in AA}
    mask_id = tok.mask_token_id

    # For each site: mask WT at that position, get log-probs over AA
    site_logp = []  # list of dict aa->logp
    t0 = time.time()
    with torch.no_grad():
        for si, p in enumerate(pos0):
            seq = list(wt)
            seq[p] = tok.mask_token if hasattr(tok, "mask_token") and tok.mask_token else "X"
            # ESM uses <mask>
            s = "".join(seq)
            # Better: tokenize WT and set input_ids at p+1 (CLS)
            enc = tok(wt, return_tensors="pt")
            ids = enc["input_ids"].clone()
            # position in token space: CLS at 0, residue i at i+1
            ids[0, p + 1] = mask_id
            ids = ids.to(dev)
            am = enc["attention_mask"].to(dev)
            logits = mdl(input_ids=ids, attention_mask=am).logits[0, p + 1]  # vocab
            # log softmax over 20 AA
            aa_ids = torch.tensor([aa_id[a] for a in AA], device=dev)
            lp = torch.log_softmax(logits[aa_ids], dim=0).cpu().numpy()
            site_logp.append({a: float(lp[j]) for j, a in enumerate(AA)})
            print(f"  site {si} pos {p} done", flush=True)

    zs = np.zeros(len(vs), dtype=np.float32)
    for i, v in enumerate(vs):
        s = 0.0
        for si, a in enumerate(v):
            s += site_logp[si].get(a, -50.0)
        zs[i] = s
    out = os.path.join(DATA, cfg["out"])
    np.savez(out, variants=np.array(vs, dtype="U4"), zs=zs, model=np.array([MODEL]))
    print(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB) in {(time.time()-t0)/60:.2f} min")
    print(f"  zs mean={zs.mean():.3f} std={zs.std():.3f} min={zs.min():.3f} max={zs.max():.3f}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["TEV", "TrpB4"]
    for t in targets:
        run(t)
