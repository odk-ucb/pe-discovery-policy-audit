"""Frozen protein-language-model embeddings for the whole GB1 candidate universe.

WHY THIS EXISTS. `representation.encoding` is a coordinate the protein-engineering literature actually varies — ALDE's
headline analysis compares encodings, and SAMPLE/EVOLVEpro-style policies use frozen PLM features. Until now this
project's `learned_embedding_frozen` value was PCA of a one-hot, which is not a learned embedding at all and would have
made any finding about that coordinate meaningless. This computes the real thing once and caches it, so the coordinate
becomes a genuine alternative rather than a placeholder.

TWO POOLINGS, BECAUSE POOLING IS ITSELF A COORDINATE, not an implementation detail:
    mean  — mean over residues, 480 dims. What a generic sequence-embedding pipeline produces.
    site  — the four mutated positions' residue embeddings concatenated, 1920 dims. What a site-specific method uses.
A policy that varies only pooling and nothing else is a legitimate one-coordinate perturbation, and the two are
measurably different objects rather than two names for the same vector.

NO FITNESS IS USED ANYWHERE HERE. The embedding is unsupervised and computed for the entire universe before any campaign
runs, so it cannot leak an outcome; and because it is computed for *every* variant, it cannot leak the identity of the
ones a policy will later query.

WHY NOT A GPU. Measured, not assumed: ESM-2 35M on local MPS runs at 378 sequences/second at batch 256, so the full
149,361-variant universe takes 6.6 minutes. A rented GPU would save minutes on a one-off cache. The case for escalating
to RunPod is ESM-2 650M — 18x the parameters, roughly two hours locally against ~10 minutes on an A100 — and that is
worth paying for only if this 35M embedding turns out to be a *decisive* coordinate, at which point embedding scale
becomes a coordinate worth probing in its own right. Escalate against a measured need.
"""
import os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

from landscape import Landscape, WT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "gb1_esm2_35M.npz")
MODEL = "facebook/esm2_t12_35M_UR50D"
# GB1 domain B1, 56 residues. The four combinatorial sites are V39 D40 G41 V54 in 1-indexed GB1 numbering.
WTSEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POS = [38, 39, 40, 53]                      # 0-indexed
BATCH = 256

assert len(WTSEQ) == 56
assert "".join(WTSEQ[p] for p in POS) == WT, "site positions do not reproduce the wild-type 4-mer"


def full_sequences(variants):
    """Substitute each variant's four residues into the wild-type backbone."""
    out = []
    for v in variants:
        s = list(WTSEQ)
        for k, p in enumerate(POS):
            s[p] = v[k]
        out.append("".join(s))
    return out


def main():
    L = Landscape()
    vs = L.v
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(dev).eval()
    D = mdl.config.hidden_size
    print(f"embedding {len(vs):,} variants with {MODEL} on {dev}; hidden size {D}")
    mean = np.zeros((len(vs), D), dtype=np.float16)
    site = np.zeros((len(vs), 4 * D), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(vs), BATCH):
            chunk = vs[i:i + BATCH]
            b = tok(full_sequences(chunk), return_tensors="pt", padding=True)
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev)
            h = mdl(input_ids=ii, attention_mask=am).last_hidden_state       # (B, T, D)
            # ESM prepends <cls>, so residue r of the protein is token r+1. Exclude special tokens from the mean.
            m = am.clone(); m[:, 0] = 0
            last = am.sum(1) - 1
            for r in range(len(chunk)): m[r, last[r]] = 0
            mm = m.unsqueeze(-1).float()
            mean[i:i + len(chunk)] = ((h * mm).sum(1) / mm.sum(1)).cpu().numpy().astype(np.float16)
            sidx = torch.tensor([p + 1 for p in POS], device=dev)
            site[i:i + len(chunk)] = h[:, sidx, :].reshape(len(chunk), -1).cpu().numpy().astype(np.float16)
            if (i // BATCH) % 60 == 0:
                done = i + len(chunk)
                print(f"   {done:>7,}/{len(vs):,}  {done/(time.time()-t0):7.1f} seq/s  "
                      f"eta {(len(vs)-done)/max(done/(time.time()-t0),1)/60:5.1f} min")
    np.savez(OUT, variants=np.array(vs, dtype="U4"), mean=mean, site=site, model=np.array([MODEL]))
    print(f"\nsaved -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB) in {(time.time()-t0)/60:.1f} min")
    # a sanity check that the embedding is not degenerate: WT must not be equidistant from everything
    d = np.linalg.norm(site.astype(np.float32) - site[L.idx[WT]].astype(np.float32), axis=1)
    print(f"   site-embedding distance from WT: median {np.median(d):.3f}, min {d.min():.3f}, max {d.max():.3f}")
    print(f"   distinct rows (sampled 5000): {len(np.unique(site[::30][:5000], axis=0)):,} of 5,000")


if __name__ == "__main__":
    main()
