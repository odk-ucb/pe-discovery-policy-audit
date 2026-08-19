"""Generate the paper's figures. Read-only against results/audit files; writes only to ../figures/.

Three figures, matching what the blind Sonnet review (2026-08-18) and the ICLR objection-transfer audit
both named as the highest-value additions:
  F1  policy-minus-base distribution by arm  -- the paper's real finding (asymmetric around the default)
  F2  the behavioural liveness map            -- which of the 42 cells are LIVE / INERT / INAPPLICABLE
  F3  per-seed paired differences, perturb vs base -- the primitive the whole Holm family is built on
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 150})


def load(fn):
    d = json.load(open(os.path.join(HERE, fn)))
    d.pop("__build__", None)
    return d


# ---------------------------------------------------------------- F2: liveness map (no dependency on the
# in-flight GB1 restoration; build this first)
def fig_liveness_map():
    rows = json.load(open(os.path.join(HERE, "liveness_audit.json")))
    coords = sorted({r["coordinate"] for r in rows})
    by_coord = {c: [] for c in coords}
    for r in rows:
        by_coord[r["coordinate"]].append(r)
    maxcells = max(len(v) for v in by_coord.values())
    color = {"LIVE": "#2a7f3f", "INERT": "#b0b0b0", "INAPPLICABLE": "#e8e8e8", "FAILS": "#c0392b"}

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(coords) + 1.2))
    for i, c in enumerate(coords):
        cells = by_coord[c]
        for j, cell in enumerate(cells):
            ax.add_patch(plt.Rectangle((j, len(coords) - 1 - i), 0.92, 0.85,
                                        facecolor=color.get(cell["verdict"], "#fff"), edgecolor="white"))
        for j in range(len(cells), maxcells):
            pass
    ax.set_xlim(0, maxcells)
    ax.set_ylim(0, len(coords))
    ax.set_yticks([len(coords) - 1 - i + 0.4 for i in range(len(coords))])
    ax.set_yticklabels(coords, fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("alternative values tested for this coordinate (one cell per value)")
    ax.set_title("Behavioural liveness audit — 42 cells, varied one at a time from a fixed base",
                  fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=v) for v in color.values()]
    ax.legend(handles, list(color.keys()), loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "f2_liveness_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_policy_minus_base(results_file, tag):
    d = load(results_file)
    base = d.get("SEED:base_onehot_ridge")
    if base is None or not base.get("seeds"):
        print(f"skip F1/F3 ({tag}): base policy not present/complete in {results_file}")
        return
    base_ids = base.get("seed_ids") or list(range(len(base["seeds"])))
    base_map = dict(zip(base_ids, base["seeds"]))

    diffs_by_channel = {}
    for k, v in d.items():
        if k == "SEED:base_onehot_ridge" or not v.get("seeds"):
            continue
        ids = v.get("seed_ids") or list(range(len(v["seeds"])))
        common = [s for s in ids if s in base_map]
        if len(common) < 3:
            continue
        vmap = dict(zip(ids, v["seeds"]))
        d_mean = float(np.mean([vmap[s] - base_map[s] for s in common]))
        diffs_by_channel.setdefault(v["channel"], []).append(d_mean)

    order = sorted(diffs_by_channel, key=lambda c: -np.mean(diffs_by_channel[c]))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.5))
    for i, ch in enumerate(order):
        vals = diffs_by_channel[ch]
        ax.scatter(vals, [i] * len(vals), alpha=0.7, s=28,
                   color="#c0392b" if np.mean(vals) < 0 else "#2a7f3f")
        ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color="black", lw=1.5)
    ax.axvline(0, color="gray", lw=1, linestyle="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("policy mean − base-policy mean (paired on shared seeds)")
    ax.set_title(f"Every executed policy vs. the hand-written default, by channel ({tag})", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"f1_policy_minus_base_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    n = sum(len(v) for v in diffs_by_channel.values())
    up = sum(1 for vs in diffs_by_channel.values() for v in vs if v > 0)
    print(f"wrote {out}  ({tag}: n={n}, {up} up / {n-up} down or zero)")


def fig_paired_seed_diffs(results_file, tag):
    d = load(results_file)
    base = d.get("SEED:base_onehot_ridge")
    perturb = {k: v for k, v in d.items() if v.get("channel") == "perturb" and v.get("seeds")}
    if base is None or not base.get("seeds") or not perturb:
        print(f"skip F3 ({tag}): base or perturb arm not present/complete")
        return
    base_ids = base.get("seed_ids") or list(range(len(base["seeds"])))
    base_map = dict(zip(base_ids, base["seeds"]))
    common = sorted(base_ids)
    per_seed = {s: [] for s in common}
    for k, v in perturb.items():
        ids = v.get("seed_ids") or list(range(len(v["seeds"])))
        vmap = dict(zip(ids, v["seeds"]))
        for s in common:
            if s in vmap:
                per_seed[s].append(vmap[s] - base_map[s])
    seeds = [s for s in common if per_seed[s]]
    means = [float(np.mean(per_seed[s])) for s in seeds]

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#2a7f3f" if m > 0 else "#c0392b" for m in means]
    ax.bar(range(len(seeds)), means, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("seed")
    ax.set_ylabel("mean(perturb) − base, that seed")
    ax.set_title(f"Per-seed paired difference, perturb channel vs. the default ({tag})", fontsize=10)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7, rotation=90)
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"f3_paired_seed_diffs_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    wins = sum(1 for m in means if m > 0)
    print(f"wrote {out}  ({tag}: {wins}/{len(means)} seeds favour perturb)")


def fig_decomposition_gb1():
    """Bar decomposition: perturb-free = (perturb-base) + (base-free). Uses analyze-free arithmetic on stored seeds."""
    # Prefer run-of-record GB1 build; fall back to older s24 file.
    for fn in ("pool_results_GB1_s24_b96i24x24.json", "pool_results_s24.json"):
        path = os.path.join(HERE, fn)
        if os.path.exists(path):
            break
    else:
        print("skip decomposition: no GB1 results")
        return
    d = load(fn)
    base = d.get("SEED:base_onehot_ridge")
    if not base or not base.get("seeds"):
        print("skip decomposition: missing base")
        return
    base_ids = base.get("seed_ids") or list(range(len(base["seeds"])))
    base_map = dict(zip(base_ids, base["seeds"]))

    def channel_seed_means(channel):
        """Per-seed mean over policies in channel, aligned to base seed ids."""
        pols = [v for k, v in d.items() if v.get("channel") == channel and v.get("seeds")]
        if not pols:
            return None
        out = []
        for s in base_ids:
            vals = []
            for v in pols:
                ids = v.get("seed_ids") or list(range(len(v["seeds"])))
                vmap = dict(zip(ids, v["seeds"]))
                if s in vmap:
                    vals.append(vmap[s])
            if vals:
                out.append(float(np.mean(vals)))
            else:
                out.append(float("nan"))
        return np.array(out)

    b = np.array([base_map[s] for s in base_ids], dtype=float)
    p = channel_seed_means("perturb")
    f = channel_seed_means("free")
    if p is None or f is None:
        print("skip decomposition: missing perturb or free channel")
        return
    m = np.isfinite(p) & np.isfinite(f) & np.isfinite(b)
    pb = float(np.mean(p[m] - b[m]))
    bf = float(np.mean(b[m] - f[m]))
    pf = float(np.mean(p[m] - f[m]))
    share = 100.0 * bf / pf if abs(pf) > 1e-12 else float("nan")

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    labels = ["perturb − base\n(model move)", "base − free\n(default vs free LLM)", "perturb − free\n(composite)"]
    vals = [pb, bf, pf]
    colors = ["#7f8c8d", "#2a7f3f", "#1a5276"]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("paired mean gain difference")
    ax.set_title(f"GB1 decomposition: {share:.1f}% of (perturb−free) is (base−free)", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "f4_decomposition_GB1.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  pb={pb:+.4f} bf={bf:+.4f} pf={pf:+.4f} share={share:.1f}%")


if __name__ == "__main__":
    fig_liveness_map()
    # Run-of-record GB1 first (build 594e…); keep legacy filename as fallback inside helpers via explicit list.
    for fn, tag in [("pool_results_GB1_s24_b96i24x24.json", "GB1"),
                    ("pool_results_TEV_s24_b96i24x24.json", "TEV"),
                    ("pool_results_TrpB4_s24_b96i24x24.json", "TrpB4")]:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            if tag == "GB1" and os.path.exists(os.path.join(HERE, "pool_results_s24.json")):
                fn, path = "pool_results_s24.json", os.path.join(HERE, "pool_results_s24.json")
            else:
                print(f"skip {tag}: {fn} not present")
                continue
        fig_policy_minus_base(fn, tag)
        fig_paired_seed_diffs(fn, tag)
    fig_decomposition_gb1()
