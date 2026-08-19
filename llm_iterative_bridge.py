#!/usr/bin/env python3
"""LLM iterative bridge via OpenRouter: multi-generation policy proposal with fitness feedback.

Logs full generator provenance (model, prompts, temperature, raw responses).
Compares to one-shot LLM proposals under matched budget/seeds.
"""
from __future__ import annotations
import copy, json, os, sys, time, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import campaign as C
import policy_space as PS

MODEL = os.environ.get("BRIDGE_MODEL", "anthropic/claude-sonnet-5")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
N_SEEDS = int(os.environ.get("N_SEEDS", "6"))
BUDGET = int(os.environ.get("BUDGET", "96"))
N_INIT = int(os.environ.get("N_INIT", "24"))
BATCH = int(os.environ.get("BATCH", "24"))
K = int(os.environ.get("K", "3"))  # proposals per generation
G = int(os.environ.get("G", "3"))
NPROC = int(os.environ.get("NPROC", "4"))
TEMP = float(os.environ.get("TEMP", "0.4"))
OUT = HERE / "llm_iterative_bridge_results.json"

# Compact coordinate menu for the LLM
def menu_text():
    lines = []
    for comp, kv in PS.COORDINATES.items():
        for k, vals in kv.items():
            lines.append(f"  {comp}.{k}: {list(vals)}")
    return "\n".join(lines)


def default_flat():
    return {f"{c}.{k}": v for c, kv in PS.DEFAULT.items() for k, v in kv.items()}


def apply_partial(partial: dict):
    coords = copy.deepcopy(PS.DEFAULT)
    for key, val in partial.items():
        if "." not in key:
            continue
        comp, k = key.split(".", 1)
        if comp in coords and k in coords[comp]:
            # validate
            allowed = PS.COORDINATES.get(comp, {}).get(k)
            if allowed is not None and val in allowed:
                coords[comp][k] = val
    return coords


def _one(args):
    coords, seed, budget, n_init, batch = args
    h = C.Harness(budget=budget, n_init=n_init, batch=batch, landscape_name="GB1")
    p = PS.Policy("x", "llm_bridge", coords)
    r = h.run(p, seed)
    if not r.ok:
        return None
    return float(r.gain) if hasattr(r, "gain") and r.gain == r.gain else float(r.best)


def eval_coords(coords, seeds):
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        rs = list(ex.map(_one, [(coords, s, BUDGET, N_INIT, BATCH) for s in seeds]))
    ok = [x for x in rs if x is not None]
    if not ok:
        return dict(mean=float("nan"), fails=len(rs), seeds=[])
    return dict(mean=float(np.mean(ok)), fails=len(rs) - len(ok), seeds=ok)


def chat(messages, temperature=TEMP):
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set")
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/odk-ucb/pe-discovery-policy-audit",
            "X-Title": "PE policy audit iterative bridge",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"], data


def parse_policies(text, k):
    """Extract JSON list of partial coordinate dicts from model text."""
    # try fenced json
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in arr[:k]:
        if isinstance(item, dict):
            # flatten nested or accept flat
            if any("." in str(kk) for kk in item.keys()):
                out.append(item)
            else:
                flat = {}
                for comp, kv in item.items():
                    if isinstance(kv, dict):
                        for kk, vv in kv.items():
                            flat[f"{comp}.{kk}"] = vv
                    else:
                        flat[str(comp)] = kv
                out.append(flat)
    return out


SYSTEM = """You design protein directed-evolution campaign configs (discovery policies).
A policy is a full assignment of discrete coordinates. Illegal values are rejected.
Return ONLY a JSON array of objects. Each object maps coordinate keys like
"representation.encoding" to allowed values. Propose diverse configs that might beat
the classical default on mean gain (best fitness minus no-op anchor).
No prose outside the JSON array."""


def propose_oneshot(k):
    user = f"""Classical DEFAULT (flat):\n{json.dumps(default_flat(), indent=2)}

Allowed coordinates:
{menu_text()}

Propose {k} distinct one-shot policies as a JSON array of flat coordinate dicts
(only list coordinates you change from DEFAULT is OK). Optimize for GB1-style
combinatorial fitness under budget 96 assays / 4 rounds of 24."""
    text, raw = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ])
    return parse_policies(text, k), text, raw, user


def propose_iterative(k, history):
    hist_txt = json.dumps(history, indent=2)[:6000]
    user = f"""You are in a multi-generation loop. Prior generations measured mean GAIN on GB1
(higher is better). DEFAULT gain baseline is in history gen 0.

History:
{hist_txt}

Allowed coordinates:
{menu_text()}

Propose {k} new policies as JSON array of flat dicts, conditioning on what worked.
Prefer single- or few-coordinate changes from the current best, but you may explore."""
    text, raw = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ])
    return parse_policies(text, k), text, raw, user


def main():
    seeds = list(range(N_SEEDS))
    log = dict(
        model=MODEL, temperature=TEMP, seeds=N_SEEDS, budget=BUDGET, K=K, G=G,
        api="openrouter", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prompts=[], oneshot=[], iterative={"history": []}, contrast={},
    )
    # default
    print("eval DEFAULT...", flush=True)
    base = eval_coords(copy.deepcopy(PS.DEFAULT), seeds)
    print(f"  DEFAULT mean={base['mean']:.4f}", flush=True)
    log["default"] = base
    hist = [dict(gen=0, mean=base["mean"], label="DEFAULT", partial={})]

    # one-shot
    print("LLM one-shot propose...", flush=True)
    partials, text, raw, user = propose_oneshot(K)
    log["prompts"].append(dict(phase="oneshot", user=user, response=text, model=MODEL, temperature=TEMP))
    os_rows = []
    for i, part in enumerate(partials):
        coords = apply_partial(part)
        r = eval_coords(coords, seeds)
        os_rows.append(dict(partial=part, mean=r["mean"], fails=r["fails"], seeds=r["seeds"]))
        print(f"  oneshot[{i}] {part} -> {r['mean']:.4f}", flush=True)
    if not os_rows:
        # fallback: empty proposals
        print("  WARN: no parseable oneshot policies", flush=True)
    log["oneshot"] = os_rows
    best_os = max(os_rows, key=lambda d: d["mean"] if d["mean"] == d["mean"] else -1e9) if os_rows else dict(mean=base["mean"], partial={})

    # iterative
    cur_best_mean = base["mean"]
    cur_best_partial = {}
    for g in range(1, G + 1):
        print(f"LLM iterative gen {g}...", flush=True)
        partials, text, raw, user = propose_iterative(K, hist)
        log["prompts"].append(dict(phase=f"iter_g{g}", user=user, response=text, model=MODEL, temperature=TEMP))
        gen_rows = []
        for i, part in enumerate(partials):
            coords = apply_partial(part)
            r = eval_coords(coords, seeds)
            gen_rows.append(dict(partial=part, mean=r["mean"], fails=r["fails"], seeds=r["seeds"]))
            print(f"  iter g{g}[{i}] {part} -> {r['mean']:.4f}", flush=True)
        if gen_rows:
            best = max(gen_rows, key=lambda d: d["mean"] if d["mean"] == d["mean"] else -1e9)
            if best["mean"] == best["mean"] and best["mean"] > cur_best_mean:
                cur_best_mean = best["mean"]
                cur_best_partial = best["partial"]
            hist.append(dict(gen=g, mean=cur_best_mean, best_of_gen=best, all=gen_rows))
        else:
            hist.append(dict(gen=g, mean=cur_best_mean, best_of_gen=None, all=[], note="parse_fail"))
    log["iterative"]["history"] = hist
    log["iterative"]["final_mean"] = cur_best_mean
    log["iterative"]["final_partial"] = cur_best_partial
    os_m = best_os.get("mean", float("nan"))
    log["contrast"] = dict(
        oneshot_minus_default=(os_m - base["mean"]) if os_m == os_m else None,
        iterative_minus_default=(cur_best_mean - base["mean"]) if cur_best_mean == cur_best_mean else None,
        iterative_minus_oneshot=(cur_best_mean - os_m) if (cur_best_mean == cur_best_mean and os_m == os_m) else None,
    )
    json.dump(log, open(OUT, "w"), indent=2)
    print(json.dumps(log["contrast"], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
