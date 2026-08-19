#!/usr/bin/env python3
"""Fully logged free-pool re-issue: generate N policies via OpenRouter, ground, execute on GB1.

Writes:
  pools/logged_free_<model_tag>.json  (with full provenance)
  pool_results_GB1_logged_free_<tag>.json
"""
from __future__ import annotations
import copy, json, os, sys, time, re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import campaign as C
import policy_space as PS

MODEL = os.environ.get("BRIDGE_MODEL", "openai/gpt-4o-mini")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
N = int(os.environ.get("N_POL", "10"))
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
NPROC = int(os.environ.get("NPROC", "4"))
TEMP = float(os.environ.get("TEMP", "0.5"))
TAG = os.environ.get("TAG", "gpt4omini")
BUDGET, N_INIT, BATCH = 96, 24, 24


def menu_text():
    lines = []
    for comp, kv in PS.COORDINATES.items():
        for k, vals in kv.items():
            lines.append(f"  {comp}.{k}: {list(vals)}")
    return "\n".join(lines)


def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": TEMP, "max_tokens": 3500}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/odk-ucb/pe-discovery-policy-audit",
            "X-Title": "logged free pool reissue",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"], data


def parse_policies(text, k):
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in arr[:k]:
        if not isinstance(item, dict):
            continue
        if "coords" in item and isinstance(item["coords"], dict):
            out.append(item)
        elif any("." in str(kk) for kk in item.keys()):
            # flat -> nested
            coords = {}
            name = item.pop("name", None) if "name" in item else None
            for key, val in item.items():
                if "." not in key:
                    continue
                comp, kk = key.split(".", 1)
                coords.setdefault(comp, {})[kk] = val
            out.append({"name": name or f"logged_{len(out)}", "coords": coords})
        else:
            out.append({"name": item.get("name", f"logged_{len(out)}"), "coords": {k: v for k, v in item.items() if k != "name"}})
    return out


def ground(coords_partial):
    out = copy.deepcopy(PS.DEFAULT)
    problems = []
    for comp, kv in (coords_partial or {}).items():
        if comp not in PS.COORDINATES:
            problems.append(f"unknown {comp}")
            continue
        if not isinstance(kv, dict):
            continue
        for k, v in kv.items():
            if k not in PS.COORDINATES[comp]:
                problems.append(f"unknown {comp}.{k}")
                continue
            if v not in PS.COORDINATES[comp][k]:
                problems.append(f"illegal {comp}.{k}={v}")
                continue
            out[comp][k] = v
    return out, problems


def _one(args):
    coords, seed = args
    h = C.Harness(budget=BUDGET, n_init=N_INIT, batch=BATCH, landscape_name="GB1")
    r = h.run(PS.Policy("x", "free_logged", coords), seed)
    if not r.ok:
        return None
    return float(r.gain) if hasattr(r, "gain") and r.gain == r.gain else float(r.best)


def main():
    if not API_KEY:
        raise SystemExit("no OPENROUTER_API_KEY")
    user = f"""Propose {N} diverse protein directed-evolution campaign configs as a JSON array.
Each element: {{"name": "...", "coords": {{"representation": {{...}}, "surrogate": {{...}}, ...}}}}
Only use allowed coordinates:
{menu_text()}
Classical DEFAULT for reference: onehot, ridge, greedy_mean, global pool.
Optimize for combinatorial fitness landscapes under 96-assay / 4x24 budget.
Return ONLY the JSON array."""
    text, raw = chat([
        {"role": "system", "content": "You design PE discovery policies. JSON only."},
        {"role": "user", "content": user},
    ])
    parsed = parse_policies(text, N)
    prov = dict(
        model=MODEL, temperature=TEMP, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prompt_user=user, response_text=text, n_requested=N, n_parsed=len(parsed),
        openrouter_id=raw.get("id"),
    )
    policies = []
    for i, rec in enumerate(parsed):
        coords, problems = ground(rec.get("coords") or {})
        policies.append(dict(
            name=rec.get("name") or f"logged_{i:02d}",
            coords=coords,
            grounding_problems=problems,
            raw=rec,
        ))
    pool_path = HERE / "pools" / f"logged_free_{TAG}.json"
    json.dump(dict(condition="LOGGED_FREE", provenance=prov, policies=policies), open(pool_path, "w"), indent=2)
    print("wrote", pool_path, "n", len(policies), flush=True)

    # execute vs default
    seeds = list(range(N_SEEDS))
    results = {}
    # default
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        db = list(ex.map(_one, [(copy.deepcopy(PS.DEFAULT), s) for s in seeds]))
    results["SEED:base_onehot_ridge"] = dict(
        channel="seed", best=float(np.nanmean([x for x in db if x is not None])),
        seeds=[x for x in db if x is not None], fail=sum(1 for x in db if x is None),
    )
    for pol in policies:
        if pol["grounding_problems"]:
            print("skip bad", pol["name"], pol["grounding_problems"][:3], flush=True)
            continue
        print("run", pol["name"], flush=True)
        with ProcessPoolExecutor(max_workers=NPROC) as ex:
            rs = list(ex.map(_one, [(pol["coords"], s) for s in seeds]))
        ok = [x for x in rs if x is not None]
        results[f"LOGGED:{pol['name'][:40]}"] = dict(
            channel="free_logged", best=float(np.mean(ok)) if ok else None,
            seeds=ok, fail=len(rs) - len(ok), provenance_tag=TAG, model=MODEL,
        )
        print("  mean", results[f"LOGGED:{pol['name'][:40]}"]["best"], flush=True)

    # beat default?
    base = results["SEED:base_onehot_ridge"]["best"]
    beats = []
    for k, v in results.items():
        if k.startswith("LOGGED") and v["best"] is not None and v["best"] > base:
            beats.append((k, v["best"] - base))
    out = dict(__build__=f"logged_free_{TAG}", __provenance__=prov, **results)
    out_path = HERE / f"pool_results_GB1_logged_free_{TAG}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    summary = dict(base=base, n_policies=len(policies), n_beats_raw=len(beats), beats=beats, out=str(out_path))
    json.dump(summary, open(HERE / f"logged_free_{TAG}_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
