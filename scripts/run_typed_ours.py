#!/usr/bin/env python3
"""Jev-General: run our stack on LocalLLaMA/typed-decisions (test split, 2000 decisions).

Readouts: letter (1 call) + pair (K calls; noul = 1 delta call).
Requires a llama-server (np=1).
"""
import argparse, glob, json, math, time, urllib.request
import pyarrow.parquet as pq

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."
BASE = '/media/super/Hard_Disk_2/hermes/research/jev-general'


def post(server, path, payload, timeout=1200):
    req = urllib.request.Request(server.rstrip('/') + path,
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tok(server, text):
    d = post(server, '/tokenize', {'content': text, 'add_special': False, 'parse_special': True})
    xs = d.get('tokens') if isinstance(d, dict) else d
    return [int(x) if isinstance(x, int) else int(x.get('id')) for x in xs]


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def variants(s):
    s = str(s)
    out = [s, " " + s, s.lower(), " " + s.lower(), s.upper(), " " + s.upper(),
           s + ".", " " + s + ".", '"' + s + '"']
    return list(dict.fromkeys(out))


def read_top(server, prompt):
    ids = tok(server, prompt)
    d = post(server, '/completion', {'prompt': ids, 'n_predict': 1, 'n_probs': 2000,
                                     'temperature': 0.0, 'cache_prompt': False})
    tps = (d.get('completion_probabilities') or [{}])[0].get('top_logprobs') or []
    return {e['token']: e['logprob'] for e in tps}


def softmax_dict(raw):
    mx = max(v for v in raw.values() if v is not None)
    ex = {k: (math.exp(v - mx) if v is not None else 0.0) for k, v in raw.items()}
    s = sum(ex.values()) or 1.0
    return {k: v / s for k, v in ex.items()}


def qtext(kind, instr, spec):
    """question text used in prompts (mirrors the dataset's framing)"""
    if kind == 'noul':
        crit = spec.get('criteria') or {}
        t = f"Is the following statement true? {instr}"
        if isinstance(crit, dict):
            t += f"\nYes means: {crit.get('true', '')}\nNo means: {crit.get('false', '')}"
        return t
    return instr


def options_of(kind, spec):
    crit = spec.get('criteria')
    if kind == 'choice':
        if isinstance(crit, dict):
            return [f"{k}: {v}" for k, v in crit.items()]
        return [str(c) for c in crit]
    if kind == 'score':
        return list(crit.values()) if isinstance(crit, dict) else [str(c) for c in crit]
    return ['Yes', 'No']


def letter_prompt(state_text, kind, instr, spec, opts):
    q = qtext(kind, instr, spec)
    letters = [chr(ord('A') + i) for i in range(len(opts))]
    opt_line = ', '.join(f'{L}={o}' for L, o in zip(letters, opts))
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{state_text}\n\n"
            f"Question: {q}\n\nOptions: {opt_line}.\n\n"
            f"Answer with exactly one option letter: {', '.join(letters)}. Output only the letter."
            f"<|im_end|>\n<|im_start|>assistant\n")


def pair_prompt(state_text, kind, instr, spec, cand):
    q = qtext(kind, instr, spec)
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{state_text}\n\n"
            f"Question: {q}\n\nCandidate answer: {cand}\n\n"
            f"Is the candidate answer correct? Reply yes or no.<|im_end|>\n<|im_start|>assistant\n")


def noul_prompt(state_text, instr, spec):
    q = qtext('noul', instr, spec)
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{state_text}\n\n"
            f"Question: {q}\n\nReply yes or no.<|im_end|>\n<|im_start|>assistant\n")


def letter_read(server, state_text, kind, instr, spec, opts):
    lut = read_top(server, letter_prompt(state_text, kind, instr, spec, opts))
    raw = {}
    for i, o in enumerate(opts):
        L = chr(ord('A') + i)
        vals = [lut[s] for s in variants(L) if s in lut]
        raw[o] = lse(vals) if vals else None
    if not any(v is not None for v in raw.values()):
        raise RuntimeError('no letter logprobs in top-2000')
    return softmax_dict(raw)


def pair_read(server, state_text, kind, instr, spec, opts):
    if kind == 'noul':
        lut = read_top(server, noul_prompt(state_text, instr, spec))
        ys = [lut[s] for s in variants('yes') if s in lut]
        ns = [lut[s] for s in variants('no') if s in lut]
        if not ys or not ns:
            raise RuntimeError('yes/no not in top-2000')
        p = 1.0 / (1.0 + math.exp(-(lse(ys) - lse(ns))))
        return {'Yes': p, 'No': 1.0 - p}
    raw = {}
    for o in opts:
        lut = read_top(server, pair_prompt(state_text, kind, instr, spec, o))
        ys = [lut[s] for s in variants('yes') if s in lut]
        ns = [lut[s] for s in variants('no') if s in lut]
        if not ys or not ns:
            raise RuntimeError(f'yes/no not in top-2000 (cand={o[:40]})')
        raw[o] = lse(ys) - lse(ns)
    return softmax_dict(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    f = glob.glob(f'{BASE}/data/typed_decisions/all/test-*.parquet')[0]
    ds_rows = pq.read_table(f).to_pylist()
    if a.limit:
        ds_rows = ds_rows[:a.limit]

    out_rows = []
    t0 = time.time()
    total = sum(len(json.loads(r['questions'])) for r in ds_rows)
    done = 0
    for row in ds_rows:
        state_text = json.dumps(json.loads(row['state']), ensure_ascii=False)
        qs = json.loads(row['questions'])
        gold = json.loads(row['gold'])
        for qname, spec in qs.items():
            kind = spec['type']
            instr = spec.get('instructions', '')
            opts = options_of(kind, spec)
            g = gold[qname]
            if kind == 'score':
                gi = int(str(g['label']))
            elif kind == 'noul':
                gi = 0 if str(g['label']) == 'true' else 1
            else:
                keys = list(spec['criteria'].keys()) if isinstance(spec['criteria'], dict) else [str(c) for c in spec['criteria']]
                gi = keys.index(str(g['label']))
            gp = [float(g['probabilities'].get(k, 0.0)) for k in (
                [str(i) for i in range(len(opts))] if kind == 'score' else
                (['true', 'false'] if kind == 'noul' else
                 (list(spec['criteria'].keys()) if isinstance(spec['criteria'], dict) else [str(c) for c in spec['criteria']])))]
            rec = {'case_id': row['id'], 'workflow': row['workflow'], 'qname': qname, 'kind': kind,
                   'k': len(opts), 'opts': opts, 'gold_index': gi, 'gold_probs': gp,
                   'gold_score': float(g['score']) if kind == 'score' and 'score' in g else None,
                   'readouts': {}}
            for rd, fn in (('letter', letter_read), ('pair', pair_read)):
                for attempt in (1, 2):
                    try:
                        probs = fn(a.server, state_text, kind, instr, spec, opts)
                        pred = max(probs, key=probs.get)
                        rec['readouts'][rd] = {'ok': True, 'probs': probs, 'pred_idx': opts.index(pred)}
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            rec['readouts'][rd] = {'ok': False, 'error': str(e)[:180]}
                        else:
                            time.sleep(1.0)
            out_rows.append(rec)
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                print(f'  {done}/{total} | {dt/done:.2f}s/decision | eta {(total-done)*dt/done/60:.0f}min', flush=True)

    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'-> {a.out} ({len(out_rows)} decisions, {(time.time()-t0)/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()