#!/usr/bin/env python3
"""Jev-General: DecisionBench bench-v4 (1071 rows) on our stack (letter+pair readouts)."""
import argparse, json, math, time, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."


def post(server, path, payload, timeout=1200):
    req = urllib.request.Request(server.rstrip('/') + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tok(server, text):
    d = post(server, '/tokenize', {'content': text, 'add_special': False, 'parse_special': True})
    xs = d.get('tokens') if isinstance(d, dict) else d
    return [int(x) if isinstance(x, int) else int(x.get('id')) for x in xs]


def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))


def variants(s):
    s = str(s)
    out = [s, " " + s, s.lower(), " " + s.lower(), s.upper(), " " + s.upper(), s + ".", " " + s + ".", '"' + s + '"']
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', default='/media/super/Hard_Disk_2/hermes/research/jev-general/data/dbench/cases.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.cases, encoding='utf-8') if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        q = r['questions'][0]
        opts = [k for k in q.get('option_order', q['options'].keys())]
        state = json.dumps(r['state'], ensure_ascii=False)
        instr = q['instructions']
        rec = {'id': r['id'], 'task': r.get('task'), 'category': r.get('category'),
               'modality': r.get('modality'), 'gold': q['gold'], 'options': opts}
        for rd in ('letter', 'pair'):
            try:
                if rd == 'letter':
                    letters = [chr(ord('A') + j) for j in range(len(opts))]
                    opt_line = ', '.join(f'{L}={o}' for L, o in zip(letters, opts))
                    pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{state}\n\n"
                          f"Question: {instr}\n\nOptions: {opt_line}.\n\n"
                          f"Answer with exactly one option letter: {', '.join(letters)}. Output only the letter."
                          f"<|im_end|>\n<|im_start|>assistant\n")
                    lut = read_top(a.server, pr)
                    raw = {}
                    for j, o in enumerate(opts):
                        L = chr(ord('A') + j)
                        vals = [lut[s] for s in variants(L) if s in lut]
                        raw[o] = lse(vals) if vals else None
                    probs = softmax_dict(raw)
                else:
                    raw = {}
                    for o in opts:
                        pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{state}\n\n"
                              f"Question: {instr}\n\nCandidate answer: {o}\n\n"
                              f"Is the candidate answer correct? Reply yes or no.<|im_end|>\n<|im_start|>assistant\n")
                        lut = read_top(a.server, pr)
                        ys = [lut[s] for s in variants('yes') if s in lut]
                        ns = [lut[s] for s in variants('no') if s in lut]
                        if not ys or not ns:
                            raise RuntimeError('yes/no missing')
                        raw[o] = lse(ys) - lse(ns)
                    probs = softmax_dict(raw)
                pred = max(probs, key=probs.get)
                rec[rd] = {'ok': True, 'probs': probs, 'pred': pred, 'correct': int(pred == q['gold'])}
            except Exception as e:  # noqa: BLE001
                rec[rd] = {'ok': False, 'error': str(e)[:160]}
                rec[rd]['correct'] = 0
        out.append(rec)
        if i % 25 == 0:
            print(f'  {i}/{len(rows)} | {(time.time()-t0)/i:.2f}s/item', flush=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'-> {a.out} ({time.time()-t0:.0f}s)', flush=True)
    for rd in ('letter', 'pair'):
        ok = [r for r in out if r[rd].get('ok')]
        c = sum(r[rd]['correct'] for r in ok)
        print(f'  [{a.tag}/{rd}] acc={c/max(1,len(ok)):.4f} ({c}/{len(ok)})', flush=True)


if __name__ == '__main__':
    main()