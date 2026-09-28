#!/usr/bin/env python3
"""Jev-General: banking20 (anyjev task replica) on our stack.
Exact replica of anyjev/bench/tasks/banking.py + Task.split(300, 200, seed=0).
"""
import argparse, glob, json, math, random, time, urllib.request
import pyarrow.parquet as pq
from collections import Counter

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."
K = 20


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
    return {k: v / s for k, v in raw.items()} if False else {k: v / s for k, v in ex.items()}


def load_banking20(base):
    ds_dir = glob.glob(f'{base}/data/banking77/*/')[0] if glob.glob(f'{base}/data/banking77/*/') else f'{base}/data/banking77/'
    tr = None; te = None
    for f in glob.glob(ds_dir + '**/*.parquet', recursive=True):
        t = pq.read_table(f).to_pylist()
        if 'train' in f:
            tr = t
        elif 'test' in f:
            te = t
    if tr is None or te is None:
        # 有时文件在子目录里，按 split 字段判断
        allf = sum([pq.read_table(f).to_pylist() for f in glob.glob(ds_dir + '**/*.parquet', recursive=True)], [])
        tr = [r for r in allf if r.get('split') == 'train']
        te = [r for r in allf if r.get('split') == 'test']
    names = {}
    for row in tr:
        names[int(row['label'])] = row['label_text']
    top = [c for c, _ in Counter(int(x) for x in [r['label'] for r in tr]).most_common(K)]
    top_sorted = sorted(top)
    pretty = [names[c].replace('_', ' ') for c in top_sorted]
    remap = {c: i for i, c in enumerate(top_sorted)}
    items = [(row['text'], remap[int(row['label'])]) for row in te if int(row['label']) in remap]
    rng = random.Random(0)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    test = [items[i] for i in idx[:300]]
    return pretty, test


def letter_prompt(text, labels):
    letters = [chr(ord('A') + i) for i in range(len(labels))]
    opt_line = ', '.join(f'{L}={o}' for L, o in zip(letters, labels))
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{text}\n\n"
            f"Question: What is the customer's intent?\n\nOptions: {opt_line}.\n\n"
            f"Answer with exactly one option letter: {', '.join(letters)}. Output only the letter."
            f"<|im_end|>\n<|im_start|>assistant\n")


def pair_prompt(text, cand):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{text}\n\n"
            f"Question: What is the customer's intent?\n\nCandidate answer: {cand}\n\n"
            f"Is the candidate answer correct? Reply yes or no.<|im_end|>\n<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='/media/super/Hard_Disk_2/hermes/research/jev-general')
    a = ap.parse_args()
    labels, test = load_banking20(a.base)
    print(f'banking20: {len(test)} test items, K={len(labels)}', flush=True)
    out = []
    t0 = time.time()
    for i, (text, gold) in enumerate(test, 1):
        rec = {'idx': i - 1, 'gold_index': gold, 'labels': labels}
        for rd in ('letter', 'pair'):
            try:
                if rd == 'letter':
                    lut = read_top(a.server, letter_prompt(text, labels))
                    raw = {}
                    for j, o in enumerate(labels):
                        L = chr(ord('A') + j)
                        vals = [lut[s] for s in variants(L) if s in lut]
                        raw[o] = lse(vals) if vals else None
                    probs = softmax_dict(raw)
                else:
                    raw = {}
                    for o in labels:
                        lut = read_top(a.server, pair_prompt(text, o))
                        ys = [lut[s] for s in variants('yes') if s in lut]
                        ns = [lut[s] for s in variants('no') if s in lut]
                        if not ys or not ns:
                            raise RuntimeError('yes/no missing')
                        raw[o] = lse(ys) - lse(ns)
                    probs = softmax_dict(raw)
                pred = max(probs, key=probs.get)
                rec[rd] = {'ok': True, 'probs': probs, 'pred_index': labels.index(pred)}
            except Exception as e:  # noqa: BLE001
                rec[rd] = {'ok': False, 'error': str(e)[:160]}
        out.append(rec)
        if i % 25 == 0:
            print(f'  {i}/{len(test)} | {(time.time()-t0)/i:.2f}s/item', flush=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'-> {a.out} ({time.time()-t0:.0f}s)', flush=True)
    for rd in ('letter', 'pair'):
        ok = [r for r in out if r[rd].get('ok')]
        c = sum(1 for r in ok if r[rd]['pred_index'] == r['gold_index'])
        print(f'  [{a.tag}/{rd}] acc={c/max(1,len(ok)):.4f} ({c}/{len(ok)})', flush=True)


if __name__ == '__main__':
    main()