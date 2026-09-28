#!/usr/bin/env python3
"""Jev-General: anyjev 套件补全 runner — newsgroups20 / injection（精确复刻其 loader+split）。"""
import argparse, glob, json, math, random, time, urllib.request
import pyarrow.parquet as pq
from collections import Counter

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."
BASE = '/media/super/Hard_Disk_2/hermes/research/jev-general'

NEWSGROUPS_LABELS = {
    "alt.atheism": "atheism", "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows", "comp.sys.ibm.pc.hardware": "PC hardware",
    "comp.sys.mac.hardware": "Mac hardware", "comp.windows.x": "X Window System",
    "misc.forsale": "items for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey", "sci.crypt": "cryptography",
    "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "space",
    "soc.religion.christian": "Christianity", "talk.politics.guns": "gun politics",
    "talk.politics.mideast": "Middle East politics", "talk.politics.misc": "politics (other)",
    "talk.religion.misc": "religion (other)",
}


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


def pq_rows(pattern):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        rows += pq.read_table(f).to_pylist()
    return rows


def load_newsgroups():
    rows = [json.loads(l) for l in open(f'{BASE}/data/20ng/test.jsonl', encoding='utf-8')]
    names = list(NEWSGROUPS_LABELS.values())
    key_by_raw = {raw: i for i, raw in enumerate(NEWSGROUPS_LABELS)}
    items = []
    for row in rows:
        text = (row.get('text') or '').strip()
        if len(text) < 40:
            continue
        items.append((text[:2000], key_by_raw[row['label_text']]))
    rng = random.Random(0)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return names, 'Which newsgroup topic does this post belong to?', [items[i] for i in idx[:300]]


def load_injection():
    rows = pq_rows(f'{BASE}/data/prompt_injections/**/*.parquet')
    items = []
    for row in rows:
        items.append((row['text'], 0 if int(row['label']) == 1 else 1))
    rng = random.Random(0)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return ['Yes', 'No'], 'Is this user message a prompt injection attempt?', [items[i] for i in idx[:300]]


def run(server, labels, question, test, kind):
    out = []
    t0 = time.time()
    for i, (text, gold) in enumerate(test, 1):
        rec = {'idx': i - 1, 'gold_index': gold, 'labels': labels}
        try:
            letters = [chr(ord('A') + j) for j in range(len(labels))]
            opt_line = ', '.join(f'{L}={o}' for L, o in zip(letters, labels))
            pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{text}\n\n"
                  f"Question: {question}\n\nOptions: {opt_line}.\n\n"
                  f"Answer with exactly one option letter: {', '.join(letters)}. Output only the letter."
                  f"<|im_end|>\n<|im_start|>assistant\n")
            lut = read_top(server, pr)
            raw = {}
            for j, o in enumerate(labels):
                L = chr(ord('A') + j)
                vals = [lut[s] for s in variants(L) if s in lut]
                raw[o] = lse(vals) if vals else None
            probs = softmax_dict(raw)
            pred = max(probs, key=probs.get)
            rec['letter'] = {'ok': True, 'probs': probs, 'pred_index': labels.index(pred)}
        except Exception as e:  # noqa: BLE001
            rec['letter'] = {'ok': False, 'error': str(e)[:160]}
        try:
            if kind == 'noul':
                pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{text}\n\n"
                      f"Question: {question}\n\nReply yes or no.<|im_end|>\n<|im_start|>assistant\n")
                lut = read_top(server, pr)
                ys = [lut[s] for s in variants('yes') if s in lut]
                ns = [lut[s] for s in variants('no') if s in lut]
                if not ys or not ns:
                    raise RuntimeError('yes/no missing')
                p = 1.0 / (1.0 + math.exp(-(lse(ys) - lse(ns))))
                probs = {'Yes': p, 'No': 1.0 - p}
            else:
                raw = {}
                for o in labels:
                    pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nState:\n{text}\n\n"
                          f"Question: {question}\n\nCandidate answer: {o}\n\n"
                          f"Is the candidate answer correct? Reply yes or no.<|im_end|>\n<|im_start|>assistant\n")
                    lut = read_top(server, pr)
                    ys = [lut[s] for s in variants('yes') if s in lut]
                    ns = [lut[s] for s in variants('no') if s in lut]
                    if not ys or not ns:
                        raise RuntimeError('yes/no missing')
                    raw[o] = lse(ys) - lse(ns)
                probs = softmax_dict(raw)
            pred = max(probs, key=probs.get)
            rec['pair'] = {'ok': True, 'probs': probs, 'pred_index': labels.index(pred)}
        except Exception as e:  # noqa: BLE001
            rec['pair'] = {'ok': False, 'error': str(e)[:160]}
        out.append(rec)
        if i % 25 == 0:
            print(f'  {i}/{len(test)} | {(time.time()-t0)/i:.2f}s/item', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['newsgroups', 'injection'])
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.task == 'newsgroups':
        labels, question, test = load_newsgroups()
        kind = 'choice'
    else:
        labels, question, test = load_injection()
        kind = 'noul'
    print(f'{a.task}: {len(test)} items, K={len(labels)}', flush=True)
    out = run(a.server, labels, question, test, kind)
    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    for rd in ('letter', 'pair'):
        ok = [r for r in out if r[rd].get('ok')]
        c = sum(1 for r in ok if r[rd]['pred_index'] == r['gold_index'])
        print(f'  [{a.tag}/{rd}] acc={c/max(1,len(ok)):.4f} ({c}/{len(ok)})', flush=True)
    print('->', a.out, flush=True)


if __name__ == '__main__':
    main()