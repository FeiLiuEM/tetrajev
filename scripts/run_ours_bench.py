#!/usr/bin/env python3
"""Jev-General: bench runner for our method (two frozen readers, two readout structures).

Readouts:
  letter : one prompt, options mapped to letters A.., read letter-token mass   (1 call/item)
  pair   : per label, "Is the candidate answer correct? yes/no" -> delta       (K calls/item)

Usage (on super, llama-server must be up):
  python3 run_ours_bench.py --items data/jevbench_our_format/easy.jsonl,data/jevbench_our_format/original.jsonl,data/jevbench_our_format/hard.jsonl \
      --server http://127.0.0.1:10361 --tag 27b --readouts letter,pair --out results/ours_27b.jsonl
"""
import argparse, json, math, time, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."


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


def variants(lab):
    s = str(lab)
    out = [s, " " + s, s.lower(), " " + s.lower(), s.upper(), " " + s.upper(),
           s + ".", " " + s + ".", '"' + s + '"']
    return list(dict.fromkeys(out))


def letter_prompt(it):
    labs = [str(x) for x in it['labels']]
    letters = [chr(ord('A') + i) for i in range(len(labs))]
    opt_line = ', '.join(f'{L}={lab}' for L, lab in zip(letters, labs))
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{it['input']}\n\n"
            f"Options: {opt_line}.\n\nAnswer with exactly one option letter: {', '.join(letters)}. "
            f"Output only the letter.<|im_end|>\n<|im_start|>assistant\n")


def pair_prompt(it, lab):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{it['input']}\n\n"
            f"Candidate answer: {lab}\n\nIs the candidate answer correct? Reply yes or no."
            f"<|im_end|>\n<|im_start|>assistant\n")


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


def letter_read(server, it):
    lut = read_top(server, letter_prompt(it))
    raw = {}
    for i, lab in enumerate(it['labels']):
        L = chr(ord('A') + i)
        vals = [lut[s] for s in variants(L) if s in lut]
        raw[str(lab)] = lse(vals) if vals else None
    if not any(v is not None for v in raw.values()):
        raise RuntimeError('no letter logprobs in top-200')
    return softmax_dict(raw)


def pair_read(server, it):
    raw = {}
    for lab in it['labels']:
        lut = read_top(server, pair_prompt(it, lab))
        ys = [lut[s] for s in variants('yes') if s in lut]
        ns = [lut[s] for s in variants('no') if s in lut]
        if not ys or not ns:
            raise RuntimeError(f'yes/no not in top-200 (label={lab})')
        raw[str(lab)] = lse(ys) - lse(ns)
    return softmax_dict(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True, help='comma list of jsonl files')
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--readouts', default='letter,pair')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = []
    for f in a.items.split(','):
        rows.extend(json.loads(l) for l in open(f, encoding='utf-8') if l.strip())
    if a.limit:
        rows = rows[:a.limit]
    want = [x.strip() for x in a.readouts.split(',') if x.strip()]
    print(f'items: {len(rows)} | readouts: {want} | server: {a.server}', flush=True)

    out_rows = []
    t0 = time.time()
    for i, it in enumerate(rows, 1):
        rec = {'id': it['id'], 'split': it.get('split'), 'family': it.get('family'),
               'n_labels': len(it['labels']), 'expected': it['expected'], 'labels': it['labels'],
               'readouts': {}}
        for rd in want:
            for attempt in (1, 2):
                try:
                    probs = (letter_read if rd == 'letter' else pair_read)(a.server, it)
                    pred = max(probs, key=probs.get)
                    rec['readouts'][rd] = {'ok': True, 'probs': probs, 'pred': pred,
                                           'correct': int(pred == it['expected'])}
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        rec['readouts'][rd] = {'ok': False, 'error': str(e)[:200]}
                    else:
                        time.sleep(1.0)
        out_rows.append(rec)
        if i % 20 == 0:
            dt = time.time() - t0
            oks = [r for r in out_rows for v in r['readouts'].values() if v.get('ok')]
            cor = sum(v['correct'] for r in out_rows for v in r['readouts'].values() if v.get('ok'))
            print(f'  {i}/{len(rows)} | {dt/i:.2f}s/item | readout-acc {cor/max(1,len(oks)):.4f} (n={len(oks)})', flush=True)

    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # summary per split per readout
    print('== summary ==', flush=True)
    for rd in want:
        for sp in sorted({r['split'] for r in out_rows} | {'ALL'}):
            sel = [r for r in out_rows if (sp == 'ALL' or r['split'] == sp) and r['readouts'].get(rd, {}).get('ok')]
            if sel:
                cor = sum(r['readouts'][rd]['correct'] for r in sel)
                print(f'  [{a.tag}/{rd}/{sp}] {cor}/{len(sel)} = {cor/len(sel):.4f}', flush=True)
    print(f'-> {a.out}', flush=True)


if __name__ == '__main__':
    main()
