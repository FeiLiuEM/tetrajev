#!/usr/bin/env python3
"""Jev-General: OpenSanctions Pairs 9,800 对（实体匹配）on our stack。"""
import argparse, json, math, time, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."
BASE = '/media/super/Hard_Disk_2/hermes/research/jev-general'

FIELD_SPECS = [
    ("name", "Names", 8, ", "), ("alias", "Aliases", 4, ", "),
    ("birthDate", "Birth Date", None, ", "), ("birthPlace", "Birth Place", 2, ", "),
    ("nationality", "Nationality", None, ", "), ("country", "Country", None, ", "),
    ("address", "Address", 3, "; "), ("idNumber", "ID Numbers", 3, ", "),
    ("passportNumber", "Passport", 2, ", "), ("gender", "Gender", None, ", "),
    ("position", "Position", 2, ", "), ("firstName", "First Name", None, ", "),
    ("lastName", "Last Name", None, ", "),
]
INSTRUCTIONS = (
    "You are an expert entity resolution system for sanctions screening. "
    "Do `entity_a` and `entity_b` refer to the same real-world person or organization? "
    "Your primary task is to identify CONFLICTS, not similarities. Name variations "
    "(transliterations, nicknames, titles) are common. Missing fields are normal - absence "
    "of data is NOT evidence of difference. The same entity often appears across multiple "
    "sources with variations. Look for CONTRADICTORY evidence (different dates, conflicting "
    "IDs, incompatible attributes). If no contradictions are found, they are the same entity. "
    "The DEFAULT is the same entity unless you find proof of difference.")


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


def format_entity(entity):
    props = entity.get('properties', {})
    lines = [f"Type: {entity.get('schema', 'Unknown')}"]
    for key, label, limit, sep in FIELD_SPECS:
        if key in props:
            values = props[key][:limit] if limit else props[key]
            lines.append(f"{label}: {sep.join(values)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(f'{BASE}/data/osbench/sample_10000.json', encoding='utf-8'))
    pairs = d['pairs'][200:]  # 论文 dev = 前 200
    if a.limit:
        pairs = pairs[:a.limit]
    out = []
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        state = f"Entity A:\n{format_entity(p['left'])}\n\nEntity B:\n{format_entity(p['right'])}"
        rec = {'i': i - 1, 'judgement': p['judgement']}
        try:
            pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{state}\n\n"
                  f"{INSTRUCTIONS}\n\nReply yes or no.<|im_end|>\n<|im_start|>assistant\n")
            lut = read_top(a.server, pr)
            ys = [lut[s] for s in variants('yes') if s in lut]
            ns = [lut[s] for s in variants('no') if s in lut]
            if not ys or not ns:
                raise RuntimeError('yes/no missing')
            p_same = 1.0 / (1.0 + math.exp(-(lse(ys) - lse(ns))))
            rec['pair'] = {'ok': True, 'p_same': p_same}
        except Exception as e:  # noqa: BLE001
            rec['pair'] = {'ok': False, 'error': str(e)[:160]}
        try:
            opts = ['Yes', 'No']
            pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{state}\n\n"
                  f"{INSTRUCTIONS}\n\nOptions: A=Yes, B=No.\n\n"
                  f"Answer with exactly one option letter: A, B. Output only the letter."
                  f"<|im_end|>\n<|im_start|>assistant\n")
            lut = read_top(a.server, pr)
            raw = {}
            for j, L in enumerate(('A', 'B')):
                vals = [lut[s] for s in variants(L) if s in lut]
                raw[j] = lse(vals) if vals else None
            if raw[0] is None or raw[1] is None:
                raise RuntimeError('letter missing')
            rec['letter'] = {'ok': True, 'p_yes': 1.0 / (1.0 + math.exp(-(raw[0] - raw[1])))}
        except Exception as e:  # noqa: BLE001
            rec['letter'] = {'ok': False, 'error': str(e)[:160]}
        out.append(rec)
        if i % 100 == 0:
            print(f'  {i}/{len(pairs)} | {(time.time()-t0)/i:.2f}s/pair', flush=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # F1（pair 档按 p>=0.5）
    for rd, key in (('pair', 'p_same'), ('letter', 'p_yes')):
        ok = [r for r in out if r[rd].get('ok')]
        tp = sum(1 for r in ok if r[rd][key] >= 0.5 and r['judgement'] == 'positive')
        fp = sum(1 for r in ok if r[rd][key] >= 0.5 and r['judgement'] == 'negative')
        fn = sum(1 for r in ok if r[rd][key] < 0.5 and r['judgement'] == 'positive')
        tn = sum(1 for r in ok if r[rd][key] < 0.5 and r['judgement'] == 'negative')
        prec = tp / (tp + fp) if tp + fp else 0
        rec_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0
        print(f'  [{a.tag}/{rd}] F1={f1*100:.2f} | P={prec*100:.2f} R={rec_*100:.2f} | TP{tp} FP{fp} FN{fn} TN{tn} | n={len(ok)}', flush=True)
    print('->', a.out, flush=True)


if __name__ == '__main__':
    main()