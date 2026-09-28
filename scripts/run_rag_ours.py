#!/usr/bin/env python3
"""Jev-General: Jev RAG Benchmark 重排序（scifact / xquad-en，冻结 top-20 候选）on our stack。"""
import argparse, json, math, time, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYS = "You are a careful decision expert. Judge strictly by the given criteria and answer with one label only."
BASE = '/media/super/Hard_Disk_2/hermes/research/jev-general'


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


def ndcg_at_k(order, gold, k=10):
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(order[:k], start=1) if d in gold)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(order, gold, k=5):
    return len({d for d in order[:k] if d in gold}) / max(1, len(gold))


def mrr_at_k(order, gold, k=10):
    for rank, d in enumerate(order[:k], start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--server', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit-queries', type=int, default=0)
    a = ap.parse_args()
    frozen = {'scifact': 'scifact-a-j-n-real.jsonl', 'xquad-en': 'xquad-en-a-j-n-real.jsonl'}[a.dataset]
    corpus = {}
    for l in open(f'{BASE}/data/rag/processed/{a.dataset}/corpus.jsonl', encoding='utf-8'):
        r = json.loads(l)
        corpus[r['doc_id']] = r
    rows = [json.loads(l) for l in open(f'{BASE}/data/rag/frozen/{frozen}', encoding='utf-8') if l.strip()]
    if a.limit_queries:
        rows = rows[:a.limit_queries]
    print(f'{a.dataset}: {len(rows)} queries, corpus {len(corpus)} docs', flush=True)
    out = []
    t0 = time.time()
    for qi, row in enumerate(rows, 1):
        q = row['question']
        gold = set(row['gold_doc_ids'])
        cands = sorted(row['candidates'], key=lambda c: c['rrf_rank'])
        base_order = [c['doc_id'] for c in cands]
        scores = []
        errors = 0
        for c in cands:
            doc = corpus.get(c['doc_id']) or {'title': '', 'text': ''}
            passage = ((doc.get('title') or '') + ' ' + (doc.get('text') or '')).strip()
            pr = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"
                  f"Question: {q}\n\nCandidate passages:\n\n[0] {passage}\n\n"
                  f"Passage [0] contains information that is needed to answer the question. "
                  f"Reply yes or no.<|im_end|>\n<|im_start|>assistant\n")
            try:
                lut = read_top(a.server, pr)
                ys = [lut[s] for s in variants('yes') if s in lut]
                ns = [lut[s] for s in variants('no') if s in lut]
                if not ys or not ns:
                    raise RuntimeError('yes/no missing')
                scores.append((c['doc_id'], lse(ys) - lse(ns), c['rrf_rank']))
            except Exception:  # noqa: BLE001
                errors += 1
                scores.append((c['doc_id'], -1e9, c['rrf_rank']))
        ours_order = [d for d, _, _ in sorted(scores, key=lambda x: (-x[1], x[2]))]
        rec = {'query_id': row['query_id'], 'gold': sorted(gold), 'n_cand': len(cands), 'errors': errors,
               'ndcg10_ours': ndcg_at_k(ours_order, gold), 'recall5_ours': recall_at_k(ours_order, gold),
               'mrr10_ours': mrr_at_k(ours_order, gold),
               'ndcg10_base': ndcg_at_k(base_order, gold), 'recall5_base': recall_at_k(base_order, gold),
               'mrr10_base': mrr_at_k(base_order, gold),
               'scores': [{'doc': d, 'delta': round(s, 4), 'rrf': r} for d, s, r in scores]}
        out.append(rec)
        if qi % 25 == 0:
            print(f'  {qi}/{len(rows)} | {(time.time()-t0)/qi:.2f}s/query', flush=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    for arm in ('ours', 'base'):
        m = {k: sum(r[k] for r in out) / len(out) for k in (f'ndcg10_{arm}', f'recall5_{arm}', f'mrr10_{arm}')}
        print(f'  [{a.tag}/{arm}] nDCG@10={m[f"ndcg10_{arm}"]*100:.2f} Recall@5={m[f"recall5_{arm}"]*100:.2f} MRR@10={m[f"mrr10_{arm}"]*100:.2f}', flush=True)
    print('->', a.out, flush=True)


if __name__ == '__main__':
    main()