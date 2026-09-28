#!/usr/bin/env python3
"""P3 scorer: OpenSanctions / spam-eval / RAG rerank（含部分完成的容错 + Jev 配对）。"""
import json, math, os
BASE = '/media/super/Hard_Disk_2/hermes/research/jev-general'

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def exists(p): return os.path.exists(p)

def f1_stats(rows, rd, key):
    ok = [r for r in rows if r[rd].get('ok')]
    tp = sum(1 for r in ok if r[rd][key] >= 0.5 and r['judgement'] == 'positive')
    fp = sum(1 for r in ok if r[rd][key] >= 0.5 and r['judgement'] == 'negative')
    fn = sum(1 for r in ok if r[rd][key] < 0.5 and r['judgement'] == 'positive')
    tn = sum(1 for r in ok if r[rd][key] < 0.5 and r['judgement'] == 'negative')
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return {'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0, 'p': pr, 'r': rc, 'n': len(ok), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

print('==================== OpenSanctions (9,800) ====================')
print('参照: GPT-4o 98.95 | Jev 1.13 98.87 [98.70,99.04] | Llama-3.1-8B 94.05 | Claude Opus 4.5 95.45 | GPT-5 Nano 95.24')
os_rows = {}
for tag in ('27b', '35b'):
    fp = f'{BASE}/results/osbench_{tag}.jsonl'
    if not exists(fp):
        print(f'  [{tag}] (未完成)')
        continue
    rows = load(fp); os_rows[tag] = rows
    for rd, key in (('pair', 'p_same'), ('letter', 'p_yes')):
        s = f1_stats(rows, rd, key)
        print(f'  {tag}/{rd}: F1={s["f1"]*100:.2f} P={s["p"]*100:.2f} R={s["r"]*100:.2f} n={s["n"]} (TP{s["tp"]} FP{s["fp"]} FN{s["fn"]} TN{s["tn"]})')
if len(os_rows) == 2:
    A, B = os_rows['27b'], os_rows['35b']
    # 双模型均值融合（p_same 平均）
    n = 0; tp = fp = fn = tn = 0
    for ra, rb in zip(A, B):
        if not (ra['pair'].get('ok') and rb['pair'].get('ok')):
            continue
        p = (ra['pair']['p_same'] + rb['pair']['p_same']) / 2
        pos = p >= 0.5
        if pos and ra['judgement'] == 'positive': tp += 1
        elif pos and ra['judgement'] == 'negative': fp += 1
        elif not pos and ra['judgement'] == 'positive': fn += 1
        else: tn += 1
        n += 1
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    print(f'  mean(pair) 2model: F1={2*pr*rc/(pr+rc)*100 if pr+rc else 0:.2f} P={pr*100:.2f} R={rc*100:.2f} n={n}')

print()
print('==================== spam-eval main (5,733) ====================')
print('参照: Jev enriched 98.64% / text 93.62% | TF-IDF regression enriched 98.87%')
pred = {}
for l in open(f'{BASE}/data/spam/pred_main.jsonl', encoding='utf-8'):
    r = json.loads(l); pred[r['file']] = r
lm = {'ham': 'legitimate', 'spam': 'spam', 'phish': 'phishing'}
# Jev 自身重算
jn = jc = 0
for f, r in pred.items():
    if r.get('jev_choice'):
        jn += 1
        jc += int(r['jev_choice'] == lm[r['label']])
print(f'  [jev/官方逐题] acc={jc/jn*100 if jn else 0:.2f}% ({jc}/{jn})')
sp_rows = {}
for tag in ('27b', '35b'):
    fp = f'{BASE}/results/spam_{tag}.jsonl'
    if not exists(fp):
        print(f'  [{tag}] (未完成)')
        continue
    rows = load(fp); sp_rows[tag] = rows
    for rd in ('letter', 'pair'):
        ok = [r for r in rows if r[rd].get('ok')]
        c = sum(1 for r in ok if r[rd]['pred'] == lm[pred[r['file']]['label']])
        print(f'  {tag}/{rd}: acc={c/max(1,len(ok))*100:.2f}% ({c}/{len(ok)})')
if len(sp_rows) == 2:
    A = {r['file']: r for r in sp_rows['27b']}
    B = {r['file']: r for r in sp_rows['35b']}
    n = c = 0
    for f in A:
        ra, rb = A[f], B.get(f)
        if not (ra['pair'].get('ok') and rb and rb['pair'].get('ok')):
            continue
        keys = ra['pair']['probs'].keys()
        mp = {k: (ra['pair']['probs'].get(k, 0) + rb['pair']['probs'].get(k, 0)) / 2 for k in keys}
        pred_cat = max(mp, key=mp.get)
        c += int(pred_cat == lm[pred[f]['label']]); n += 1
    print(f'  mean(pair) 2model: acc={c/max(1,n)*100:.2f}% ({c}/{n})')

print()
print('==================== RAG rerank ====================')
print('参照: scifact Jev 79.29 / NVIDIA 78.70 / 无重排 71.67 | xquad Jev 98.93 / NVIDIA 99.37 / 无重排 98.11')
for ds, ref in (('scifact', 'Jev 79.29 | NVIDIA 78.70 | base 71.67'), ('xquad', 'Jev 98.93 | NVIDIA 99.37 | base 98.11')):
    print(f'  -- {ds}（{ref}）--')
    rag_files = {}
    for tag in ('27b', '35b'):
        fp = f'{BASE}/results/rag_{ds}_{tag}.jsonl'
        if not exists(fp):
            print(f'    [{tag}] (未完成)')
            continue
        rows = load(fp); rag_files[tag] = {r['query_id']: r for r in rows}
        k = 'ours'
        n = len(rows)
        nd = sum(r[f'ndcg10_{k}'] for r in rows) / n * 100
        rc = sum(r[f'recall5_{k}'] for r in rows) / n * 100
        mr = sum(r[f'mrr10_{k}'] for r in rows) / n * 100
        nb = sum(r['ndcg10_base'] for r in rows) / n * 100
        print(f'    {tag}/ours: nDCG@10={nd:.2f} Recall@5={rc:.2f} MRR@10={mr:.2f} | base={nb:.2f} | n={n}')
    if len(rag_files) == 2:
        A, B = rag_files['27b'], rag_files['35b']
        def ndcg(order, gold):
            dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(order[:10], start=1) if d in gold)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
            return dcg / idcg if idcg else 0.0
        vals = []
        for qid, ra in A.items():
            rb = B.get(qid)
            if not rb:
                continue
            sc = {}
            for cand in ra['scores']:
                other = next((c for c in rb['scores'] if c['doc'] == cand['doc']), None)
                if other is None:
                    continue
                sc[cand['doc']] = cand['delta'] + other['delta']
            order = [d for d, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
            vals.append(ndcg(order, set(ra['gold'])))
        print(f'    mean(2model): nDCG@10={sum(vals)/len(vals)*100:.2f} (n={len(vals)})')