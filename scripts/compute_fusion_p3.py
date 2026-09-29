#!/usr/bin/env python3
"""P3 融合层计算：osbench / spam / RAG(两集) 的 35B 单读数 + R4 融合。"""
import json, math, pathlib

R = pathlib.Path.home() / 'hermes/projects/jev-general/results'

def load(name):
    return [json.loads(l) for l in (R / name).read_text(encoding='utf-8').splitlines() if l.strip()]

# ---------- osbench ----------
o27, o35 = load('osbench_27b.jsonl'), load('osbench_35b.jsonl')
assert len(o27) == len(o35) == 9800

def f1_bin(rows_probs):
    tp = fp = fn = 0
    for r, p in rows_probs:
        pred = p >= 0.5
        pos = r['judgement'] == 'positive'
        if pred and pos: tp += 1
        elif pred and not pos: fp += 1
        elif (not pred) and pos: fn += 1
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    return 100 * 2 * prec * rec / (prec + rec) if prec + rec else 0

def comp(rows, rd, key):
    return [(r, r[rd][key]) for r in rows if r[rd].get('ok')]

print('== osbench (F1, n=9800) ==')
print(f"  27b letter {f1_bin(comp(o27,'letter','p_yes')):.2f} | 27b pair {f1_bin(comp(o27,'pair','p_same')):.2f}")
print(f"  35b letter {f1_bin(comp(o35,'letter','p_yes')):.2f} | 35b pair {f1_bin(comp(o35,'pair','p_same')):.2f}")
fused4 = [(a, sum([a['letter']['p_yes'], a['pair']['p_same'], b['letter']['p_yes'], b['pair']['p_same']]) / 4)
          for a, b in zip(o27, o35) if a['letter'].get('ok') and a['pair'].get('ok') and b['letter'].get('ok') and b['pair'].get('ok')]
fused2 = [(a, (a['letter']['p_yes'] + b['letter']['p_yes']) / 2) for a, b in zip(o27, o35)
          if a['letter'].get('ok') and b['letter'].get('ok')]
print(f"  R4 fused(4) F1={f1_bin(fused4):.2f} (n={len(fused4)}) | 2-model-letter F1={f1_bin(fused2):.2f}")
# R2 一致性（两模型 letter 判定一致）
agree = [(a, b) for a, b in zip(o27, o35) if a['letter'].get('ok') and b['letter'].get('ok')]
same = [x for x in agree if (x[0]['letter']['p_yes'] >= 0.5) == (x[1]['letter']['p_yes'] >= 0.5)]
acc_same = sum(1 for a, b in same if (a['letter']['p_yes'] >= 0.5) == (a['judgement'] == 'positive')) / len(same)
print(f"  R2/unanimous: coverage {len(same)/len(agree)*100:.1f}% @ acc(F1单类近似) {acc_same*100:.2f}% / split {100-len(same)/len(agree)*100:.1f}%")

# ---------- spam ----------
s27, s35 = load('spam_27b.jsonl'), load('spam_35b.jsonl')
LM = {'ham': 'legitimate', 'spam': 'spam', 'phish': 'phishing'}
print('\n== spam (acc, n=%d) ==' % len(s27))
def acc_rows(rows, rd):
    ok = [r for r in rows if r[rd].get('ok')]
    good = sum(1 for r in ok if r[rd]['pred'] == LM[r['label']])
    return good / len(ok) * 100, len(ok)
for tag, rows in (('27b', s27), ('35b', s35)):
    al, nl = acc_rows(rows, 'letter'); ap, np_ = acc_rows(rows, 'pair')
    print(f"  {tag}: letter {al:.2f} (n={nl}) | pair {ap:.2f} (n={np_})")
by27 = {r['file']: r for r in s27}; by35 = {r['file']: r for r in s35}
fused4_rows = []
for f, a in by27.items():
    b = by35.get(f)
    if not b or not all(x.get('ok') for x in (a['letter'], a['pair'], b['letter'], b['pair'])):
        continue
    keys = set(a['letter']['probs']) | set(b['letter']['probs'])
    p = {k: (a['letter']['probs'].get(k, 0) + a['pair']['probs'].get(k, 0) + b['letter']['probs'].get(k, 0) + b['pair']['probs'].get(k, 0)) / 4 for k in keys}
    pred = max(p, key=p.get)
    fused4_rows.append((f, LM[a['label']], pred, p))
good = sum(1 for _, lb, pr, _ in fused4_rows if pr == lb)
print(f"  R4 fused acc {good/len(fused4_rows)*100:.2f} (n={len(fused4_rows)})")
# 与 Jev 配对（fused vs jev）
import pathlib as _pl
pred = {}
for l in open(_pl.Path.home() / 'hermes/projects/jev-general/data_local/pred_main.jsonl', encoding='utf-8'):
    x = json.loads(l); pred[x['file']] = x
both = ours = jev = 0
for f, lb, pr, _ in fused4_rows:
    jok = pred[f].get('jev_choice') == lb
    oo = pr == lb
    if oo and jok: both += 1
    elif oo: ours += 1
    elif jok: jev += 1
n = len(fused4_rows); jacc = (both + jev) / n * 100; oacc = (both + ours) / n * 100
print(f"  R4 vs Jev 配对: ours {oacc:.2f} vs Jev {jacc:.2f} (both {both} | ours-only {ours} | jev-only {jev})")

# ---------- RAG ----------
def ndcg(order, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(order[:k], 1) if d in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0
def recall5(order, gold):
    return len({d for d in order[:5] if d in gold}) / max(1, len(gold))
def mrr10(order, gold):
    for i, d in enumerate(order[:10], 1):
        if d in gold: return 1 / i
    return 0
print('\n== RAG ==')
for ds, f27, f35 in (('scifact', 'rag_scifact_27b.jsonl', 'rag_scifact_35b.jsonl'),
                     ('xquad-en', 'rag_xquad_27b.jsonl', 'rag_xquad_35b.jsonl')):
    r27, r35 = load(f27), load(f35)
    for tag, rs in (('27b', r27), ('35b', r35)):
        nd = sum(r['ndcg10_ours'] for r in rs) / len(rs) * 100
        rc = sum(r['recall5_ours'] for r in rs) / len(rs) * 100
        mr = sum(r['mrr10_ours'] for r in rs) / len(rs) * 100
        print(f"  {ds} {tag}: nDCG@10 {nd:.2f} | R@5 {rc:.2f} | MRR {mr:.2f} (n={len(rs)})")
    # 融合（delta 平均）
    by = {r['query_id']: r for r in r35}
    nd = rc = mr = 0; nn = 0
    for a in r27:
        b = by.get(a['query_id'])
        if not b: continue
        da = {c['doc']: c['delta'] for c in a['scores']}
        db = {c['doc']: c['delta'] for c in b['scores']}
        docs = [c['doc'] for c in a['scores']]
        rrf = {c['doc']: c['rrf'] for c in a['scores']}
        fused = {d: (da.get(d, 0) + db.get(d, 0)) / 2 for d in docs}
        order = sorted(docs, key=lambda d: (-fused[d], rrf[d]))
        gold = set(a['gold'])
        nd += ndcg(order, gold); rc += recall5(order, gold); mr += mrr10(order, gold); nn += 1
    print(f"  {ds} R4-fused(2 models): nDCG@10 {nd/nn*100:.2f} | R@5 {rc/nn*100:.2f} | MRR {mr/nn*100:.2f} (n={nn})")
