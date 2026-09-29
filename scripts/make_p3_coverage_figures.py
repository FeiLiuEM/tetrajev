#!/usr/bin/env python3
"""P3 覆盖率-准确率曲线：① 八套件 R4 主图；② osbench/spam 细图（组件+35B+R2）。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path.home() / 'hermes/projects/jev-general/results'
OUT = Path.home() / 'hermes/projects/jev-general/reports/figs'

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def norm(d):
    s = sum(v for v in d.values() if v is not None)
    return {k: (v / s if v is not None else 0.0) for k, v in d.items()} if s else {k: 0.0 for k in d}

def mean_probs(ds):
    ds = [norm(d) for d in ds]
    return {k: sum(d[k] for d in ds) / len(ds) for k in ds[0]}

def curve(pairs):
    p = sorted([(c, k) for c, k in pairs if c is not None], key=lambda x: -x[0])
    n = len(p)
    corr = np.array([k for _, k in p], float)
    return np.arange(1, n + 1) / n, np.cumsum(corr) / np.arange(1, n + 1), corr.mean()

def cov_at(c, th):
    cov, cum, _ = c
    ok = cov[cum >= th]
    return float(ok.max()) if len(ok) else 0.0

def margin(p):
    s = sorted(p.values(), reverse=True)
    return (s[0] - s[1]) if len(s) > 1 else s[0]

# ================= 各套件 R4 抽取（8 套件） =================
def jevbench():
    A = load(R / 'ours_27b_v2.jsonl'); B = {r['id']: r for r in load(R / 'ours_35b_v2.jsonl')}
    rows = []
    for a in A:
        b = B.get(a['id'])
        if not (b and all(a['readouts'][x].get('ok') for x in ('letter', 'pair')) and all(b['readouts'][x].get('ok') for x in ('letter', 'pair'))):
            continue
        rows.append({'probs4': [a['readouts']['letter']['probs'], a['readouts']['pair']['probs'], b['readouts']['letter']['probs'], b['readouts']['pair']['probs']],
                     'pa': a['readouts']['letter']['probs'], 'pb': b['readouts']['letter']['probs'], 'gold': a['expected']})
    return rows

def typed():
    A = load(R / 'typed_27b.jsonl'); B = {(r['case_id'], r['qname']): r for r in load(R / 'typed_35b.jsonl')}
    rows = []
    for a in A:
        b = B.get((a['case_id'], a['qname']))
        if not b or not all(a['readouts'][x].get('ok') for x in ('letter', 'pair')) or not all(b['readouts'][x].get('ok') for x in ('letter', 'pair')):
            continue
        rows.append({'probs4': [a['readouts']['letter']['probs'], a['readouts']['pair']['probs'], b['readouts']['letter']['probs'], b['readouts']['pair']['probs']],
                     'pa': a['readouts']['letter']['probs'], 'pb': b['readouts']['letter']['probs'], 'gold': a['opts'][a['gold_index']]})
    return rows

def ab(task):
    A = load(R / f'{task}_27b.jsonl'); B = {r['idx']: r for r in load(R / f'{task}_35b.jsonl')}
    rows = []
    for a in A:
        b = B.get(a['idx'])
        if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        rows.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                     'pa': a['letter']['probs'], 'pb': b['letter']['probs'], 'gold': a['labels'][a['gold_index']]})
    return rows

def injection():
    A = load(R / 'injection_27b.jsonl'); B = {r['idx']: r for r in load(R / 'injection_35b.jsonl')}
    rows = []
    for a in A:
        b = B.get(a['idx'])
        if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        rows.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                     'pa': a['pair']['probs'], 'pb': b['pair']['probs'], 'gold': a['labels'][a['gold_index']]})
    return rows

def dbench():
    A = load(R / 'dbench_27b.jsonl'); B = {r['id']: r for r in load(R / 'dbench_35b.jsonl')}
    rows = []
    for a in A:
        b = B.get(a['id'])
        if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        rows.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                     'pa': a['letter']['probs'], 'pb': b['letter']['probs'], 'gold': a['gold']})
    return rows

def osbench():
    A = load(R / 'osbench_27b.jsonl'); B = {r['i']: r for r in load(R / 'osbench_35b.jsonl')}
    rows = []
    for a in A:
        b = B.get(a['i'])
        if not b:
            continue
        def dist(r):
            return {'same': r['letter']['p_yes'], 'diff': 1 - r['letter']['p_yes']} if r['letter'].get('ok') else None
        def dist_pair(r):
            return {'same': r['pair']['p_same'], 'diff': 1 - r['pair']['p_same']} if r['pair'].get('ok') else None
        d = [dist(a), dist_pair(a), dist(b), dist_pair(b)]
        if any(x is None for x in d):
            continue
        rows.append({'probs4': d, 'pa': dist(a), 'pb': dist(b),
                     'gold': 'same' if a['judgement'] == 'positive' else 'diff'})
    return rows

def spam():
    A = load(R / 'spam_27b.jsonl'); B = {r['file']: r for r in load(R / 'spam_35b.jsonl')}
    LM = {'ham': 'legitimate', 'spam': 'spam', 'phish': 'phishing'}
    rows = []
    for a in A:
        b = B.get(a['file'])
        if not b:
            continue
        if not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        rows.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                     'pa': a['letter']['probs'], 'pb': b['letter']['probs'], 'gold': LM[a['label']]})
    return rows

SUITES = {
    'JevBench-231': jevbench, 'typed-decisions': typed, 'banking20': lambda: ab('banking20'),
    'newsgroups': lambda: ab('newsgroups'), 'injection': injection, 'DecisionBench': dbench,
    'OpenSanctions': osbench, 'spam-eval': spam,
}

stat = {}
print(f'{"suite":16s} {"n":>6s} | R4 full | cov@98 cov@95 cov@93 | R2: unan cov@acc / strict cov@acc')
for name, fn in SUITES.items():
    rows = fn()
    pairs4 = []
    for r in rows:
        mp = mean_probs(r['probs4'])
        pred = max(mp, key=mp.get)
        pairs4.append((max(mp.values()), int(pred == r['gold'])))
    c4 = curve(pairs4)
    g = []
    for r in rows:
        pa, pb = r['pa'], r['pb']
        va, vb = max(pa, key=pa.get), max(pb, key=pb.get)
        g.append({'agree': va == vb, 'margin': min(margin(pa), margin(pb)), 'correct': int(va == r['gold'])})
    un = [x for x in g if x['agree']]
    un_c = sum(x['correct'] for x in un) / max(1, len(un))
    un_sorted = sorted(un, key=lambda x: -x['margin'])
    strict = un_sorted[: len(un_sorted) // 2]
    st_c = sum(x['correct'] for x in strict) / max(1, len(strict))
    stat[name] = {'c4': c4, 'n': len(rows), 'un_cov': len(un) / len(rows), 'un_acc': un_c, 'st_cov': len(strict) / len(rows), 'st_acc': st_c}
    cov, cum, full = c4
    print(f'{name:16s} {len(rows):6d} | {full*100:6.2f}% | {cov_at(c4,0.98)*100:6.1f} {cov_at(c4,0.95)*100:6.1f} {cov_at(c4,0.93)*100:6.1f} | {stat[name]["un_cov"]*100:5.1f}%@{un_c*100:5.1f} / {stat[name]["st_cov"]*100:5.1f}%@{st_c*100:5.1f}')

# ================= 主图：八套件 R4 =================
fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=200)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
for (name, s), c in zip(stat.items(), colors):
    cov, cum, full = s['c4']
    ax.plot(cov, cum, lw=2.2, color=c, label=f'{name} (n={s["n"]}, R4 {full*100:.1f}%)')
ax.axhline(0.95, color='gray', ls=':', lw=1)
ax.text(0.012, 0.952, '5% risk', fontsize=7, color='gray')
ax.set_title('TetraJev R4 (four-reading fusion) — coverage vs accuracy, eight decision suites', fontsize=10)
ax.set_xlabel('coverage (fraction answered)'); ax.set_ylabel('accuracy')
ax.set_xlim(0, 1.0); ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.3)
ax.legend(fontsize=7.6, loc='lower left')
fig.tight_layout()
fig.savefig(OUT / 'coverage_r4_all_suites.png'); fig.savefig(OUT / 'coverage_r4_all_suites.svg')
print('->', OUT / 'coverage_r4_all_suites.png')

# ================= 细图：osbench + spam =================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=200)
# osbench
ax = axes[0]
A = load(R / 'osbench_27b.jsonl'); B = {r['i']: r for r in load(R / 'osbench_35b.jsonl')}
def binconf(r, rd):
    if rd == 'letter': return max(r['letter']['p_yes'], 1 - r['letter']['p_yes']) if r['letter'].get('ok') else None
    return max(r['pair']['p_same'], 1 - r['pair']['p_same']) if r['pair'].get('ok') else None
def bincorr(r, rd, gold_key):
    p = r[rd][gold_key]
    return int((p >= 0.5) == (r['judgement'] == 'positive'))
comps = [('27B letter', 'tab:blue', '-'), ('27B pair', 'tab:cyan', '-'), ('35B letter', 'tab:orange', '-'), ('35B pair', 'tab:brown', '-')]
for tag, rows, rdm, key in (('27B', A, 'letter', 'p_yes'), ('27B', A, 'pair', 'p_same'), ('35B', None, 'letter', 'p_yes'), ('35B', None, 'pair', 'p_same')):
    pass
for (lab, c, ls), (rows, rd) in zip(comps, ((A, 'letter'), (A, 'pair'), (list(B.values()), 'letter'), (list(B.values()), 'pair'))):
    pairs = [(binconf(r, rd), bincorr(r, rd, 'p_yes' if rd == 'letter' else 'p_same')) for r in rows]
    cov, cum, full = curve(pairs)
    ax.plot(cov, cum, lw=1.5, ls=ls, color=c, label=f'{lab} ({full*100:.1f}%)')
# R4
pairs4 = []
for a in A:
    b = B.get(a['i'])
    if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
        continue
    pf = (a['letter']['p_yes'] + a['pair']['p_same'] + b['letter']['p_yes'] + b['pair']['p_same']) / 4
    pairs4.append((max(pf, 1 - pf), int((pf >= 0.5) == (a['judgement'] == 'positive'))))
cov, cum, full = curve(pairs4)
ax.plot(cov, cum, lw=3.0, color='crimson', label=f'R4 fused (4 readings) ({full*100:.1f}%)')
# R2（letter 一致）
ag = [(a, B[a['i']]) for a in A if a['i'] in B and a['letter'].get('ok') and B[a['i']]['letter'].get('ok')]
same = [(a, b) for a, b in ag if (a['letter']['p_yes'] >= 0.5) == (b['letter']['p_yes'] >= 0.5)]
acc_s = sum(1 for a, b in same if (a['letter']['p_yes'] >= 0.5) == (a['judgement'] == 'positive')) / max(1, len(same))
ax.scatter([len(same) / len(ag)], [acc_s], s=90, marker='*', color='crimson', edgecolors='k', linewidths=0.6, zorder=5,
           label=f'R2/unanimous ({len(same)/len(ag)*100:.1f}%@{acc_s*100:.1f}%)')
ax.set_title('OpenSanctions pairs (9,800) — components, R4, R2')
ax.set_xlabel('coverage'); ax.set_ylabel('accuracy'); ax.set_xlim(0, 1); ax.grid(alpha=0.3)
ax.legend(fontsize=7.4, loc='lower left')
# spam
ax = axes[1]
S27 = load(R / 'spam_27b.jsonl'); S35 = {r['file']: r for r in load(R / 'spam_35b.jsonl')}
LM = {'ham': 'legitimate', 'spam': 'spam', 'phish': 'phishing'}
def spamconf(r, rd):
    if not r[rd].get('ok'): return None
    return max(r[rd]['probs'].values())
def spamcorr(r, rd):
    return int(r[rd]['pred'] == LM[r['label']])
for (lab, rows, rd, c, ls) in (('27B letter', S27, 'letter', 'tab:blue', '-'), ('27B pair', S27, 'pair', 'tab:cyan', '-'),
                               ('35B letter', list(S35.values()), 'letter', 'tab:orange', '-'), ('35B pair', list(S35.values()), 'pair', 'tab:brown', '-')):
    pairs = [(spamconf(r, rd), spamcorr(r, rd)) for r in rows]
    cov, cum, full = curve(pairs)
    ax.plot(cov, cum, lw=1.5, ls=ls, color=c, label=f'{lab} ({full*100:.1f}%)')
pairs4 = []
for a in S27:
    b = S35.get(a['file'])
    if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
        continue
    mp = mean_probs([a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']])
    pred = max(mp, key=mp.get)
    pairs4.append((max(mp.values()), int(pred == LM[a['label']])))
cov, cum, full = curve(pairs4)
ax.plot(cov, cum, lw=3.0, color='crimson', label=f'R4 fused (4 readings) ({full*100:.1f}%)')
# R2（pair 一致 + letter 一致）
for rd, marker, lab in (('pair', '*', 'R2/unanimous (pair)'), ('letter', '^', 'R2/unanimous (letter)')):
    ag = [(a, S35[a['file']]) for a in S27 if a['file'] in S35 and a[rd].get('ok') and S35[a['file']][rd].get('ok')]
    sm = [(a, b) for a, b in ag if a[rd]['pred'] == b[rd]['pred']]
    acc_s = sum(1 for a, b in sm if a[rd]['pred'] == LM[a['label']]) / max(1, len(sm))
    ax.scatter([len(sm) / len(ag)], [acc_s], s=90, marker=marker, color='darkred' if rd == 'pair' else 'darkorange', edgecolors='k', linewidths=0.6, zorder=5,
               label=f'{lab} ({len(sm)/len(ag)*100:.1f}%@{acc_s*100:.1f}%)')
ax.set_title('spam-eval (5,733) — components, R4, R2')
ax.set_xlabel('coverage'); ax.set_xlim(0, 1); ax.grid(alpha=0.3)
ax.legend(fontsize=7.4, loc='lower left')
fig.tight_layout()
fig.savefig(OUT / 'coverage_p3_osbench_spam.png'); fig.savefig(OUT / 'coverage_p3_osbench_spam.svg')
print('->', OUT / 'coverage_p3_osbench_spam.png')
