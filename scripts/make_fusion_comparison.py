#!/usr/bin/env python3
"""R2/R4 融合体的覆盖率对比（不采用单读数做对比）。
R4 = 双模型×双结构 四读数等权均值（fit-free；口径注記）。
R2 = 双模型一致性路由（各取主结构一次读数）：strict=一致且margin前50%，unanimous=一致，split=不一致。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HG = Path.home()
R = HG / 'hermes/projects/jev-general/results'
DB = HG / 'hermes/github_temp/decision-bench/results/bench-v4'
OUT = HG / 'hermes/projects/jev-general/reports/figs'


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def norm(d):
    s = sum(v for v in d.values() if v is not None)
    return {k: (v / s if v is not None else 0.0) for k, v in d.items()} if s else {k: 0.0 for k in d}


def mean_probs(ds):
    ds = [norm(d) for d in ds]
    keys = list(ds[0].keys())
    return {k: sum(d[k] for d in ds) / len(ds) for k in keys}


def curve(pairs):
    p = sorted([(c, k) for c, k in pairs if c is not None], key=lambda x: -x[0])
    n = len(p)
    corr = np.array([k for _, k in p], float)
    return np.arange(1, n + 1) / n, np.cumsum(corr) / np.arange(1, n + 1), corr.mean()


def cov_at(c, th):
    cov, cum, _ = c
    ok = cov[cum >= th]
    return float(ok.max()) if len(ok) else 0.0


# ---- 每套件的 R4/R2 抽取器 ----
def jevbench_rows():
    A = load(R / 'ours_27b_v2.jsonl'); B = load(R / 'ours_35b_v2.jsonl')
    Bi = {r['id']: r for r in B}
    out = []
    for a in A:
        b = Bi.get(a['id'])
        if not (b and all(a['readouts'][x].get('ok') for x in ('letter', 'pair')) and all(b['readouts'][x].get('ok') for x in ('letter', 'pair'))):
            continue
        out.append({'probs4': [a['readouts']['letter']['probs'], a['readouts']['pair']['probs'],
                               b['readouts']['letter']['probs'], b['readouts']['pair']['probs']],
                    'pa': a['readouts']['letter']['probs'], 'pb': b['readouts']['letter']['probs'],
                    'gold': a['expected']})
    return out, ('label',)

def typed_rows():
    A = load(R / 'typed_27b.jsonl'); B = load(R / 'typed_35b.jsonl')
    Bi = {(r['case_id'], r['qname']): r for r in B}
    out = []
    for a in A:
        b = Bi.get((a['case_id'], a['qname']))
        if not b:
            continue
        ok = all(a['readouts'][x].get('ok') for x in ('letter', 'pair')) and all(b['readouts'][x].get('ok') for x in ('letter', 'pair'))
        if not ok:
            continue
        out.append({'probs4': [a['readouts']['letter']['probs'], a['readouts']['pair']['probs'],
                               b['readouts']['letter']['probs'], b['readouts']['pair']['probs']],
                    'pa': a['readouts']['letter']['probs'], 'pb': b['readouts']['letter']['probs'],
                    'gold': a['opts'][a['gold_index']]})
    return out, ('label',)

def ab_rows(task):
    A = load(R / f'{task}_27b.jsonl'); B = load(R / f'{task}_35b.jsonl')
    Bi = {r['idx']: r for r in B}
    out = []
    for a in A:
        b = Bi.get(a['idx'])
        if not b:
            continue
        if not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        out.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                    'pa': a['letter']['probs'], 'pb': b['letter']['probs'],
                    'gold': a['labels'][a['gold_index']]})
    return out, ('label',)

def injection_rows():
    A = load(R / 'injection_27b.jsonl'); B = load(R / 'injection_35b.jsonl')
    Bi = {r['idx']: r for r in B}
    out = []
    for a in A:
        b = Bi.get(a['idx'])
        if not b:
            continue
        if not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        out.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                    'pa': a['pair']['probs'], 'pb': b['pair']['probs'],
                    'gold': a['labels'][a['gold_index']]})
    return out, ('label',)

def dbench_rows():
    A = load(R / 'dbench_27b.jsonl'); B = load(R / 'dbench_35b.jsonl')
    Bi = {r['id']: r for r in B}
    out = []
    for a in A:
        b = Bi.get(a['id'])
        if not b:
            continue
        if not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        out.append({'probs4': [a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']],
                    'pa': a['letter']['probs'], 'pb': b['letter']['probs'],
                    'gold': a['gold']})
    return out, ('label',)


SUITES = {
    'JevBench-231': jevbench_rows,
    'typed-decisions': typed_rows,
    'banking20': lambda: ab_rows('banking20'),
    'newsgroups': lambda: ab_rows('newsgroups'),
    'injection': injection_rows,
    'DecisionBench': dbench_rows,
}

def margin(p):
    s = sorted(p.values(), reverse=True)
    return (s[0] - s[1]) if len(s) > 1 else s[0]

results = {}
print(f'{"suite":16s} {"n":>5s} | R4: full  ≥98%   ≥95%   ≥93%  | R2 unanimous: cov@acc | R2 strict: cov@acc | split')
for name, fn in SUITES.items():
    rows, _ = fn()
    n = len(rows)
    # R4 曲线
    pairs4 = []
    for r in rows:
        mp = mean_probs(r['probs4'])
        pred = max(mp, key=mp.get)
        pairs4.append((max(mp.values()), int(pred == r['gold'])))
    c4 = curve(pairs4)
    # R2 门控
    def gate_rows():
        g = []
        for r in rows:
            pa, pb = r['pa'], r['pb']
            va, vb = max(pa, key=pa.get), max(pb, key=pb.get)
            agree = va == vb
            g.append({'agree': agree, 'margin': min(margin(pa), margin(pb)), 'correct': int(va == r['gold'])})
        return g
    g = gate_rows()
    un = [x for x in g if x['agree']]
    un_c = sum(x['correct'] for x in un) / max(1, len(un))
    # strict = 一致且 margin 前 50%（按 margin 排序取前半）
    un_sorted = sorted(un, key=lambda x: -x['margin'])
    strict = un_sorted[: len(un_sorted) // 2]
    st_c = sum(x['correct'] for x in strict) / max(1, len(strict))
    split = (n - len(un)) / n
    results[name] = {'c4': c4, 'n': n, 'un_cov': len(un) / n, 'un_acc': un_c, 'st_cov': len(strict) / n, 'st_acc': st_c, 'split': split}
    cov, cum, full = c4
    print(f'{name:16s} {n:5d} | {full*100:5.1f}% {cov_at(c4,0.98)*100:6.1f}% {cov_at(c4,0.95)*100:6.1f}% {cov_at(c4,0.93)*100:6.1f}% | '
          f'{len(un)/n*100:5.1f}%@{un_c*100:5.1f}% | {len(strict)/n*100:5.1f}%@{st_c*100:5.1f}% | {split*100:4.1f}%')

# ---- 图：R4 曲线 vs 官方（DecisionBench + JevBench）----
def official_curve(m):
    p = DB / m / 'predictions.jsonl'
    return curve([(r.get('confidence'), int(bool(r.get('correct')))) for r in load(p)])

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=200)
ax = axes[0]
cov, cum, full = results['DecisionBench']['c4']
ax.plot(cov, cum, lw=3.0, color='crimson', label=f'ours R4 (4 readings) ({full*100:.1f}%)')
for m, c, style in (('jev-1.13', 'k', '-.'), ('gemini-3.5-flash', 'tab:blue', '--'),
                    ('gpt-6-luna', 'tab:green', '--'), ('claude-sonnet-5', 'tab:purple', '--'),
                    ('qwen3-32b', 'tab:orange', '--'), ('laya-routed', 'tab:brown', '--')):
    cov, cum, full = official_curve(m)
    ax.plot(cov, cum, lw=1.5, ls=style, color=c, label=f'{m} ({full*100:.1f}%)')
r2 = results['DecisionBench']
ax.scatter([r2['un_cov']], [r2['un_acc']], s=70, marker='*', color='crimson', zorder=5, label=f"ours R2/unanimous ({r2['un_cov']*100:.1f}%@{r2['un_acc']*100:.1f}%)")
ax.scatter([r2['st_cov']], [r2['st_acc']], s=70, marker='^', color='darkred', zorder=5, label=f"ours R2/strict ({r2['st_cov']*100:.1f}%@{r2['st_acc']*100:.1f}%)")
ax.set_title('DecisionBench bench-v4 (1071) — ours R4/R2 vs official models')
ax.set_xlabel('coverage'); ax.set_ylabel('accuracy'); ax.set_xlim(0.2, 1.0); ax.grid(alpha=0.3)
ax.legend(fontsize=7.2, loc='lower left')

ax = axes[1]
cov, cum, full = results['JevBench-231']['c4']
ax.plot(cov, cum, lw=3.0, color='crimson', label=f'ours R4 ({full*100:.1f}%)')
cov, cum, full = official_curve('jev-1.13')
ax.plot(cov, cum, lw=1.8, ls='-.', color='k', label=f'JevBench-231 Jev ({full*100:.1f}%)')
r2 = results['JevBench-231']
ax.scatter([r2['un_cov']], [r2['un_acc']], s=70, marker='*', color='crimson', zorder=5, label=f"R2/unanimous ({r2['un_cov']*100:.1f}%@{r2['un_acc']*100:.1f}%)")
ax.scatter([r2['st_cov']], [r2['st_acc']], s=70, marker='^', color='darkred', zorder=5, label=f"R2/strict ({r2['st_cov']*100:.1f}%@{r2['st_acc']*100:.1f}%)")
ax.set_title('JevBench-231 — ours R4/R2 vs Jev')
ax.set_xlabel('coverage'); ax.set_xlim(0.2, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=7.2, loc='lower left')
fig.tight_layout()
fig.savefig(OUT / 'coverage_R4R2_vs_official.png'); fig.savefig(OUT / 'coverage_R4R2_vs_official.svg')
print()
print('figure ->', OUT / 'coverage_R4R2_vs_official.png')