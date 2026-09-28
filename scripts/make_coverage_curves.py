#!/usr/bin/env python3
"""jev-general: coverage–accuracy 曲线（成功率 vs 覆盖率）。
数据=本地 results/*.jsonl（只读）；输出 reports/figs/ 两图 + coverage_table.md。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path.home() / 'hermes/projects/jev-general/results'
OUT = Path.home() / 'hermes/projects/jev-general/reports/figs'
OUT.mkdir(parents=True, exist_ok=True)


def load(f):
    return [json.loads(l) for l in (R / f).read_text(encoding='utf-8').splitlines() if l.strip()]


def curve(pairs):
    """pairs: list of (confidence, correct01) -> (coverage array, acc array, full acc)."""
    p = sorted([(c, k) for c, k in pairs if c is not None], key=lambda x: -x[0])
    n = len(p)
    corr = np.array([k for _, k in p])
    cum = np.cumsum(corr) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    return cov, cum, cum[-1]


def build():
    sets = {}

    # 1) JevBench 231 — letter
    rows = load('ours_27b_v2.jsonl')
    sets['JevBench-231 · 27B'] = curve([(max(r['readouts']['letter']['probs'].values()), r['readouts']['letter']['correct'])
                                        for r in rows if r['readouts']['letter'].get('ok')])
    sets['JevBench-231 · Jev'] = curve([(max(r['probs'].values()), r['correct']) for r in load('jev_231_v2.jsonl')])
    rows = load('ours_35b_v2.jsonl')
    sets['JevBench-231 · 35B'] = curve([(max(r['readouts']['letter']['probs'].values()), r['readouts']['letter']['correct'])
                                        for r in rows if r['readouts']['letter'].get('ok')])

    # 2) typed-decisions 2000 — letter
    rows = load('typed_27b.jsonl')
    sets['typed-decisions-2000 · 27B'] = curve([(max(r['readouts']['letter']['probs'].values()),
                                                  int(r['readouts']['letter']['pred_idx'] == r['gold_index']))
                                                for r in rows if r['readouts']['letter'].get('ok')])
    rows = load('typed_35b.jsonl')
    sets['typed-decisions-2000 · 35B'] = curve([(max(r['readouts']['letter']['probs'].values()),
                                                  int(r['readouts']['letter']['pred_idx'] == r['gold_index']))
                                                for r in rows if r['readouts']['letter'].get('ok')])

    # 3) banking20 — letter
    for tag in ('27b', '35b'):
        rows = load(f'banking20_{tag}.jsonl')
        sets[f'banking20-300 · {tag.upper()}'] = curve([(max(r['letter']['probs'].values()), int(r['letter']['pred_index'] == r['gold_index']))
                                                        for r in rows if r['letter'].get('ok')])

    # 4) newsgroups — letter
    for tag in ('27b', '35b'):
        rows = load(f'newsgroups_{tag}.jsonl')
        sets[f'newsgroups-300 · {tag.upper()}'] = curve([(max(r['letter']['probs'].values()), int(r['letter']['pred_index'] == r['gold_index']))
                                                         for r in rows if r['letter'].get('ok')])

    # 5) injection — pair
    for tag in ('27b', '35b'):
        rows = load(f'injection_{tag}.jsonl')
        sets[f'injection-300 · {tag.upper()}'] = curve([(max(r['pair']['probs'].values()), int(r['pair']['pred_index'] == r['gold_index']))
                                                        for r in rows if r['pair'].get('ok')])

    # 6) DecisionBench — letter
    for tag in ('27b', '35b'):
        rows = load(f'dbench_{tag}.jsonl')
        sets[f'DecisionBench-1071 · {tag.upper()}'] = curve([(max(r['letter']['probs'].values()), int(r['letter']['pred'] == r['gold']))
                                                            for r in rows if r['letter'].get('ok')])

    # 7) OpenSanctions — letter（27B 已完成）
    rows = load('osbench_27b.jsonl')
    sets['OpenSanctions-9800 · 27B'] = curve([(max(r['letter']['p_yes'], 1 - r['letter']['p_yes']), int((r['letter']['p_yes'] >= 0.5) == (r['judgement'] == 'positive')))
                                              for r in rows if r['letter'].get('ok')])
    return sets


sets = build()

# ---------- 图 1：主图（27B 主力曲线）----------
main_keys = [k for k in sets if k.endswith('· 27B')]
fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
colors = plt.cm.tab10(np.linspace(0, 1, len(main_keys)))
for c, k in zip(colors, sorted(main_keys)):
    cov, acc, full = sets[k]
    ax.plot(cov, acc, lw=1.9, color=c, label=f"{k}  (full={full*100:.1f}%)")
if 'JevBench-231 · Jev' in sets:
    cov, acc, full = sets['JevBench-231 · Jev']
    ax.plot(cov, acc, lw=1.9, ls='--', color='k', label=f"JevBench-231 · Jev (full={full*100:.1f}%)")
ax.set_xlabel('Coverage (fraction of items answered, most-confident first)')
ax.set_ylabel('Accuracy on answered subset')
ax.set_title('Coverage–accuracy trade-off — jev-general benchmarks (zero-training, local 24GB, np=1)')
ax.set_xlim(0.1, 1.0); ax.grid(alpha=0.3)
ax.legend(fontsize=8.2, loc='lower left', framealpha=0.9)
fig.tight_layout(); fig.savefig(OUT / 'coverage_accuracy_jev_general.png')
fig.savefig(OUT / 'coverage_accuracy_jev_general.svg')

# ---------- 图 2：分面 ----------
suites = [
    ('JevBench-231', ['JevBench-231 · 27B', 'JevBench-231 · 35B', 'JevBench-231 · Jev']),
    ('typed-decisions-2000', ['typed-decisions-2000 · 27B', 'typed-decisions-2000 · 35B']),
    ('banking20-300', ['banking20-300 · 27B', 'banking20-300 · 35B']),
    ('newsgroups-300', ['newsgroups-300 · 27B', 'newsgroups-300 · 35B']),
    ('injection-300', ['injection-300 · 27B', 'injection-300 · 35B']),
    ('DecisionBench-1071', ['DecisionBench-1071 · 27B', 'DecisionBench-1071 · 35B']),
    ('OpenSanctions-9800', ['OpenSanctions-9800 · 27B']),
]
fig2, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), dpi=200, sharey=False)
for ax, (name, keys) in zip(axes.ravel(), suites):
    for k in keys:
        cov, acc, full = sets[k]
        ls = '--' if 'Jev' in k else '-'
        ax.plot(cov, acc, lw=1.6, ls=ls, label=f"{k.split('· ')[1]} ({full*100:.1f}%)")
    ax.set_title(name, fontsize=10)
    ax.set_xlim(0.2, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc='lower left')
    ax.set_xlabel('coverage'); 
axes[1][3].axis('off')
axes[1][3].text(0.5, 0.5, 'spam / RAG suites\n(pending completion)', ha='center', va='center', fontsize=10, color='gray')
fig2.suptitle('Coverage–accuracy by suite (27B / 35B readers; Jev reference where available)', y=1.02)
fig2.tight_layout(); fig2.savefig(OUT / 'coverage_by_suite.png')
fig2.savefig(OUT / 'coverage_by_suite.svg')

# ---------- 覆盖@精度档表 ----------
lines = ['# 覆盖率 @ 精度档（27B 主力读数；max-prob 置信度排序）', '',
         '| 套件 | 全量 acc | ≥98% 覆盖 | ≥95% 覆盖 | ≥93% 覆盖 |', '|---|---|---|---|---|']
for k in sorted(main_keys):
    cov, acc, full = sets[k]
    def cov_at(th):
        ok = cov[acc >= th]
        return f"{ok.max()*100:.1f}%" if len(ok) else '—'
    lines.append(f"| {k.split(' · ')[0]} | {full*100:.1f}% | {cov_at(0.98)} | {cov_at(0.95)} | {cov_at(0.93)} |")
(Path.home() / 'hermes/projects/jev-general/reports/coverage_table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print()
print('figures ->', OUT / 'coverage_accuracy_jev_general.png', '|', OUT / 'coverage_by_suite.png')