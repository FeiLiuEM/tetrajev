#!/usr/bin/env python3
"""覆盖率对比：我们 vs DecisionBench 官方 14 模型 vs JevBench 的 Jev vs anyjev cov@5%。"""
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


def curve(pairs):
    p = sorted([(c, k) for c, k in pairs if c is not None], key=lambda x: -x[0])
    n = len(p)
    if n == 0:
        return None
    corr = np.array([k for _, k in p], dtype=float)
    cum = np.cumsum(corr) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    return cov, cum, cum[-1]


def cov_at(c, th):
    cov, cum, _ = c
    ok = cov[cum >= th]
    return float(ok.max()) if len(ok) else 0.0


print('=' * 80)
print('【DecisionBench bench-v4：覆盖率 vs 官方模型（1071 行，同题）】')
print(f'{"model":28s} {"full":>6s} {"≥98%":>7s} {"≥95%":>7s} {"≥93%":>7s} {"≥90%":>7s} {"cov@5%":>7s}')
rows = []
# 我们
for tag in ('27b', '35b'):
    c = curve([(max(r['letter']['probs'].values()), int(r['letter']['pred'] == r['gold']))
               for r in load(R / f'dbench_{tag}.jsonl') if r['letter'].get('ok')])
    rows.append((f'ours-{tag}/letter', c))
# 官方
off = ['jev-1.13', 'gemini-3.5-flash', 'gemini-flash-lite-latest', 'gpt-6-luna', 'claude-sonnet-5',
       'deepseek-v4.1-flash', 'glm-5.3-flash', 'gpt-5.6-luna', 'claude-haiku-4.5',
       'qwen3-32b', 'laya-routed', 'nova-micro-v1', 'sage', 'tev1-4b-experimental']
for m in off:
    p = DB / m / 'predictions.jsonl'
    if not p.exists():
        print(f'{m:28s} (缺文件)')
        continue
    c = curve([(r.get('confidence'), int(bool(r.get('correct')))) for r in load(p)])
    rows.append((m, c))
for name, c in rows:
    if c is None:
        continue
    cov, cum, full = c
    print(f'{name:28s} {full*100:5.1f}% {cov_at(c,0.98)*100:6.1f}% {cov_at(c,0.95)*100:6.1f}% {cov_at(c,0.93)*100:6.1f}% {cov_at(c,0.90)*100:6.1f}% {cov_at(c,0.95)*100:6.1f}%')

# 图：dbench 曲线（选 8 条）
pick = ['ours-27b/letter', 'ours-35b/letter', 'jev-1.13', 'gemini-3.5-flash', 'gpt-6-luna',
        'claude-sonnet-5', 'qwen3-32b', 'laya-routed']
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=200)
ax = axes[0]
for name, c in rows:
    if name not in pick or c is None:
        continue
    cov, cum, full = c
    lw = 2.6 if name.startswith('ours') else 1.5
    ls = '-' if name.startswith('ours') else ('--' if name != 'jev-1.13' else '-.')
    ax.plot(cov, cum, lw=lw, ls=ls, label=f'{name} ({full*100:.1f}%)')
ax.set_title('DecisionBench bench-v4 (1071) — coverage vs official models')
ax.set_xlabel('coverage'); ax.set_ylabel('accuracy'); ax.set_xlim(0.2, 1.0)
ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc='lower left')

# JevBench panel
ax = axes[1]
jb = []
for tag in ('27b', '35b'):
    c = curve([(max(r['readouts']['letter']['probs'].values()), r['readouts']['letter']['correct'])
               for r in load(R / f'ours_{tag}_v2.jsonl') if r['readouts']['letter'].get('ok')])
    jb.append((f'ours-{tag}/letter', c))
c = curve([(max(r['probs'].values()), r['correct']) for r in load(R / 'jev_231_v2.jsonl')])
jb.append(('jev-1.13 (native)', c))
for name, c in jb:
    cov, cum, full = c
    lw = 2.6 if name.startswith('ours') else 1.8
    ax.plot(cov, cum, lw=lw, ls='-' if name.startswith('ours') else '-.', label=f'{name} ({full*100:.1f}%)')
ax.set_title('JevBench-231 — coverage vs Jev')
ax.set_xlabel('coverage'); ax.set_xlim(0.2, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc='lower left')
fig.tight_layout()
fig.savefig(OUT / 'coverage_vs_official.png'); fig.savefig(OUT / 'coverage_vs_official.svg')
print()
print('figure ->', OUT / 'coverage_vs_official.png')

print()
print('=' * 80)
print('【anyjev 套件：cov@5%（覆盖率@风险≤5%）——我们 vs 其已发布表】')
# 我们的
ours = {}
for task, rd in (('banking20', 'letter'), ('newsgroups', 'letter'), ('injection', 'pair')):
    c = curve([(max(r[rd]['probs'].values()), int(r[rd]['pred_index'] == r['gold_index']))
               for r in load(R / f'{task}_27b.jsonl') if r[rd].get('ok')])
    ours[task] = (cov_at(c, 0.95), c[2])
for task, (cv, full) in ours.items():
    print(f'  ours-27B {task:12s}: cov@5%={cv*100:.1f}%  (full acc {full*100:.1f}%)')
print('  对照（anyjev 发布）: banking20: 8B-L1 52.0% / 8B-L0 46.3% / 32B-L0 39.0% | newsgroups: 32B-L0 47.0% / 8B-L0 42.3% | injection: 8B-L0 1.3% / 32B-L0 48.3%')