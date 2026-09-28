#!/usr/bin/env python3
"""R4 vs 其他模型：成功率(accuracy)-覆盖率 图（三面板）。
A: DecisionBench vs 官方 6 家 + R2 操作点；B: JevBench vs Jev；C: anyjev 套件（R4）vs 其发布最好 cov@5%。
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
plt.rcParams.update({'font.size': 10.5})


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


def official_curve(m):
    return curve([(r.get('confidence'), int(bool(r.get('correct')))) for r in load(DB / m / 'predictions.jsonl')])


def margin(p):
    s = sorted(p.values(), reverse=True)
    return (s[0] - s[1]) if len(s) > 1 else s[0]


def r2_points(rows_pa_pb):
    g = []
    for pa, pb, gold in rows_pa_pb:
        va, vb = max(pa, key=pa.get), max(pb, key=pb.get)
        g.append({'agree': va == vb, 'margin': min(margin(pa), margin(pb)), 'correct': int(va == gold)})
    un = [x for x in g if x['agree']]
    un_s = sorted(un, key=lambda x: -x['margin'])
    st = un_s[: len(un_s) // 2]
    return ((len(un) / len(g), sum(x['correct'] for x in un) / max(1, len(un))),
            (len(st) / len(g), sum(x['correct'] for x in st) / max(1, len(st))))


# ---- 数据装配 ----
# DecisionBench
A27 = load(R / 'dbench_27b.jsonl'); A35 = load(R / 'dbench_35b.jsonl')
B35 = {r['id']: r for r in A35}
rows4, rows_r2 = [], []
for a in A27:
    b = B35.get(a['id'])
    if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
        continue
    rows4.append(mean_probs([a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']]))
    rows_r2.append((a['letter']['probs'], b['letter']['probs'], a['gold']))
db_pairs = [(max(mp.values()), int(max(mp, key=mp.get) == r['gold'])) for mp, r in zip(rows4, [r for r in A27 if r['id'] in B35])]
# 修正：上面 zip 需同序过滤——重算一次（稳健写法）
db_pairs = []
for a in A27:
    b = B35.get(a['id'])
    if not b or not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
        continue
    mp = mean_probs([a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']])
    db_pairs.append((max(mp.values()), int(max(mp, key=mp.get) == a['gold'])))
c_db = curve(db_pairs)
un, st = r2_points(rows_r2)

# JevBench
J27 = load(R / 'ours_27b_v2.jsonl'); J35 = load(R / 'ours_35b_v2.jsonl')
J35i = {r['id']: r for r in J35}
jb_pairs, jb_r2 = [], []
for a in J27:
    b = J35i.get(a['id'])
    if not b or not all(a['readouts'][x].get('ok') for x in ('letter', 'pair')) or not all(b['readouts'][x].get('ok') for x in ('letter', 'pair')):
        continue
    mp = mean_probs([a['readouts']['letter']['probs'], a['readouts']['pair']['probs'],
                     b['readouts']['letter']['probs'], b['readouts']['pair']['probs']])
    jb_pairs.append((max(mp.values()), int(max(mp, key=mp.get) == a['expected'])))
    jb_r2.append((a['readouts']['letter']['probs'], b['readouts']['letter']['probs'], a['expected']))
c_jb = curve(jb_pairs)
jb_un, jb_st = r2_points(jb_r2)
c_jev = curve([(max(r['probs'].values()), r['correct']) for r in load(R / 'jev_231_v2.jsonl')])

# anyjev 套件（R4）
def suite_r4(task, pa_key='letter'):
    A = load(R / f'{task}_27b.jsonl'); B = load(R / f'{task}_35b.jsonl')
    Bi = {r['idx']: r for r in B}
    pairs = []
    for a in A:
        b = Bi.get(a['idx'])
        if not b:
            continue
        if not all(a[x].get('ok') for x in ('letter', 'pair')) or not all(b[x].get('ok') for x in ('letter', 'pair')):
            continue
        mp = mean_probs([a['letter']['probs'], a['pair']['probs'], b['letter']['probs'], b['pair']['probs']])
        pairs.append((max(mp.values()), int(max(mp, key=mp.get) == a['labels'][a['gold_index']])))
    return curve(pairs)

c_bank = suite_r4('banking20'); c_ng = suite_r4('newsgroups'); c_inj = suite_r4('injection')

# ---- 画图 ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=200)
red = 'crimson'

ax = axes[0]
cov, cum, full = c_db
ax.plot(cov, cum, lw=3.2, color=red, label=f'ours R4 · 4 readings ({full*100:.1f}%)')
ax.scatter([un[0]], [un[1]], s=90, marker='*', color=red, zorder=5, label=f"ours R2/unanimous ({un[0]*100:.0f}%@{un[1]*100:.1f}%)")
ax.scatter([st[0]], [st[1]], s=90, marker='^', color='darkred', zorder=5, label=f"ours R2/strict ({st[0]*100:.0f}%@{st[1]*100:.1f}%)")
for m, c, ls in (('jev-1.13', 'k', '-.'), ('gemini-3.5-flash', 'tab:blue', '--'),
                 ('gpt-6-luna', 'tab:green', '--'), ('claude-haiku-4.5', 'tab:purple', '--'),
                 ('qwen3-32b', 'tab:orange', '--'), ('laya-routed', 'tab:brown', '--')):
    cov, cum, full = official_curve(m)
    ax.plot(cov, cum, lw=1.4, ls=ls, color=c, label=f'{m} ({full*100:.1f}%)')
ax.axhline(0.95, lw=0.8, color='gray', ls=':', alpha=0.8)
ax.text(0.21, 0.953, '5% risk line', fontsize=7.5, color='gray')
ax.set_title('A. DecisionBench bench-v4 (1071 rows)', fontsize=11)
ax.set_xlabel('coverage (auto-answered fraction)'); ax.set_ylabel('accuracy')
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.5, 1.01); ax.grid(alpha=0.3)
ax.legend(fontsize=7.6, loc='lower left', ncol=2, framealpha=0.9)

ax = axes[1]
cov, cum, full = c_jb
ax.plot(cov, cum, lw=3.2, color=red, label=f'ours R4 · 4 readings ({full*100:.1f}%)')
ax.scatter([jb_un[0]], [jb_un[1]], s=90, marker='*', color=red, zorder=5, label=f"ours R2/unanimous ({jb_un[0]*100:.0f}%@{jb_un[1]*100:.1f}%)")
ax.scatter([jb_st[0]], [jb_st[1]], s=90, marker='^', color='darkred', zorder=5, label=f"ours R2/strict ({jb_st[0]*100:.0f}%@{jb_st[1]*100:.1f}%)")
cov, cum, full = c_jev
ax.plot(cov, cum, lw=1.8, ls='-.', color='k', label=f'Jev 1.13 native ({full*100:.1f}%)')
ax.axhline(0.95, lw=0.8, color='gray', ls=':', alpha=0.8)
ax.text(0.21, 0.953, '5% risk line', fontsize=7.5, color='gray')
ax.set_title('B. JevBench-231', fontsize=11)
ax.set_xlabel('coverage (auto-answered fraction)')
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.5, 1.01); ax.grid(alpha=0.3)
ax.legend(fontsize=7.6, loc='lower left', framealpha=0.9)

ax = axes[2]
for c, name in ((c_bank, 'banking20 (K=20)'), (c_inj, 'injection (K=2)'), (c_ng, 'newsgroups (K=20)')):
    cov, cum, full = c
    ax.plot(cov, cum, lw=2.2, label=f'ours R4 · {name} ({full*100:.1f}%)')
for x, lab, col, dx, dy in ((0.520, 'anyjev best 52.0%', 'tab:blue', 12, 30),
                            (0.483, 'anyjev best 48.3%', 'tab:orange', 14, -38),
                            (0.470, 'anyjev best 47.0%', 'tab:green', -132, 2)):
    ax.scatter([x], [0.95], s=85, marker='x', color=col, zorder=6)
    ax.annotate(lab, (x, 0.95), textcoords='offset points', xytext=(dx, dy), fontsize=7.4, color=col)
ax.axhline(0.95, lw=0.8, color='gray', ls=':', alpha=0.8)
ax.set_title('C. anyjev suites — ours R4 vs their best cov@5%', fontsize=11)
ax.set_xlabel('coverage (auto-answered fraction)')
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.5, 1.01); ax.grid(alpha=0.3)
ax.legend(fontsize=7.6, loc='lower left', framealpha=0.9)

fig.suptitle('Coverage–accuracy: our R4 (zero-training) vs other systems', y=1.03, fontsize=13)
fig.tight_layout()
p = OUT / 'r4_vs_models_coverage.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(OUT / 'r4_vs_models_coverage.svg', bbox_inches='tight')
print('figure ->', p)
print(f'A: R4 full={c_db[2]*100:.1f}% | R2 strict {st[0]*100:.1f}%@{st[1]*100:.1f}% unanimous {un[0]*100:.1f}%@{un[1]*100:.1f}%')
print(f'B: R4 full={c_jb[2]*100:.1f}% | strict {jb_st[0]*100:.1f}%@{jb_st[1]*100:.1f}% unanimous {jb_un[0]*100:.1f}%@{jb_un[1]*100:.1f}% | Jev full={c_jev[2]*100:.1f}%')
print(f'C: banking {c_bank[2]*100:.1f}% / newsgroups {c_ng[2]*100:.1f}% / injection {c_inj[2]*100:.1f}%')