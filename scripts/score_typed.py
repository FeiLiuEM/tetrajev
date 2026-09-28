#!/usr/bin/env python3
"""Score typed-decisions runs: metrics identical to anyjev/bench/metrics.py."""
import argparse, json
import numpy as np


def acc(probs, labels):
    return float(np.mean(np.argmax(probs, 1) == np.asarray(labels)))


def ece(probs, labels, n_bins=15):
    conf = np.max(probs, 1)
    correct = (np.argmax(probs, 1) == np.asarray(labels)).astype(float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    n = len(conf)
    total = 0.0
    for chunk in np.array_split(np.arange(n), min(n_bins, n)):
        if len(chunk):
            total += len(chunk) / n * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def brier(probs, labels):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), np.asarray(labels)], 1e-12, None)
    return float(-np.mean(np.log(p)))


def metrics(rows, readout, model_filter=None):
    """逐行标量统计（K 可变，不能拼矩阵）。"""
    sel = [r for r in rows if r['readouts'].get(readout, {}).get('ok')]
    if not sel:
        return None
    conf, correct, soft, brier_list, nll_list, scmae = [], [], [], [], [], []
    for r in sel:
        pr = r['readouts'][readout]['probs']
        pv = np.array([pr.get(o, 0.0) for o in r['opts']], dtype=float)
        gi = r['gold_index']
        pred = int(np.argmax(pv))
        conf.append(float(pv.max()))
        correct.append(1.0 if pred == gi else 0.0)
        soft.append(float(np.dot(pv, np.array(r['gold_probs'], dtype=float))))
        onehot = np.zeros_like(pv)
        onehot[gi] = 1.0
        brier_list.append(float(np.sum((pv - onehot) ** 2)))
        nll_list.append(float(-np.log(max(pv[gi], 1e-12))))
        if r['gold_score'] is not None:
            scmae.append(abs(float(np.dot(pv, np.arange(len(pv)))) - r['gold_score']))
    conf = np.array(conf)
    correct = np.array(correct)
    order = np.argsort(conf)
    e = 0.0
    for chunk in np.array_split(order, min(15, len(order))):
        if len(chunk):
            e += len(chunk) / len(conf) * abs(conf[chunk].mean() - correct[chunk].mean())
    out = {'n': len(sel), 'acc': float(correct.mean()), 'ece': float(e),
           'brier': float(np.mean(brier_list)),
           'brier_mean': float(np.mean(brier_list) / np.mean([len(r['opts']) for r in sel])),
           'nll': float(np.mean(nll_list)), 'soft_acc': float(np.mean(soft))}
    if scmae:
        out['score_mae'] = float(np.mean(scmae))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    a = ap.parse_args()
    A = [json.loads(l) for l in open(a.a, encoding='utf-8') if l.strip()]
    B = [json.loads(l) for l in open(a.b, encoding='utf-8') if l.strip()]
    Bn = {r['case_id'] + '/' + r['qname']: r for r in B}
    print(f'{"config":24s} {"acc":>7s} {"soft":>7s} {"ece":>7s} {"brier_m":>8s} {"score_mae":>9s} {"n":>5s}')
    for name, rows, rd in [('27b/letter', A, 'letter'), ('27b/pair', A, 'pair'),
                           ('35b/letter', B, 'letter'), ('35b/pair', B, 'pair')]:
        m = metrics(rows, rd)
        if m:
            print(f'{name:24s} {m["acc"]:7.4f} {m["soft_acc"]:7.4f} {m["ece"]:7.4f} {m["brier_mean"]:8.4f} '
                  f'{m.get("score_mae", float("nan")):9.4f} {m["n"]:5d}')
    # 融合：两模型×两读数 均值（按 case/qname 对齐）
    fused_letter, fused_pair, fused_all = [], [], []
    for ra in A:
        key = ra['case_id'] + '/' + ra['qname']
        rb = Bn.get(key)
        if not rb:
            continue
        import copy
        def avg(readouts, rs):
            ds = [r['readouts'].get(x, {}) for r, x in rs]
            if not all(d.get('ok') for d in ds):
                return None
            out = {}
            for o in ra['opts']:
                out[o] = sum(d['probs'].get(o, 0.0) for d in ds) / len(ds)
            return out
        la = avg(None, [(ra, 'letter'), (rb, 'letter')])
        pa = avg(None, [(ra, 'pair'), (rb, 'pair')])
        aa = avg(None, [(ra, 'letter'), (ra, 'pair'), (rb, 'letter'), (rb, 'pair')])
        for rows_out, probs in ((fused_letter, la), (fused_pair, pa), (fused_all, aa)):
            rr = copy.deepcopy(ra)
            if probs:
                rr['readouts'] = {'fused': {'ok': True, 'probs': probs,
                                            'pred_idx': ra['opts'].index(max(probs, key=probs.get))}}
            rows_out.append(rr)
    for name, rows in [('mean(letter) 2model', fused_letter), ('mean(pair) 2model', fused_pair),
                       ('mean(all4)', fused_all)]:
        m = metrics(rows, 'fused')
        if m:
            print(f'{name:24s} {m["acc"]:7.4f} {m["soft_acc"]:7.4f} {m["ece"]:7.4f} {m["brier_mean"]:8.4f} '
                  f'{m.get("score_mae", float("nan")):9.4f} {m["n"]:5d}')
    # 分类型（27b/letter & all4）
    for name, rows, rd in [('27b/letter choice', [r for r in A if r['kind'] == 'choice'], 'letter'),
                           ('27b/letter noul', [r for r in A if r['kind'] == 'noul'], 'letter'),
                           ('27b/letter score', [r for r in A if r['kind'] == 'score'], 'letter')]:
        m = metrics(rows, rd)
        if m:
            print(f'{name:24s} {m["acc"]:7.4f} {m["soft_acc"]:7.4f} {m["ece"]:7.4f} {m["brier_mean"]:8.4f} '
                  f'{m.get("score_mae", float("nan")):9.4f} {m["n"]:5d}')


if __name__ == '__main__':
    main()