#!/usr/bin/env python3
"""Jev-General: merge our runs + Jev run -> per-split comparison table."""
import argparse, itertools, json
from collections import defaultdict


def load(p, key='readouts'):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def acc_of(rows, pred_fn, split=None):
    sel = [r for r in rows if split is None or r['split'] == split]
    if not sel:
        return None
    c = sum(1 for r in sel if pred_fn(r))
    return c, len(sel), c / len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='ours run file (model A)')
    ap.add_argument('--b', required=True, help='ours run file (model B)')
    ap.add_argument('--jev', default='', help='jev run file')
    ap.add_argument('--splits', default='easy,original,hard')
    a = ap.parse_args()
    A = {r['id']: r for r in load(a.a)}
    B = {r['id']: r for r in load(a.b)}
    J = {r['id']: r for r in load(a.jev)} if a.jev else {}
    ids = [i for i in A if i in B]
    splits = [s for s in a.splits.split(',') if s] + ['ALL']

    def get(r, rd):
        v = r.get('readouts', {}).get(rd)
        return v.get('probs') if v and v.get('ok') else None

    def fuse_mean(ds):
        ds = [d for d in ds if d]
        if not ds:
            return None
        out = {}
        for k in ds[0]:
            out[k] = sum(d.get(k, 0.0) for d in ds) / len(ds)
        return out

    def pred_of(probs):
        return None if not probs else max(probs, key=probs.get)

    print(f'{"config":28s} ' + ' '.join(f'{s:>14s}' for s in splits))
    defs = [
        ('27b/letter', lambda r: pred_of(get(A[r['id']], 'letter'))),
        ('27b/pair', lambda r: pred_of(get(A[r['id']], 'pair'))),
        ('35b/letter', lambda r: pred_of(get(B[r['id']], 'letter'))),
        ('35b/pair', lambda r: pred_of(get(B[r['id']], 'pair'))),
        ('mean(letter)', lambda r: pred_of(fuse_mean([get(A[r['id']], 'letter'), get(B[r['id']], 'letter')]))),
        ('mean(pair)', lambda r: pred_of(fuse_mean([get(A[r['id']], 'pair'), get(B[r['id']], 'pair')]))),
        ('mean(all4)', lambda r: pred_of(fuse_mean([get(A[r['id']], x) for x in ('letter', 'pair')] +
                                                   [get(B[r['id']], x) for x in ('letter', 'pair')]))),
        ('unanimous(pair)', lambda r: (lambda pa, pb: pa if (pa and pb and pa == pb) else None)(
            pred_of(get(A[r['id']], 'pair')), pred_of(get(B[r['id']], 'pair')))),
    ]
    allrows = []
    for i in ids:
        row = {'id': i, 'split': A[i]['split'], 'expected': A[i]['expected']}
        allrows.append(row)
    print('(coverage-weighted for gates; plain acc otherwise)')
    for name, fn in defs:
        cells = []
        for sp in splits:
            sel = [r for r in allrows if sp == 'ALL' or r['split'] == sp]
            if name.startswith('unanimous'):
                hit = [r for r in sel if fn(r) is not None]
                if hit:
                    cor = sum(1 for r in hit if fn(r) == A[r['id']]['expected'])
                    cells.append(f'{cor/len(hit):.3f}@{len(hit)/len(sel):.2f}')
                else:
                    cells.append('n/a')
            else:
                ok = [r for r in sel if fn(r) is not None]
                if ok:
                    cor = sum(1 for r in ok if fn(r) == A[r['id']]['expected'])
                    cells.append(f'{cor/len(sel):.4f}')
                else:
                    cells.append('n/a')
        print(f'{name:28s} ' + ' '.join(f'{c:>14s}' for c in cells))
    if J:
        cells = []
        for sp in splits:
            sel = [r for r in allrows if sp == 'ALL' or r['split'] == sp]
            ok = [r for r in sel if J.get(r['id'], {}).get('ok')]
            if ok:
                cor = sum(1 for r in ok if J[r['id']].get('correct'))
                cells.append(f'{cor/len(ok):.4f}')
            else:
                cells.append('n/a')
        print(f'{"Jev (native choice)":28s} ' + ' '.join(f'{c:>14s}' for c in cells))
    # readout agreement stats
    for rd in ('letter', 'pair'):
        both = [r for r in allrows if get(A[r['id']], rd) and get(B[r['id']], rd)]
        agree = sum(1 for r in both if pred_of(get(A[r['id']], rd)) == pred_of(get(B[r['id']], rd)))
        if both:
            print(f'[{rd}] both-ok n={len(both)} | two-reader agreement {agree/len(both):.3f}')


if __name__ == '__main__':
    main()
