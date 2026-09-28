#!/usr/bin/env python3
"""banking20 letter 读数的选项顺序翻转率（对齐 anyjev flip 列）。"""
import argparse, json, math, time, urllib.request
import sys
sys.path.insert(0, '/media/super/Hard_Disk_2/hermes/research/jev-general/harness')
from run_banking20_ours import load_banking20, letter_prompt, read_top, variants, lse, softmax_dict

ap = argparse.ArgumentParser()
ap.add_argument('--server', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

labels, test = load_banking20('/media/super/Hard_Disk_2/hermes/research/jev-general')
rev_labels = list(reversed(labels))
out = []
t0 = time.time()
for i, (text, gold) in enumerate(test, 1):
    rec = {'idx': i - 1, 'gold_index': gold}
    for name, labs in (('fwd', labels), ('rev', rev_labels)):
        lut = read_top(a.server, letter_prompt(text, labs))
        raw = {}
        for j, o in enumerate(labs):
            L = chr(ord('A') + j)
            vals = [lut[s] for s in variants(L) if s in lut]
            raw[o] = lse(vals) if vals else None
        probs = softmax_dict(raw)
        rec[name] = {'pred': max(probs, key=probs.get)}
    rec['flip'] = int(rec['fwd']['pred'] != rec['rev']['pred'])
    out.append(rec)
    if i % 50 == 0:
        print(f'  {i}/{len(test)} | {(time.time()-t0)/i:.2f}s/item', flush=True)
with open(a.out, 'w', encoding='utf-8') as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
flip = sum(r['flip'] for r in out) / len(out)
print(f'  [{a.tag}] flip rate = {flip:.4f} ({sum(r["flip"] for r in out)}/{len(out)})', flush=True)