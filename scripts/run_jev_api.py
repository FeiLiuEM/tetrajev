#!/usr/bin/env python3
"""Jev-General: Jev API runner on bench items (native choice primitive)."""
import argparse, json, os, random, time, urllib.error, urllib.request

API = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-latest'


def load_key():
    k = (os.environ.get('TYPESAFE_API_KEY') or '').strip()
    if k:
        return k
    for p in ('/media/super/Hard_Disk_2/hermes/research/decision-model/.secrets/jev_api_key',
              os.path.expanduser('~/hermes/projects/decision-model/.secrets/jev_api_key')):
        if os.path.exists(p):
            return open(p).read().strip()
    raise SystemExit('no API key found')


def post_json(payload, key, timeout):
    req = urllib.request.Request(API, data=json.dumps(payload).encode('utf-8'), method='POST',
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8')), resp.status


def call(it, key, timeout=120):
    payload = {'state': it['input'], 'model': MODEL, 'questions': {
        'answer': {'type': 'choice',
                   'instructions': 'Which of the following options correctly answers the question?',
                   'criteria': {str(l): str(l) for l in it['labels']}}}}
    last = ''
    for backoff in (0.0, 1.0, 2.0, 4.0, 8.0):
        if backoff:
            time.sleep(backoff + random.random())
        try:
            body, status = post_json(payload, key, timeout)
            return {'ok': True, 'status': status, 'answers': body.get('answers'),
                    'model': body.get('model'), 'usage': body.get('usage')}
        except urllib.error.HTTPError as e:
            msg = ''
            try:
                msg = e.read().decode('utf-8', 'replace')[:300]
            except Exception:
                pass
            last = f'HTTP {e.code}: {msg}'
            if e.code == 429 or e.code >= 500:
                continue
            break
        except Exception as e:  # noqa: BLE001
            last = f'{type(e).__name__}: {e}'
            continue
    return {'ok': False, 'error': last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    key = load_key()
    rows = []
    for f in a.items.split(','):
        rows.extend(json.loads(l) for l in open(f, encoding='utf-8') if l.strip())
    out = []
    t0 = time.time()
    for i, it in enumerate(rows, 1):
        r = call(it, key)
        rec = {'id': it['id'], 'split': it.get('split'), 'family': it.get('family'),
               'expected': it['expected'], 'labels': it['labels']}
        if r['ok']:
            ans = (r.get('answers') or {}).get('answer') or {}
            dist = ans.get('probabilities') or ans.get('distribution') or {}
            dist = {str(k): float(v) for k, v in dist.items()}
            rec.update(ok=True, probs=dist,
                       pred=(max(dist, key=dist.get) if dist else None),
                       correct=int(bool(dist) and max(dist, key=dist.get) == it['expected']))
        else:
            rec.update(ok=False, error=r.get('error'))
        out.append(rec)
        if i % 20 == 0:
            print(f'  {i}/{len(rows)} | {(time.time()-t0)/i:.2f}s/item', flush=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for sp in sorted({r['split'] for r in out} | {'ALL'}):
        sel = [r for r in out if (sp == 'ALL' or r['split'] == sp) and r.get('ok')]
        if sel:
            cor = sum(r['correct'] for r in sel)
            print(f'  [jev/{sp}] {cor}/{len(sel)} = {cor/len(sel):.4f}', flush=True)
    print(f'-> {a.out}', flush=True)


if __name__ == '__main__':
    main()
