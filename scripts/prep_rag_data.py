#!/usr/bin/env python3
"""下载并归一化 RAG 数据：scifact(BEIR) + xquad.en —— 复刻 jev-rag-bench/data.py。"""
import hashlib, io, json, pathlib, urllib.request, zipfile

OUT = pathlib.Path.home() / 'hermes/projects/jev-general/data_local/rag_processed'
XQUAD_URL = "https://raw.githubusercontent.com/deepmind/xquad/master/xquad.en.json"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download(url, timeout=300):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write(out_dir, docs, queries):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'corpus.jsonl', 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(out_dir / 'queries.jsonl', 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


print('下载 xquad.en.json ...', flush=True)
raw = json.loads(download(XQUAD_URL))
docs, queries = {}, []
for article in raw.get('data', []):
    title = str(article.get('title', '')).strip()
    for para in article.get('paragraphs', []):
        ctx = str(para.get('context', '')).strip()
        if not ctx:
            continue
        doc_id = 'xq-' + hashlib.sha1(ctx.encode()).hexdigest()[:12]
        docs.setdefault(doc_id, {'doc_id': doc_id, 'title': title, 'text': ctx})
        for qa in para.get('qas', []):
            answers = [str(a['text']).strip() for a in qa.get('answers', []) if str(a.get('text', '')).strip()]
            question = str(qa.get('question', '')).strip()
            if not answers or not question:
                continue
            queries.append({'query_id': str(qa['id']), 'question': question,
                            'gold_doc_ids': [doc_id], 'answers': answers})
write(OUT / 'xquad-en', list(docs.values()), queries)
print(f'xquad: {len(docs)} docs, {len(queries)} queries', flush=True)

print('下载 scifact.zip ...', flush=True)
payload = download(SCIFACT_URL)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    names = z.namelist()
    corpus_name = next(n for n in names if n.endswith('corpus.jsonl'))
    queries_name = next(n for n in names if n.endswith('queries.jsonl'))
    qrels_name = next(n for n in names if n.endswith('qrels/test.tsv'))
    docs = []
    for line in z.read(corpus_name).decode('utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            docs.append({'doc_id': str(row['_id']), 'title': str(row.get('title') or '').strip(),
                         'text': str(row.get('text') or '').strip()})
    queries = []
    for line in z.read(queries_name).decode('utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            queries.append({'query_id': str(row['_id']), 'question': str(row.get('text', '')).strip(),
                            'gold_doc_ids': [], 'answers': []})
    by_id = {q['query_id']: q for q in queries}
    for line in z.read(qrels_name).decode('utf-8').splitlines():
        s = line.strip()
        if not s or s.lower().startswith('query'):
            continue
        parts = s.split()
        if len(parts) >= 3 and parts[0] in by_id:
            by_id[parts[0]]['gold_doc_ids'].append(parts[1])
write(OUT / 'scifact', docs, queries)
print(f'scifact: {len(docs)} docs, {len(queries)} queries', flush=True)
# 与冻结结果对账用：候选覆盖统计
for name in ('scifact', 'xquad-en'):
    corpus = {json.loads(l)['doc_id'] for l in open(OUT / name / 'corpus.jsonl', encoding='utf-8')}
    print(f'{name}: corpus ids = {len(corpus)}')