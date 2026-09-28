# Scripts

Per-suite runners and the fusion/coverage tooling. All runners speak to a local
OpenAI-compatible server exposing token log-probabilities
(llama.cpp `llama-server`); passages below assume the two frozen Qwen3.5 readers
described in `../docs/method.md`.

Serve a reader (example):

```sh
llama-server -m Qwen3.5-27B-UD-Q5_K_XL.gguf -ngl 99 -c 16384 -np 1 \
  -b 2048 -ub 512 -t 16 --flash-attn on --port 10361
```

`-np 1` is required: it gives every request the full context window (with multiple
slots llama.cpp splits the context and long prompts fail with HTTP 400).

Edit the `BASE` path at the top of each runner for your data layout; result files
are written where you point `--out`.

| Suite | Runner | Notes |
|---|---|---|
| JevBench-231 | `run_ours_bench.py` | items from the public JevBench set; letter + pair readouts |
| DecisionBench bench-v4 | `run_dbench_ours.py` | reads `cases.jsonl` from the DecisionBench corpus checkout |
| typed-decisions | `run_typed_ours.py` | `LocalLLaMA/typed-decisions` (parquet); choice/noul/score |
| banking20 / newsgroups / injection | `run_banking20_ours.py`, `run_anyjev_tasks.py` | same splits as the AnyJev benchmark (K=20 top intents; 20-newsgroups test; prompt-injections) |
| option-order flip probe | `run_banking_flip.py` | letter readout, reversed option order |
| OpenSanctions pairs | `run_osbench_ours.py` | paper test pairs 200–9,999; conflict-first entity instructions |
| spam-eval | `run_spam_ours.py` | enriched email state (links / Reply-To / attachment metadata) |
| RAG reranking | `run_rag_ours.py` | frozen top-20 candidate pools (SciFact / XQuAD-en); `prep_rag_data.py` prepares the corpora |
| scoring & figures | `score_*.py`, `make_*.py` | Wilson intervals, coverage–accuracy curves, R4/R2 fusion comparison |

## Sources & attribution

Benchmark corpora are read from their original projects and are **not redistributed** (see `../DATA_POLICY.md`). Portions of prompt wording or item formatting were adapted from:

- OpenSanctions entity text & instructions — `panios/jev-opensanctions-benchmark` (MIT) and the OpenSanctions Pairs paper code `chansmi/OSINT_entity_resolution`
- spam category definitions & enriched-state recipe — `bitnovus/jev-spam-eval` (MIT)
- banking20 / newsgroups / injection loaders mirror `nokia-applied-research/AnyJev` (Apache-2.0)
- RAG relevance wording — `emretheus/jev-rag-benchmark` (MIT)
- DecisionBench corpus — `atlanai/decision-bench` (MIT code; corpus under its own source terms)

`run_jev_api.py` reproduces reference runs against a hosted SystemOne-compatible API; it reads the API key from `TYPESAFE_API_KEY` or a local key file and ships no credentials.
