# Results — initial release (2026-09-28)

All numbers were produced by the scripts in `scripts/` on public benchmark suites, with the two frozen readers on one 24 GB GPU (`-np 1`). Aggregate results only; per-item files are not redistributed (see `../DATA_POLICY.md`).

**Metric conventions.** **R4** = four-reading equal-weight fusion (fit-free). **R2** = agreement routing (strict / unanimous / split). "Coverage @ ≥X% precision" = the largest fraction of items that can be auto-released while the released subset keeps accuracy ≥ X%. Component reading accuracies are listed as diagnostics only — comparisons use R4 / R2.

## Per-suite results

| Suite (n) | R4 full acc | coverage@≥98 | @≥95 | @≥93 | R2 strict | R2 unanimous | split |
|---|---:|---:|---:|---:|---|---|---:|
| DecisionBench bench-v4 (1,070) | 85.9% | 40.3% | 75.0% | 84.2% | 35.5% @ 94.5% | 71.0% @ 90.0% | 29.0% |
| JevBench-231 (231) | 78.4% | 46.3% | 61.0% | 64.9% | 34.2% @ 96.2% | 68.4% @ 84.8% | 31.6% |
| banking20 (298) | 85.9% | 23.2% | 41.6% | 72.8% | 47.0% @ 95.0% | 94.0% @ 87.5% | 6.0% |
| newsgroups (299) | 76.6% | 7.0% | 29.8% | 40.5% | 37.8% @ 87.6% | 75.9% @ 82.4% | 24.1% |
| injection (300) | 82.3% | 17.0% | 53.7% | 63.3% | 42.0% @ 94.4% | 84.0% @ 80.6% | 16.0% |
| typed-decisions (2,000) | 71.8% | 5.3% | 16.1% | 23.6% | 25.4% @ 90.2% | 50.8% @ 77.3% | 49.1% |
| OpenSanctions, 27B pass (9,800) | F1 98.58 letter / 97.85 pair | — | — | — | — | — | — |

## Reference values (as published by their authors, same suites)

- **DecisionBench bench-v4 official leaderboard** (decisionbench.ai): Jev 1.13 **92.4%** [90.7–93.9]; Gemini 3.5 Flash 94.3%; GPT-6 Luna 94.5%; Claude Sonnet 5 92.7%; Claude Haiku 4.5 91.3%; Qwen3-32B 84.1% (57 no-answer); Laya 52.8%.
- **JevBench-231**: Jev 1.13 measured by us on the same 231 items (native API): **78.8%** (reference; see caveats).
- **OpenSanctions pairs**: Jev 1.13 **98.87** F1 [98.70–99.04]; GPT-4o 98.95 (paper Table 3).
- **banking20 / newsgroups / injection**: best published open-reproduction values from the AnyJev benchmark documentation — 80.7% / 73.7% / 86.0%.
- **typed-decisions**: Jev 1.13 **72.7%** (as compiled by the Laya benchmark release); best published open-reproduction 70.0%.

## Component readings (diagnostic only)

| Suite | 27B·letter | 27B·pair | 35B·letter | 35B·pair |
|---|---:|---:|---:|---:|
| JevBench-231 | 79.2% | 76.6% | 61.9% | 70.6% |
| typed-decisions | 70.7% | 70.3% | 51.2% | 58.7% |
| banking20 | 83.3% | 63.0% | 84.3% | 73.2% |
| newsgroups | 75.3% | 75.0% | 65.3% | 59.5% |
| injection | 83.0% | 83.3% | 71.0% | 68.0% |
| DecisionBench | 86.1% | 84.7% | 68.0% | 73.9% |

Observed: R4 improves on the best component in most suites; where a component is markedly weak (35B on DecisionBench; pair at K = 20 in banking20) equal-weight fusion dilutes high-precision coverage — see `method.md`, Known limits.

## Robustness — option-order flip rate

banking20, letter readout, options reversed: **27B flips 9.0%** of answers, **35B 9.7%** — single-pass, without permutation averaging.

## Determinism check

9,486 items (OpenSanctions pairs) were read twice — on an 8k-context and a 16k-context server (same weights and flags otherwise). Both readouts: **9,486/9,486 letter and pair values bit-identical; maximum probability delta 0.00e+00; 9,486/9,486 identical predictions.** Serving configuration does not change readings (f16 KV cache).

## In progress (will be updated in place)

- **spam-eval** (5,733 messages; long-context classification) — 27B pass finishing at release time.
- **Jev RAG reranking** — SciFact (300 queries) and XQuAD-en (1,190 queries), frozen top-20 candidate pools.
- **OpenSanctions second-reader + R4** and **35B passes** for all suites above.

## Caveats

1. Reference numbers are each project's own published values (sources above); they were not re-run here except where noted (Jev on JevBench-231).
2. Protocol differences exist by design: official harnesses ask models to *generate* answers with probabilities; TetraJev reads token log-probabilities. Comparisons are between complete systems, not identical prompts.
3. `R4` here is equal-weight (fit-free). A dev-weighted variant exists in the medical sibling project; for these suites no separate dev split was used, deliberately.
4. `JevBench-231` Jev reference was measured by us with its native choice API on the same items; other Jev rows are as published.
