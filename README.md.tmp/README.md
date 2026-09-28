# TetraJev

**Four readings. One routed decision.** Two frozen open-weight models, four independent readings per task, one fit-free decision layer — Jev-class decision behaviour with **no fine-tuning and no distillation**.

![TetraJev — system architecture: two frozen readers, four readings (letter + pair per reader), R4 fusion, R2 agreement router, auto-release or human review](assets/tetrajev-architecture.svg)

<sub>Vector source: [`assets/tetrajev-architecture.svg`](assets/tetrajev-architecture.svg). Each frozen reader reads a task twice: a **letter readout** (probability mass over the option letters) and a **pair readout** (per-candidate yes/no Δ). The four per-option distributions are combined fit-free (**R4**) and routed by agreement (**R2**: strict / unanimous / split). Split or low-confidence items abstain to human review.</sub>

TetraJev turns off-the-shelf open-weight models — [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (dense) and [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (MoE), served as GGUF via llama.cpp on a single 24 GB GPU — into a calibrated decision layer for **bounded decision tasks**: a state plus a finite option set (K = 2…20). Classification, routing, yes/no gates, entity matching, reranking and multiple-choice judgment all reduce to the same primitive *(state + options) → probability distribution*, and that is all TetraJev needs to run.

## Why four readings?

Two readers × two readout structures. A single reading is one opinion; four readings give the router something a single reading cannot provide: **agreement**. R4 (fusion) raises full-coverage accuracy; R2 (agreement routing) sorts items into *strict*, *unanimous* and *split*, so a deployment can auto-release only the items both readers and both structures see the same way — and abstain on the rest. Everything is fit-free: no trained weights, no constants calibrated on the evaluation data; readings are deterministic greedy decodes.

## Results so far

![Coverage–accuracy: our R4/R2 vs other systems](assets/r4_vs_models_coverage.png)

<sub>Coverage–accuracy curves. Left: DecisionBench vs official models. Middle: JevBench vs Jev. Right: three classification suites vs the best published cov@5% points of the open-reproduction family. Higher-left is better; the dotted line marks the 5% risk level (≥95% precision).</sub>

Headline numbers on public decision suites (initial release; full tables and caveats in [`docs/results.md`](docs/results.md)):

| Suite | items | R4 accuracy (full coverage) | coverage @ ≥95% precision | reference |
|---|---:|---:|---:|---|
| DecisionBench bench-v4 | 1,070 | **85.9%** | **75.0%** | Jev 1.13: 92.4% |
| JevBench-231 | 231 | **78.4%** | 61.0% | Jev 1.13 (same items): 78.8% |
| banking20 | 298 | **85.9%** | 41.6% | best published open repro: 80.7% |
| newsgroups | 299 | **76.6%** | 29.8% | best published open repro: 73.7% |
| injection | 300 | **82.3%** | 53.7% | best published open repro: 86.0% |
| typed-decisions | 2,000 | **71.8%** | 16.1% | Jev 1.13: 72.7% |
| OpenSanctions (27B pass) | 9,800 | F1 **98.58** (letter) · 97.85 (pair) | — | Jev 1.13: 98.87 |

### Reference systems — DecisionBench bench-v4 (official runs)

Where TetraJev (R4) sits against the published leaderboard. Accuracy = official published value (1,071 rows); coverage columns are recomputed from each run's published per-item confidence, among answered items; "no answer" is the official count. Full machine-readable table: [`docs/decisionbench-bench-v4-reference.csv`](docs/decisionbench-bench-v4-reference.csv).

| Model | Accuracy | cov@≥98% | cov@≥95% | cov@≥93% | No answer |
|---|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash | 94.2% | 74.8% | 97.7% | 100% | 1 |
| Gemini Flash Lite | 94.1% | 16.8% | 97.5% | 100% | 0 |
| GPT-6 Luna | 93.7% | 75.4% | 98.6% | 100% | 10 |
| DeepSeek V4.1 Flash | 92.7% | 77.7% | 96.9% | 100% | 6 |
| Claude Sonnet 5 | 92.6% | 62.8% | 93.7% | 99.2% | 1 |
| GPT-5.6 Luna | 92.6% | 64.5% | 93.0% | 100% | 5 |
| Jev 1.13 | 92.4% | 80.2% | 94.0% | 98.6% | 0 |
| GLM 5.3 Flash | 92.3% | 86.1% | 95.9% | 99.5% | 7 |
| Sage † | 92.1% | 4.2% | 84.5% | 96.1% | 0 |
| Claude Haiku 4.5 | 90.6% | 38.3% | 81.7% | 93.0% | 8 |
| Tev1 4B † | 85.4% | — | — | — | 0 |
| **TetraJev R4 (this work)** | **85.9%** | **40.3%** | **75.0%** | **84.2%** | **1** |
| Qwen3-32B | 79.6% | 1.4% | 1.4% | 8.8% | 57 |
| Nova Micro | 66.4% | 0.8% | 4.5% | 4.7% | 107 |
| Laya (routed) | 52.8% | 1.1% | 8.3% | 9.3% | 0 |

† marked "partial or older corpus" in the official notes. Source: decisionbench.ai leaderboard (23 Sep 2026); coverage recomputed from the published per-item files.

### Reference systems — other suites (as published)

| Suite | System | Result | Note |
|---|---|---|---|
| JevBench-231 | Jev 1.13 | 78.8% | measured by us on the same 231 items (native API) |
| OpenSanctions (9,800 pairs) | Jev 1.13 | F1 98.87 [98.70–99.04] | |
| | GPT-4o | F1 98.95 | paper Table 3 |
| | DeepSeek-R1-Distill-Qwen-14B | F1 97.76 / 98.23 | 0-shot / optimized |
| | Claude Opus 4.5 | F1 95.45 | |
| | GPT-5 Nano | F1 95.24 | |
| | GPT-3.5 Turbo | F1 94.49 | |
| | Llama-3.1-8B | F1 94.05 / 95.94 | 0-shot / optimized |
| | nomenklatura RegressionV1 (rules) | F1 91.33 | non-LLM baseline |
| typed-decisions (2,000) | laya-typed-decisions † | 76.8% | fine-tuned on this suite's train split |
| | Jev 1.13 | 72.7% | as compiled by the Laya benchmark release |
| | Qwen3-32B · L0 | 70.0% | open reproduction family |
| | Qwen3-8B · L0 | 64.7% | |
| | Qwen2.5-7B · L0 | 62.8% | |
| banking20 | Qwen3-8B · L1 | 80.7% | best published open reproduction |
| newsgroups | Qwen3-32B · L1 | 73.7% | best published open reproduction |
| injection | Qwen3-32B · L0-bc | 86.0% | best published open reproduction |

† trained on the suite; listed for context, not as a zero-training baseline.

## Design invariants

- **No training of any kind** — weights are frozen; nothing is fitted, distilled or fine-tuned on evaluation data.
- **Fit-free fusion** — R4 combines readings with fixed equal weights; R2 thresholds are rule-based, not fitted.
- **Deterministic readings** — greedy decode, one-token continuation, log-probability readout. Verified bit-identical across serving configurations (8k vs 16k context, 9,486 shared items, max probability delta 0.00e+00).
- **Model-agnostic** — any local server exposing token logprobs works; the shipped results pin the two Qwen3.5 readers above.
- **Local & private by construction** — one 24 GB GPU; no data leaves the machine.

## Repository layout

```
scripts/   per-suite runners + fusion/coverage/figure tools (reproduce every number in docs/results.md)
docs/      method.md (readings, R4/R2, serving) · results.md (all published numbers + caveats)
assets/    architecture diagram · coverage figures
```

## Status & roadmap

Initial release: scripts + aggregate results for seven public suites. **In progress:** spam-eval (5,733 messages), Jev RAG reranking (SciFact 300 / XQuAD-en 1,190 queries), and the second-reader pass for OpenSanctions — results will be updated in place. Roadmap: quality-weighted fusion (after the fit-free baseline), release-gate calibration reports, additional suites.

## Reproducing

- Serve the two frozen readers (GGUF, llama.cpp, `-np 1` so each request gets the full context); runners are standard-library only and speak to a local OpenAI-compatible server (`--server`).
- `scripts/README.md` maps each suite to its runner; `requirements.txt` covers the plotting tools (figures only).
- Benchmarks are fetched from their original sources by the scripts; **nothing from those corpora is redistributed here** — see [`DATA_POLICY.md`](DATA_POLICY.md).

## License & provenance

- MIT — see [`LICENSE`](LICENSE).
- Development assisted by AI tooling; every number above is reproducible from the shipped scripts against the per-suite result files.
- TetraJev is an independent research project, not affiliated with TypeSafe AI. "Jev" in the name refers to the decision-benchmark lineage the wider community compares against.
