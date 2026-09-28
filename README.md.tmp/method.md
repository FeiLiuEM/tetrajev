# Method — readings, fusion, routing

## The primitive

Every covered task is a **bounded decision**: a state (evidence) plus a finite option set (K = 2…20) with exactly one correct answer or a labelled target. The system returns a probability distribution over the options, a confidence, and a routing tier. Classification, extraction routing, yes/no gates, entity resolution, reranking and multiple-choice questions all instantiate this primitive.

## The four readings (R4)

Each frozen reader answers every item twice, with two independent readout structures:

1. **Letter readout** — options are presented as a letter mapping (A, B, …); the model is asked to output exactly one letter; the first-token log-probability mass over the letter variants is read from the top-2,000 logprobs and normalized into a distribution. **One forward pass per item.**
2. **Pair readout** — for every candidate option, the model answers "Is the candidate answer correct? Reply yes or no." The yes−no logit difference (Δ) is each candidate's score; the score vector is softmaxed into a distribution. **K forward passes per item.**

This yields four per-option distributions per item (2 readers × 2 structures). All readings are single greedy decodes with `temperature 0` and `n_predict = 1` — deterministic across re-runs and across serving context sizes (f16 KV cache).

## R4 — fusion

Equal-weight mean of the four per-option distributions. Fit-free: no weights, no temperature scaling, no constants fitted on evaluation data. R4 is the primary headline configuration.

## R2 — agreement routing

Two readings per item (each reader's primary structure):

| tier | rule | route |
|---|---|---|
| **strict** | both readers' top option agrees **and** the minimum top-1 margin is in the top half | auto-release |
| **unanimous** | both readers' top option agrees | auto-release |
| **split** | readers disagree | abstain → human review |

The router is fit-free; tiers and thresholds are the rule above. Coverage–accuracy curves (in `docs/results.md`) show the operating points: they are read as "accuracy on the auto-released subset" vs "fraction auto-released".

## Serving & tested artifacts

- llama.cpp server, `-np 1` (single slot: every request gets the full context window; this fixed a class of long-prompt failures).
- Reading requests: `/tokenize` + `/completion` with `n_probs = 2000`, `temperature = 0`, `cache_prompt = false`.
- Tested readers (GGUF, official distributions, linked not mirrored):
  - `Qwen/Qwen3.5-27B` — UD-Q5_K_XL · sha256 `423ebe4779f784a2921da340a0fab0c372fbafb4b9eb502ae3768218de01597a`
  - `Qwen/Qwen3.5-35B-A3B` — UD-Q4_K_XL · sha256 `69171bb5fbea1486d8390395bde313f4d511f8eb7810b752322316a60d8710d4`
- Hardware: a single 24 GB GPU holds either reader at full offload (they are run alternately, not concurrently).

## Known limits (honest list)

- **Equal-weight fusion dilutes weak readings.** On suites where one reading is markedly weaker (e.g., the 35B reader on DecisionBench; the pair readout at K = 20), equal-weight R4 can trail the best single reading at high-precision tiers. Quality-weighted fusion is roadmap work; the fit-free baseline ships first.
- **The letter readout caps at K ≤ 26** (option letters); larger label sets need a different structure (e.g., span readout). The K = 20 suites are within range.
- **Binary items (K = 2) make the two structures near-equivalent**, so R4's structural diversity adds little there; the two-reader diversity still applies.
- Reference numbers for other systems are their published values on the same suites; protocol differences (each project's own harness vs the readouts here) are stated per suite in `docs/results.md`.
