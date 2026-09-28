# Data policy

TetraJev publishes **code, aggregate results and figures** — never benchmark corpora, question text, or per-item evaluation files.

- **Benchmarks are read from their original sources.** The evaluation scripts fetch or read the public datasets (DecisionBench bench-v4, JevBench-231, typed-decisions, BANKING77 / 20 Newsgroups / prompt-injections, OpenSanctions pairs, SciFact / XQuAD-en, spam corpora) from the upstream projects; their licenses and terms apply. Nothing from those corpora is redistributed here.
- **Aggregate results only.** This repository ships per-suite summaries, coverage curves and the scripts that compute them. Per-item judged files — which would embed benchmark content — are intentionally not included.
- **Model weights are linked, not mirrored.** Tested readers are referenced via their official distributions and sha256 checksums (see `docs/method.md`); no weights are hosted here.
- **No private data.** No personal data, credentials or proprietary corpora appear anywhere in this repository.
