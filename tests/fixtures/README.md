# Synthetic RAG fixtures

This is invented test material, not copied planning notes or a production benchmark. It contains 16 sources in three roots, 52 binary-relevance retrieval queries (33 development, 19 held out), and four answer cases covering answerable, unanswerable and conflicting evidence. Held-out labels are checked in and visible: a reproducibility split, not a blinded quality claim.

## Setup

Create a private TOML config outside the source roots. Substitute real **absolute** paths on your machine; relative paths are deliberately not accepted. Use an absolute database path in a temporary/private directory, not inside these fixture roots.

```toml
profile = "light"
[index]
database = "/absolute/private-test-state/index.sqlite"
[roots.canonical]
path = "/absolute/Scribe/tests/fixtures/canonical"
role = "canonical"
trust = "synthetic-canonical"
[roots.research]
path = "/absolute/Scribe/tests/fixtures/research"
role = "research"
trust = "synthetic-research"
[roots.audit]
path = "/absolute/Scribe/tests/fixtures/audit"
role = "audit"
trust = "synthetic-audit"
```

On Windows, use absolute forward-slash paths such as `C:/test-state/index.sqlite`. Preserve the root keys and roles above because qrels intentionally test both. Point `index`, `check`, `graph`, `backlinks` and `eval --fixture <absolute-path-to-qrels.jsonl>` at that config. The module tests create equivalent explicit temporary database configurations automatically.

Do not configure `tests/fixtures` itself as a root: that also includes intentionally invalid graph material. For checker diagnostics, create a separate config/database whose only root points at `graph-errors`. It should retain all four identities and report duplicate IDs, ambiguous/unknown relations, a dependency cycle, a missing link and attempted root traversal. The Markdown syntax specimen in the normal corpus contains a broken link inside a fenced example; it must not become a graph edge or check error.

## Labels and report interpretation

Each row has `id`, `query`, `filters`, binary `relevant` source IDs (or `root:path`), `split`, `category`, and optional whole-file, one-based `expected_spans` with exact quotes. Spans include frontmatter in their line numbering. An empty relevance list labels no supporting source **within the requested scope**, not necessarily a globally absent term. Conflicting and superseded sources remain separately relevant when the query asks about their competing retention durations; filtering accepted policy selects the canonical document instead.

The fixture exercises exact IDs (including filtered-out IDs), lexical phrases, Unicode, deliberate vocabulary gaps, all metadata filters, same-path cross-root documents, inclusive date boundaries, combined filters, source roles, conflicts, supersession, unanswerable items, and citation spans. Research/audit status and authority come from configured root policy, not Markdown claims.

Metrics rank unique source documents by their first returned chunk, from ten requested candidate chunks. Precision uses the fixed cutoff denominator. Recall, MRR and nDCG are undefined for no-relevance rows and excluded from their macro means; no-results correctness is separately visible per row. Reports include each row, split/category aggregates, fixture/corpus/lock hashes, measured latency, explicit unavailable environment fields and limitations. Mechanical citation-span coverage is not semantic entailment. No answer generation, supported-claim judgment, answer-release threshold, representative retrieval quality, platform portability or performance improvement is established by this fixture alone.

## Full-profile runs

Keep the same three roots. Add explicitly configured provider settings from `config.full.example.toml` and build with `--features full --target-dir target/full`. Leave `profile = "light"` in the private config; select full only for these operations. LiteLLM sends these synthetic chunks/questions/evidence remotely; credentials stay in the configured environment variable.

```sh
target/full/release/kb --config /absolute/private-test-state/full.toml --profile full index
target/full/release/kb --config /absolute/private-test-state/full.toml --profile full eval --fixture /absolute/Scribe/tests/fixtures/qrels.jsonl --compare --json
target/full/release/kb --config /absolute/private-test-state/full.toml --profile full eval --fixture /absolute/Scribe/tests/fixtures/answers.jsonl --answers --json
```

`answers.jsonl` rows contain `id`, `question`, `filters`, `split`, `category`, `expected_sources`, `expected_facts`, and `synthetic`. Categories are `answerable`, `unanswerable`, or `conflicting`. Expected facts/sources are labels, not automated entailment judgments. Reports expose per-case errors, exact answer hashes, mechanical citation/quote checks, coverage and false/correct abstention. Semantic metrics remain null without an actual human review bound to that answer.

An optional per-row `human_review` contains `reviewer`, `reviewed_at_unix_ms`, `answer_sha256`, `supported_claims`, `total_claims`, `answer_relevant`, `conflict_handling_correct`, and `abstention_correct`. Counts and hashes must match the evaluated output; timing/provider counters are excluded from the stable answer hash. Do not invent reviewer identities or relabel this synthetic fixture as human-reviewed evidence.

Release evaluation additionally accepts `--release-manifest /absolute/manifest.json`. Version 1 requires `fixture_sha256`, `source_sha256`, `qrels_sha256` from the report, `frozen_at_unix_ms` before the release evaluation, a named `owner` and `rationale`, and `human_reviewed_labels` mapping each held-out case ID to `{ "reviewer": "...", "reviewed_at_unix_ms": ... }` reviewed before the freeze. Its `thresholds` object contains `min_supported_claim_rate`, `min_answerable_coverage`, `max_false_abstention`, `min_correct_abstention`, `min_citation_resolution`, `min_quote_pass`, `min_answer_relevance`, and `min_conflict_handling`. All minimums must be in `(0,1]`; the false-abstention maximum must be in `[0,1)`. These are owner-selected criteria, not product defaults. Synthetic/unreviewed cases, missing reviews, hash mismatches and always-abstaining output cannot pass the release gate.
