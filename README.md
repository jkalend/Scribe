# Scribe

Local Markdown retrieval, cited evidence packs, and opt-in grounded answers. One Rust crate; SQLite FTS5/BM25. **Light is the runtime default and default Cargo build**, with no models, credentials, model service, HTTP client, or tokenizer dependency. Code and synthetic fixtures live here; private documents and generated state stay in explicitly configured instance locations.

The `full` feature adds embeddings, hybrid retrieval, and mechanically verified answers under the [accepted plan](https://github.com/jkalend/project-planning/blob/main/tools/kb-rs-spec.md). Full requires an explicitly selected profile and configured providers. It supports local Ollama or an explicitly authorized LiteLLM endpoint; remote full inference sends source chunks, queries and evidence off-device. Merely compiling full or configuring a provider does not enable it.

## Build and configure

Requires Rust 1.89+ and the platform C build tools for bundled SQLite (Visual Studio C++ Build Tools on Windows; a C compiler on Linux). `Cargo.lock` pins dependencies; the reviewed parser/SQLite/MCP releases are pinned in `Cargo.toml` too.

```sh
cargo build --release --locked
cargo test --locked
```

Copy `config.example.toml` to a private location, edit its **absolute** source and database paths, then explicitly select it:

```sh
kb --config /absolute/private/scribe.toml index
kb --config /absolute/private/scribe.toml search "recovery" --json
kb --config /absolute/private/scribe.toml search "recovery" --tag rust --status active
kb --config /absolute/private/scribe.toml get notes:design.md
kb --config /absolute/private/scribe.toml context "recovery" --budget-bytes 16384 --json
kb --config /absolute/private/scribe.toml backlinks P-EXAMPLE
kb --config /absolute/private/scribe.toml graph --format mermaid
kb --config /absolute/private/scribe.toml check --json
kb --config /absolute/private/scribe.toml stats
```

Use `target/release/kb` (`kb.exe` on Windows) before installing it on PATH, or `cargo install --path . --locked`. `KB_CONFIG` may select the configuration file. `--config` wins over `KB_CONFIG`; `--profile` wins over `KB_PROFILE`, then config, then `light`. Source paths and database come only from config or `--root KEY=ABSOLUTE_PATH` / `--database ABSOLUTE_PATH`. `--root-filter KEY` restricts a search without changing configured scope. Configured roles/trust are operator policy; document frontmatter cannot grant itself authority.

`search` treats input as plain text, not FTS grammar. An exact frontmatter ID selects that document's chunks, preserving duplicate-ID ambiguity. `context P-EXAMPLE` supports the project-oriented workflow. Filters cover root, role, status, exact project/type/tag, and inclusive dates. BM25 scores are labeled **lower is better**, never probabilities. JSON envelopes carry schema version, selected profile, generation, actual retrieval legs/model contracts, and fusion policy where used; metadata-only operations do not fabricate model contracts.

## Provenance, limits, and recovery

- YAML frontmatter only at the beginning; known fields are strictly typed. Supported lists: `tags`, `aliases`, `depends_on`, `supersedes`, `conflicts`. Unknown scalar/flat scalar-list metadata is retained, not treated as authority. YAML aliases/anchors/explicit tags and nested unknown metadata are rejected; depth and metadata bytes are bounded.
- CommonMark inline/reference links and images, headings, and tables are parsed. Links/headings inside code are not interpreted. Source IDs and parsed links drive the graph; arbitrary text mentions are not backlinks. `check` reports duplicate IDs, unknown dependencies/relations, broken local links, cycles and stale source state. It does not promote IDs mentioned in prose into a trusted registry.
- Raw bytes are SHA-256 hashed. Excerpts normalize CRLF/CR to LF and cite inclusive one-based **whole-file** lines, including frontmatter. Oversized single lines fail instead of inventing partial-line citations. Search/get/context reject changed or missing selected sources; run `index` to refresh. Citations bind to the captured hash/generation, not an atomic filesystem snapshot.
- Roots are explicit and canonicalized. Scanning skips symlinks/reparse points and ignored directories; retrieval canonicalizes and checks root containment. This is cooperative path scoping, **not an OS sandbox** against a privileged actor racing path changes.
- `context --budget-bytes` bounds the serialized JSON `data` payload (including citation metadata and its truncation manifest), not tokens or the outer envelope. Minimum 512 bytes; maximum configured `max_context_bytes`. It excludes whole chunks rather than silently clipping citations, deduplicates overlapping spans and retains explicit supersedes/conflicts. A small budget may yield no evidence. Human rendering is not a tokenizer budget.
- All readers open SQLite read-only and pin a generation. One bounded-busy writer publishes changes and the active generation in one SQLite transaction. WAL preserves old rows for active readers; unchanged sources reuse parsed chunks. Failure leaves the preceding complete generation visible. SQLite state is disposable: recover from corruption by configuring a **new** database and rebuilding from source. Do not place a live WAL database on a network filesystem.
- Legacy Python `files` and FTS `chunks` column shapes are retained. Schema 0 is rebuilt transactionally into epoch 1 with additive provenance/metadata. First migrate a SQLite backup/private copy, not an uncheckpointed main-file copy. Older unsupported epochs fail closed. The accompanying Python tool must reject nonzero `user_version` before using a Rust-owned index. There is no cross-file database replacement or downgrade migration.
- Source reads/parsing, query/result count, response bytes, file count and wall-clock work are bounded. Defaults appear in `config.example.toml`; configured limits also have hard maxima. Scanning is serial until measurements justify parallel parsing. No speedup/SLA is claimed.

## Opt-in full RAG

Keep the ordinary light executable at `target/release/kb`; build a full-capable executable separately:

```sh
cargo build --release --locked --features full --target-dir target/full
cargo test --locked --features full
```

The full-capable executable also defaults to **light**. Configure a private instance using `config.full.example.toml`, leave `profile = "light"`, and select full explicitly:

```sh
target/full/release/kb --config /absolute/private/full.toml --profile full index
target/full/release/kb --config /absolute/private/full.toml --profile full search "how are backups retained?" --json
target/full/release/kb --config /absolute/private/full.toml --profile full answer "How often do backups run?" --budget-bytes 16384 --json
target/full/release/kb --config /absolute/private/full.toml --profile full mcp
```

Use `.exe` on Windows. Without the Cargo feature, selecting full returns `profile_unavailable`. `answer` under light returns `profile_requires_full`. Missing embeddings, generator, tokenizer requirements or stale vectors are explicit operation errors, never a lexical/raw-text fallback.

### Provider and identity policy

- **LiteLLM:** explicit `provider = "litellm"`, HTTPS base URL (HTTP only for literal loopback), configured embedding/generation aliases, and `api_key_env` naming an environment variable. Credentials are not copied from another application's files or stored in this configuration/index. No ambient proxies, redirects, implicit model selection, retries or downloads.
- LiteLLM contracts report **`provider_identity_only`**. A catalog alias does not attest backend artifacts, immutable revisions or runtime identity. Observed response models and provider-reported usage are reported separately. Undisclosed alias/backend embedding-space changes require an operator-triggered rebuild (`index --force`); catalog membership cannot detect them.
- **Ollama:** literal loopback address with explicit port, installed model identity and pinned local `tokenizer_path`/`tokenizer_sha256`. Embeddings use explicit query/document prefixes and `truncate=false`. Generation uses a pinned raw `chatml-v1` or `plain-v1` template and exact local token counting. Configure the external server with loopback binding and `OLLAMA_NO_CLOUD=1`; this binary does not start, download or reconfigure it.
- Independently hashed artifact bytes are separate from service-reported digests. Hashing a selected file does not attest that a server used it. Strict artifact-hash policies fail when the prerequisite is missing; no hash is invented for a remote alias.
- Without a pinned remote tokenizer, input is bounded in **explicit bytes**, not estimated tokens. Provider output-token caps and observed usage are reported. Upstream LiteLLM chat framing/tokenization remains unknown locally; no exact total-context-fit claim is made.

### Retrieval and answer semantics

Full indexing publishes FTS/provenance and the complete normalized `f32` vector set in the same SQLite transaction. Vectors bind to stable document/chunk keys, source hash, generation and embedding contract—not FTS rowids. Failed embedding/index work leaves the prior generation. Light updates invalidate affected vectors without calling a provider; later full retrieval fails until refreshed.

Filtered cosine scan and lexical BM25 share the same filters before candidate generation. Default fusion is RRF; `weighted_score` uses disclosed per-leg minmax normalization. Results preserve raw leg scores/ranks and fused scores/ranks. A known exact ID constrains both legs after metadata filtering and remains first-ranked; an excluded ID returns no unrelated dense fallback. All corpus vectors are still validated. No raw BM25/cosine comparison, vector service or reranker is hidden in the path.

Answers retrieve an evidence pack and send a delimited, escaped untrusted-data payload to the configured generator with no tools or filesystem capabilities. The strict result contains claims, evidence IDs and quotes. The verifier drops claims with invalid citations/quotes, derives paths/lines/hashes from the pack, checks conflict IDs and rehashes selected sources immediately before output. No surviving claims means abstention. Explicit conflicts involving cited sources (including incoming selected-source relations) block synthesis; unrelated selected-source conflicts stay visible without suppressing an independently supported answer. A resolved model-reported conflict remains a conservative abstention. The provider schema uses a supported grammar subset; local verification still enforces constraints such as unique conflict IDs.

**Mechanical grounding is not semantic entailment.** Exact quotes do not prove that a claim follows, answers the question, is complete, or uses authoritative/current evidence. No answer trace is persisted by default.

## Components boundary

Scribe is a product repository, not an ecosystem, and it is not a workspace package. Its engine — storage, ingest, retrieval, context packs, graph and the MCP server — was extracted on 2026-11-01 to [`Components/crates/knowledge`](../../Components/crates/knowledge/src/lib.rs) ([D-0012](../../project-planning/decisions/0012-component-repositories.md)). This repository keeps the CLI (`src/main.rs`), the product evaluation harness (`src/eval.rs`, `src/answer_eval.rs`) and the fixtures and config documents they pin.

`kb` depends on `knowledge` by path and re-exports it, so commands, configuration and MCP behaviour are unchanged. Tests and unit tests remain split: engine tests live with the package, product tests here.

## Read-only MCP

```sh
kb --config /absolute/private/scribe.toml --profile light mcp
```

Official RMCP 3.6.0 stdio transport. A light server registers only `search`, `get`, `context`; a full server additionally registers read-only `answer`. Profile/roots are fixed at startup and cannot be changed by request arguments. Results include text and `structuredContent`; stdout is protocol-only. No indexing, shell, arbitrary file reads, model downloads, or config writes are exposed. Model calls occur only for explicitly selected full retrieval/answers. Host privileges remain those of the launching user.

Tool argument shapes:

```json
{"name":"search","arguments":{"query":"recovery","filters":{"tag":"rust"},"limit":5}}
{"name":"get","arguments":{"key":"notes:design.md","line_start":1,"line_end":20}}
{"name":"context","arguments":{"query":"recovery","budget_bytes":16384,"limit":10}}
{"name":"answer","arguments":{"question":"How often do backups run?","budget_bytes":16384,"limit":10}}
```

A client launches the binary with the command/arguments above. No particular desktop/IDE client integration is claimed merely from SDK conformance. Cancellation/deadlines are cooperative: SQLite can be interrupted and work checks cancellation between bounded phases, but an already running OS file read cannot be forcibly stopped. Blocking workers retain their concurrency slot until they finish.

## Synthetic evaluation

See `tests/fixtures/README.md` for roots and fixture setup. `kb eval --fixture PATH --json` reports per-query and aggregate document-deduplicated Recall/Precision@5/10, MRR, nDCG@10, exact-ID/filter checks, source-span checks, fixture identity, timing and limitations. The synthetic development/held-out split exercises contracts; it is not a representative quality benchmark or a human-reviewed answer release gate. No semantic answer quality is measured in light.

Full-capable evaluation commands:

```sh
kb --config /absolute/private/full.toml --profile full eval --fixture /absolute/qrels.jsonl --compare --json
kb --config /absolute/private/full.toml --profile full eval --fixture /absolute/answers.jsonl --answers --json
```

Comparison reports per-query light/full results against the same pinned generation/labels. Answer evaluation separates mechanical citation/quote checks, answerable coverage, false/correct abstention and optional **human** semantic judgments. Human reviews bind to the exact stable answer hash; the generator never grades itself.

Without `--release-manifest`, human-reviewed held-out labels and frozen nonvacuous owner thresholds, answer-release acceptance is explicitly blocked. The included answer fixture is synthetic/unreviewed and cannot establish that gate. A manifest pins fixture/source/qrel hashes, owner/rationale, pre-result timestamps and positive coverage/quality thresholds plus a false-abstention cap below 1. Missing/unbound human reviews, synthetic labels and an always-abstaining workflow cannot produce a release-passed claim.

## Instance privacy

Do not commit private configuration, corpus, indexes/WAL sidecars, embeddings, model caches, or credentials. `.gitignore` excludes generated/private state; inspect `cargo package --list --allow-dirty` before a release. Runtime tests use temporary directories and checked-in synthetic data only.

## Verification record — 2026-10-21

- Rust 1.98.1: `cargo test --locked` passes 40 tests on Windows and 41 on Ubuntu/WSL (the additional test covers Unix symlink traversal).
- Actual CLI index/search/get/context and all 52 synthetic evaluation queries ran on both platforms. Linux retrieval also passed inside an isolated network namespace with no model service.
- Real subprocess MCP coverage exercises legacy `2025-11-25` initialization and `2026-07-28` discovery, tools, argument/response limits, stale sources, cancellation, admission limits, deadlines and EOF. A separate private-instance smoke exercised all three tools.
- An active writer was killed before publication; reopening retained the previous complete generation and passed SQLite integrity checking, then a rebuild succeeded. Migration/failure and pinned-reader publication paths are covered by the contract suite.
- `cargo package --list --allow-dirty` contained only implementation, setup material, dependency manifests and synthetic fixtures. No release was published; package metadata still needs the owner's license/repository choices before publication.

These checks establish the exercised light workflows, not a performance SLA, representative retrieval quality, semantic answer support, or named desktop-client compatibility.

## Verification record — 2026-10-22

Rust 1.98.1, locked dependencies:

| Command | Windows | Ubuntu / WSL |
|---|---:|---:|
| `cargo test --locked` | 40 passed | 41 passed |
| `cargo test --locked --features full` | 92 passed | 93 passed |

Full totals include two isolated-child checks for credential/proxy environment safety. Their helper entry points are marked ignored in the parent harness and explicitly executed by their owning tests.

- Real LiteLLM inference exercised full index/search/context/answer, all 52 retrieval comparison queries, and all four synthetic answer cases. Both answerable cases returned quote-resolved claims; the unknown and conflicting cases abstained. The answer run had zero execution errors, valid submitted citations/quotes, and null human semantic metrics. These visible, synthetic labels are not a representative or blinded quality result.
- Real stdio MCP sessions exercised all three default-light tools without the credential environment variable and all four full tools with live inference. The full-capable release binary still returned `profile = "light"`, lexical-only legs and null model contracts when no profile was selected, despite configured full providers.
- A separate explicitly configured private instance was indexed and produced an answer whose returned source hash, line span and normalized quote were independently checked against the actual Markdown file. Its config, corpus, index and evaluation reports are not distributed.
- Regression coverage includes failed embedding-batch rollback, edit/rename/delete followed by light invalidation and full refresh, filtered-out exact IDs without unrelated dense fallback, incoming one-sided conflicts, and unrelated conflicts that must not censor independently cited claims.
- Live inference was verified through LiteLLM. Ollama was exercised against synthetic loopback protocol fixtures, not an installed local model. No artifact/runtime attestation, semantic entailment, performance improvement, human-reviewed answer-release acceptance, or named desktop-client compatibility is claimed.
