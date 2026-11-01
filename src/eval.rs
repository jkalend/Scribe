use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
    time::Instant,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    config::Config,
    db,
    error::{Error, Result},
    ingest,
    model::{Document, Envelope, Filters, Hit, SearchRequest},
    retrieve,
};

const MAX_FIXTURE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Span {
    source: String,
    line_start: usize,
    line_end: usize,
    #[serde(default)]
    quote: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    id: String,
    query: String,
    #[serde(default)]
    filters: Filters,
    relevant: Vec<String>,
    split: String,
    category: String,
    #[serde(default)]
    expected_spans: Vec<Span>,
}

#[derive(Clone, Debug, Serialize)]
struct Metrics {
    recall_at_5: Option<f64>,
    recall_at_10: Option<f64>,
    precision_at_5: f64,
    precision_at_10: f64,
    mrr: Option<f64>,
    ndcg_at_10: Option<f64>,
}

fn metrics(ranked: &[String], relevant: &BTreeSet<String>) -> Metrics {
    let count = |k: usize| {
        ranked
            .iter()
            .take(k)
            .filter(|key| relevant.contains(*key))
            .count()
    };
    let recall = |k| {
        if relevant.is_empty() {
            None
        } else {
            Some(count(k) as f64 / relevant.len() as f64)
        }
    };
    let reciprocal = ranked
        .iter()
        .position(|key| relevant.contains(key))
        .map(|i| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0);
    let dcg: f64 = ranked
        .iter()
        .take(10)
        .enumerate()
        .filter(|(_, key)| relevant.contains(*key))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum();
    let ideal: f64 = (0..relevant.len().min(10))
        .map(|i| 1.0 / ((i + 2) as f64).log2())
        .sum();
    Metrics {
        recall_at_5: recall(5),
        recall_at_10: recall(10),
        precision_at_5: count(5) as f64 / 5.0,
        precision_at_10: count(10) as f64 / 10.0,
        mrr: if relevant.is_empty() {
            None
        } else {
            Some(reciprocal)
        },
        ndcg_at_10: if ideal == 0.0 {
            None
        } else {
            Some(dcg / ideal)
        },
    }
}

fn scope_matches(hit: &Hit, filters: &Filters) -> bool {
    let same = |filter: &Option<String>, actual: &str| {
        filter.as_deref().is_none_or(|wanted| wanted == actual)
    };
    same(&filters.root, &hit.root_key)
        && same(&filters.role, &hit.role)
        && same(&filters.status, &hit.status)
        && same(&filters.project, &hit.project)
        && same(&filters.doc_type, &hit.doc_type)
        && filters
            .tag
            .as_ref()
            .is_none_or(|tag| hit.tags.contains(tag))
        && filters
            .after
            .as_ref()
            .is_none_or(|after| hit.updated.as_ref().is_some_and(|updated| updated >= after))
        && filters.before.as_ref().is_none_or(|before| {
            hit.updated
                .as_ref()
                .is_some_and(|updated| updated <= before)
        })
}

pub(crate) fn resolve_label<'a>(label: &str, docs: &'a [Document]) -> Result<&'a Document> {
    let scoped_path = label.split_once(':');
    let mut matches = docs.iter().filter(|doc| {
        label == doc.meta.id
            || scoped_path.is_some_and(|(root, path)| root == doc.root_key && path == doc.path)
    });
    let first = matches.next().ok_or_else(|| {
        Error::new(
            "invalid_fixture",
            format!("Qrel source {label:?} is not indexed"),
        )
    })?;
    if matches.next().is_some() {
        return Err(Error::new(
            "invalid_fixture",
            format!("Qrel source {label:?} is ambiguous; use root:path"),
        ));
    }
    Ok(first)
}

fn load_fixture(path: &Path) -> Result<(Vec<u8>, Vec<Query>)> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_FIXTURE_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_FIXTURE_BYTES {
        return Err(Error::new(
            "fixture_too_large",
            "Evaluation fixture exceeds 16 MiB",
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::new("invalid_fixture", "Fixture must be UTF-8 JSONL"))?;
    let mut rows = Vec::new();
    let mut ids = BTreeSet::new();
    for (line, raw) in text.lines().enumerate() {
        if raw.trim().is_empty() {
            continue;
        }
        let row: Query = serde_json::from_str(raw).map_err(|error| {
            Error::new(
                "invalid_fixture",
                format!("Fixture line {}: {error}", line + 1),
            )
        })?;
        if row.id.trim().is_empty()
            || row.query.trim().is_empty()
            || row.category.trim().is_empty()
            || !matches!(row.split.as_str(), "development" | "held_out")
            || !ids.insert(row.id.clone())
        {
            return Err(Error::new(
                "invalid_fixture",
                format!(
                    "Fixture line {} has an empty/duplicate ID, empty query/category, or unsupported split",
                    line + 1
                ),
            ));
        }
        if row.relevant.iter().collect::<BTreeSet<_>>().len() != row.relevant.len() {
            return Err(Error::new(
                "invalid_fixture",
                format!("Fixture row {:?} repeats a relevant label", row.id),
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(Error::new(
            "invalid_fixture",
            "Evaluation fixture has no query rows",
        ));
    }
    Ok((bytes, rows))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (total, count) = values.fold((0.0, 0usize), |(sum, n), value| (sum + value, n + 1));
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

fn aggregate<'a>(rows: impl Iterator<Item = &'a Value> + Clone) -> Value {
    let metric = |name: &str| mean(rows.clone().filter_map(|row| row["metrics"][name].as_f64()));
    let bool_rate = |name: &str| {
        mean(
            rows.clone()
                .filter_map(|row| row[name].as_bool().map(|ok| if ok { 1.0 } else { 0.0 })),
        )
    };
    json!({"queries":rows.clone().count(),"answerable_queries":rows.clone().filter(|r| r["relevant_sources"].as_array().is_some_and(|a| !a.is_empty())).count(),
        "recall_at_5":metric("recall_at_5"),"recall_at_10":metric("recall_at_10"),"precision_at_5":metric("precision_at_5"),"precision_at_10":metric("precision_at_10"),
        "mrr":metric("mrr"),"ndcg_at_10":metric("ndcg_at_10"),"scope_correctness":bool_rate("scope_correct"),"exact_id_pass_rate":bool_rate("exact_id_pass"),
        "exact_id_queries":rows.clone().filter(|r| !r["exact_id_pass"].is_null()).count(),
        "expected_spans":rows.clone().filter_map(|r| r["citation_spans"]["expected"].as_u64()).sum::<u64>(),
        "covered_spans":rows.clone().filter_map(|r| r["citation_spans"]["covered"].as_u64()).sum::<u64>()})
}

pub fn run(config: &Config, fixture: &Path) -> Result<Envelope<Value>> {
    let (fixture_bytes, queries) = load_fixture(fixture)?;
    let conn = db::open_reader(config)?;
    let generation = db::generation(&conn)?;
    let docs = db::load_documents(&conn)?;
    // All queries reuse this pinned read-only connection, including concurrent publication.
    let mut source_cache: BTreeMap<String, String> = BTreeMap::new();
    let mut rows = Vec::new();
    let mut latencies = Vec::new();
    let mut execution = None;
    for query in &queries {
        let relevant: BTreeSet<String> = query
            .relevant
            .iter()
            .map(|label| resolve_label(label, &docs).map(|doc| doc.key.clone()))
            .collect::<Result<_>>()?;
        for span in &query.expected_spans {
            let doc = resolve_label(&span.source, &docs)?;
            if !relevant.contains(&doc.key)
                || span.line_start == 0
                || span.line_start > span.line_end
            {
                return Err(Error::new(
                    "invalid_fixture",
                    format!("Invalid expected span in row {:?}", query.id),
                ));
            }
            if !source_cache.contains_key(&doc.key) {
                let path = config.resolve_source(&doc.root_key, &doc.path)?;
                let mut raw = Vec::new();
                File::open(path)?
                    .take(config.limits.max_file_bytes as u64 + 1)
                    .read_to_end(&mut raw)?;
                if raw.len() > config.limits.max_file_bytes {
                    return Err(Error::new(
                        "source_too_large",
                        "Evaluation citation source exceeds configured file size",
                    ));
                }
                if ingest::digest(&raw) != doc.sha256 {
                    return Err(Error::new(
                        "stale_source",
                        format!(
                            "Evaluation citation source {}:{} changed; run index",
                            doc.root_key, doc.path
                        ),
                    ));
                }
                source_cache.insert(doc.key.clone(), ingest::normalized_source(&raw)?);
            }
            let text = &source_cache[&doc.key];
            if span.line_end > text.lines().count() {
                return Err(Error::new(
                    "invalid_fixture",
                    format!("Expected span extends past source in row {:?}", query.id),
                ));
            }
            let excerpt = text
                .lines()
                .skip(span.line_start - 1)
                .take(span.line_end - span.line_start + 1)
                .collect::<Vec<_>>()
                .join("\n");
            if span.quote.as_ref().is_some_and(|quote| quote != &excerpt) {
                return Err(Error::new(
                    "invalid_fixture",
                    format!(
                        "Expected quote does not equal its whole-source lines in row {:?}",
                        query.id
                    ),
                ));
            }
        }
        let started = Instant::now();
        let result = retrieve::search_on(
            config,
            &conn,
            &SearchRequest {
                query: query.query.clone(),
                filters: query.filters.clone(),
                limit: 10,
            },
        )?;
        let latency = started.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency);
        if result.index_generation != generation {
            return Err(Error::new(
                "generation_changed",
                "Index generation changed during evaluation; rerun against a stable index",
            ));
        }
        let metadata = json!({
            "profile":result.profile,"retrieval_legs":result.retrieval_legs,
            "embedding_contract_id":result.embedding_contract_id,
            "generator_contract_id":result.generator_contract_id,
            "model_contracts":result.model_contracts,"fusion_method":result.fusion_method
        });
        if execution.as_ref().is_some_and(|previous: &Value| {
            [
                "profile",
                "embedding_contract_id",
                "generator_contract_id",
                "fusion_method",
            ]
            .iter()
            .any(|key| previous[*key] != metadata[*key])
        }) {
            return Err(Error::new(
                "model_contract_changed",
                "Model contract changed during evaluation",
            ));
        }
        execution = Some(metadata.clone());
        let hits = result.data;
        let mut seen = BTreeSet::new();
        let ranked: Vec<String> = hits
            .iter()
            .filter(|h| seen.insert(h.document_key.clone()))
            .map(|h| h.document_key.clone())
            .collect();
        let scores = metrics(&ranked, &relevant);
        let scope_violations: Vec<&str> = hits
            .iter()
            .filter(|hit| !scope_matches(hit, &query.filters))
            .map(|hit| hit.chunk_key.as_str())
            .collect();
        let has_exact_id = docs.iter().any(|doc| doc.meta.id == query.query);
        let exact_keys: BTreeSet<&str> = docs
            .iter()
            .filter(|doc| doc.meta.id == query.query && relevant.contains(&doc.key))
            .map(|doc| doc.key.as_str())
            .collect();
        let exact_id_pass = if !has_exact_id {
            None
        } else if exact_keys.is_empty() {
            Some(hits.is_empty())
        } else {
            Some(
                hits.first()
                    .is_some_and(|hit| exact_keys.contains(hit.document_key.as_str())),
            )
        };
        let mut spans = Vec::new();
        for span in &query.expected_spans {
            let doc = resolve_label(&span.source, &docs)?;
            let covered = hits.iter().any(|hit| {
                if hit.document_key != doc.key
                    || hit.source_sha256 != doc.sha256
                    || hit.line_start > span.line_start
                    || hit.line_end < span.line_end
                {
                    return false;
                }
                let expected_body = source_cache[&doc.key]
                    .lines()
                    .skip(hit.line_start.saturating_sub(1))
                    .take(hit.line_end.saturating_sub(hit.line_start) + 1)
                    .collect::<Vec<_>>()
                    .join("\n");
                // A single terminal newline is immaterial to a line-oriented excerpt.
                hit.line_start > 0
                    && hit.body.strip_suffix('\n').unwrap_or(&hit.body) == expected_body
            });
            spans.push(json!({"source":span.source,"line_start":span.line_start,"line_end":span.line_end,"covered":covered}));
        }
        let covered = spans.iter().filter(|span| span["covered"] == true).count();
        let ranked_chunks: Vec<Value> = hits.iter().map(|hit| json!({
            "document_key":hit.document_key,"chunk_key":hit.chunk_key,"root_key":hit.root_key,
            "path":hit.path,"source_sha256":hit.source_sha256,"line_start":hit.line_start,"line_end":hit.line_end,
            "legs":hit.legs,"fused_rank":hit.fused_rank,"fused_score":hit.fused_score
        })).collect();
        rows.push(json!({"id":query.id,"query":query.query,"filters":query.filters,"split":query.split,"category":query.category,
            "relevant_sources":relevant,"ranked_sources":ranked,"returned_chunks":hits.len(),"metrics":scores,
            "scope_correct":scope_violations.is_empty(),"scope_violations":scope_violations,"exact_id_pass":exact_id_pass,
            "citation_spans":{"expected":spans.len(),"covered":covered,"spans":spans},
            "execution":metadata,"hits":ranked_chunks,
            "unanswerable_no_results":if relevant.is_empty() { Some(hits.is_empty()) } else { None },"latency_ms":latency}));
    }
    let mut by_split = BTreeMap::new();
    let mut by_category = BTreeMap::new();
    for split in ["development", "held_out"] {
        by_split.insert(
            split,
            aggregate(rows.iter().filter(|row| row["split"] == split)),
        );
    }
    for category in queries
        .iter()
        .map(|query| query.category.as_str())
        .collect::<BTreeSet<_>>()
    {
        by_category.insert(
            category,
            aggregate(rows.iter().filter(|row| row["category"] == category)),
        );
    }
    latencies.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        latencies[((latencies.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(latencies.len() - 1)]
    };
    let corpus_manifest: Vec<Value> = docs
        .iter()
        .map(|doc| json!({"root":doc.root_key,"path":doc.path,"sha256":doc.sha256}))
        .collect();
    let mut ordered_manifest = corpus_manifest;
    ordered_manifest.sort_by(|a, b| {
        (a["root"].as_str(), a["path"].as_str()).cmp(&(b["root"].as_str(), b["path"].as_str()))
    });
    let mut report = config.envelope(
        generation,
        json!({"report_version":1,"fixture_sha256":ingest::digest(&fixture_bytes),"query_count":queries.len(),
        "corpus_sha256":ingest::digest(&serde_json::to_vec(&ordered_manifest)?),"documents":docs.len(),
        "metric_contract":{"retrieval_unit":"unique source documents, first-ranked chunk wins","candidate_chunks":10,"precision_denominator":"fixed cutoff k","relevance":"binary",
            "unanswerable":"recall/MRR/nDCG undefined (null), excluded from their macro means; precision is zero","scope":"all returned chunks, inclusive date bounds","citation_spans":"whole expected span contained in a returned chunk with exact current-source body; not semantic support"},
        "aggregate":aggregate(rows.iter()),"by_split":by_split,"by_category":by_category,"rows":rows,
        "environment":{"scribe_version":env!("CARGO_PKG_VERSION"),"target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,"sqlite_version":rusqlite::version(),
            "dependency_lock_sha256":ingest::digest(include_bytes!("../Cargo.lock")),"rustc_version":option_env!("RUSTC_VERSION"),"os_release":Value::Null,"peak_memory_bytes":Value::Null},
        "latency_ms":{"median":percentile(0.5),"p95":percentile(0.95)},
        "limitations":["Checked-in relevance labels are synthetic; external fixture provenance is not attested. No production-quality claim.","Held-out labels are visible in this repository; no blinded evaluation or answer-release claim.",
            "Lexical retrieval may miss paraphrases; answer correctness, semantic entailment, conflict handling and abstention are not evaluated.",
            "Metrics are deterministic for fixed inputs; wall-clock latency is observational and cache state is uncontrolled.","Compiler version (unless supplied at build time) and OS release are not measured; peak memory is not measured.",
            "Only ten candidate chunks are requested; duplicate chunks can reduce the distinct-source pool. No source-level overfetch is hidden."]}),
    );
    let execution = execution.expect("nonempty validated fixture");
    report.retrieval_legs = serde_json::from_value(execution["retrieval_legs"].clone())?;
    report.embedding_contract_id =
        serde_json::from_value(execution["embedding_contract_id"].clone())?;
    report.generator_contract_id =
        serde_json::from_value(execution["generator_contract_id"].clone())?;
    report.model_contracts = execution["model_contracts"].clone();
    report.fusion_method = serde_json::from_value(execution["fusion_method"].clone())?;
    Ok(report)
}

/// Evaluate identical fixtures in both selected stacks; light never prepares a model.
#[cfg(feature = "full")]
pub fn compare(config: &Config, fixture: &Path) -> Result<Envelope<Value>> {
    let mut light_config = config.clone();
    light_config.profile = crate::config::Profile::Light;
    let mut full_config = config.clone();
    full_config.profile = crate::config::Profile::Full;
    let light = run(&light_config, fixture)?;
    let full = run(&full_config, fixture)?;
    if light.index_generation != full.index_generation
        || light.data["fixture_sha256"] != full.data["fixture_sha256"]
        || light.data["corpus_sha256"] != full.data["corpus_sha256"]
    {
        return Err(Error::new(
            "generation_changed",
            "Comparison requires the same fixture and source generation",
        ));
    }
    let light_rows = light.data["rows"].as_array().expect("evaluation rows");
    let full_rows = full.data["rows"].as_array().expect("evaluation rows");
    let differences: Vec<Value> = light_rows.iter().zip(full_rows).map(|(a,b)| {
        let deltas: BTreeMap<&str, Option<f64>> = ["recall_at_5","recall_at_10","precision_at_5","precision_at_10","mrr","ndcg_at_10"].into_iter()
            .map(|name| (name,a["metrics"][name].as_f64().zip(b["metrics"][name].as_f64()).map(|(x,y)| y-x))).collect();
        json!({"id":a["id"],"split":a["split"],"category":a["category"],"light_sources":a["ranked_sources"],
            "full_sources":b["ranked_sources"],"ranking_changed":a["ranked_sources"]!=b["ranked_sources"],
            "full_minus_light":deltas,"light_execution":a["execution"],"full_execution":b["execution"]})
    }).collect();
    let mut report = full;
    let full_data = std::mem::take(&mut report.data);
    report.data = json!({"report_version":1,"comparison":"full minus light, same fixture/source generation",
        "light":light,"full":full_data,"rows":differences,"release_acceptance":false,
        "limitations":["Retrieval comparison does not establish answer quality or V1 release acceptance."]});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_metrics_have_fixed_precision_and_correct_discounts() {
        let relevant = BTreeSet::from(["a".into(), "b".into()]);
        let ranked = vec!["x".into(), "a".into(), "b".into()];
        let result = metrics(&ranked, &relevant);
        assert_eq!(result.recall_at_5, Some(1.0));
        assert_eq!(result.precision_at_5, 0.4);
        assert_eq!(result.precision_at_10, 0.2);
        assert_eq!(result.mrr, Some(0.5));
        let expected = (1.0 / 3.0_f64.log2() + 0.5) / (1.0 + 1.0 / 3.0_f64.log2());
        assert!((result.ndcg_at_10.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn unanswerable_metrics_are_not_fabricated_perfect_recall() {
        let result = metrics(&[], &BTreeSet::new());
        assert_eq!(result.recall_at_10, None);
        assert_eq!(result.mrr, None);
        assert_eq!(result.ndcg_at_10, None);
        assert_eq!(result.precision_at_10, 0.0);
        let missed = metrics(&[], &BTreeSet::from(["a".into()]));
        assert_eq!(missed.recall_at_10, Some(0.0));
        assert_eq!(missed.mrr, Some(0.0));
    }

    #[test]
    fn exact_label_resolution_rejects_ambiguous_ids() {
        let doc = |path: &str| Document {
            root_key: "test".into(),
            path: path.into(),
            key: path.into(),
            sha256: String::new(),
            mtime: 0.0,
            size: 0,
            meta: crate::model::Metadata {
                id: "DUP".into(),
                ..Default::default()
            },
            chunks: vec![],
            links: vec![],
        };
        let docs = vec![doc("a.md"), doc("b.md")];
        assert!(resolve_label("DUP", &docs).is_err());
        assert_eq!(resolve_label("test:a.md", &docs).unwrap().path, "a.md");
        assert!(resolve_label("MISSING", &docs).is_err());
    }

    #[test]
    fn checked_in_fixture_is_labeled_and_has_a_holdout() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qrels.jsonl");
        let (_, rows) = load_fixture(&path).unwrap();
        assert!(rows.len() >= 40);
        assert!(
            rows.iter()
                .any(|row| row.split == "held_out" && !row.relevant.is_empty())
        );
        assert!(rows.iter().any(|row| row.relevant.is_empty()));
        assert!(rows.iter().any(|row| !row.expected_spans.is_empty()));
        assert!(rows.iter().any(|row| row.filters.root.is_some()));
    }

    #[test]
    fn synthetic_evaluation_reports_rows_scope_and_exact_ids() {
        let temp = tempfile::tempdir().unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let config = Config {
            database: temp.path().join("index.sqlite"),
            profile: Default::default(),
            full: None,
            limits: Default::default(),
            roots: ["canonical", "research", "audit"]
                .into_iter()
                .map(|key| {
                    (
                        key.into(),
                        crate::config::Root {
                            path: fixtures.join(key).canonicalize().unwrap(),
                            role: key.into(),
                            trust: "synthetic".into(),
                            ignore_dirs: vec![],
                        },
                    )
                })
                .collect(),
        };
        db::index(&config, false).unwrap();
        let report = run(&config, &fixtures.join("qrels.jsonl")).unwrap();
        assert!(report.data["query_count"].as_u64().unwrap() >= 40);
        assert_eq!(
            report.data["rows"].as_array().unwrap().len() as u64,
            report.data["query_count"].as_u64().unwrap()
        );
        assert_eq!(report.data["aggregate"]["scope_correctness"], 1.0);
        assert_eq!(report.data["aggregate"]["exact_id_pass_rate"], 1.0);
        assert!(
            report.data["by_split"]["held_out"]["queries"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(report.data["fixture_sha256"].as_str().unwrap().len(), 64);
        // No recall threshold: this small lexical synthetic corpus is a harness, not a quality claim.
    }
}
