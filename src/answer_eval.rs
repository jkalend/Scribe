//! Mechanical diagnostics are not semantic judgments. Human reviews bind to the
//! exact stable answer payload hash; no generator or judge grades its own output.
use knowledge::{
    config::Config,
    db,
    error::{Error, Result},
    ingest,
    model::{AnswerRequest, Envelope, Filters},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_BYTES: u64 = 16 * 1024 * 1024;
fn synthetic_default() -> bool {
    true
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HumanReview {
    reviewer: String,
    reviewed_at_unix_ms: u64,
    answer_sha256: String,
    supported_claims: usize,
    total_claims: usize,
    answer_relevant: bool,
    conflict_handling_correct: bool,
    abstention_correct: bool,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    question: String,
    #[serde(default)]
    filters: Filters,
    split: String,
    category: String,
    expected_sources: Vec<String>,
    expected_facts: Vec<String>,
    #[serde(default = "synthetic_default")]
    synthetic: bool,
    #[serde(default)]
    human_review: Option<HumanReview>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelReview {
    reviewer: String,
    reviewed_at_unix_ms: u64,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Thresholds {
    min_supported_claim_rate: f64,
    min_answerable_coverage: f64,
    max_false_abstention: f64,
    min_correct_abstention: f64,
    min_citation_resolution: f64,
    min_quote_pass: f64,
    min_answer_relevance: f64,
    min_conflict_handling: f64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseManifest {
    version: u32,
    fixture_sha256: String,
    source_sha256: String,
    qrels_sha256: String,
    frozen_at_unix_ms: u64,
    owner: String,
    rationale: String,
    human_reviewed_labels: BTreeMap<String, LabelReview>,
    thresholds: Thresholds,
}
fn bytes(path: &Path) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    File::open(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut result)?;
    if result.len() as u64 > MAX_BYTES {
        return Err(Error::new(
            "fixture_too_large",
            "Evaluation input exceeds 16 MiB",
        ));
    }
    Ok(result)
}
fn cases(raw: &[u8]) -> Result<Vec<Case>> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| Error::new("invalid_fixture", "Answer fixture must be UTF-8 JSONL"))?;
    let mut result = Vec::new();
    let mut ids = BTreeSet::new();
    for (line, raw) in text
        .lines()
        .enumerate()
        .filter(|(_, s)| !s.trim().is_empty())
    {
        let row: Case = serde_json::from_str(raw).map_err(|e| {
            Error::new(
                "invalid_fixture",
                format!("Answer fixture line {}: {e}", line + 1),
            )
        })?;
        if row.id.trim().is_empty()
            || row.question.trim().is_empty()
            || !ids.insert(row.id.clone())
            || !matches!(row.split.as_str(), "development" | "held_out")
            || !matches!(
                row.category.as_str(),
                "answerable" | "unanswerable" | "conflicting"
            )
            || row.expected_sources.iter().any(|s| s.trim().is_empty())
            || row.expected_facts.iter().any(|s| s.trim().is_empty())
            || row.expected_sources.iter().collect::<BTreeSet<_>>().len()
                != row.expected_sources.len()
            || (row.category != "unanswerable"
                && (row.expected_sources.is_empty() || row.expected_facts.is_empty()))
        {
            return Err(Error::new(
                "invalid_fixture",
                format!("Invalid or duplicate answer case on line {}", line + 1),
            ));
        }
        result.push(row);
    }
    if result.is_empty() {
        return Err(Error::new("invalid_fixture", "Answer fixture has no cases"));
    }
    Ok(result)
}
fn ratio(n: u64, d: u64) -> Option<f64> {
    if d == 0 {
        None
    } else {
        Some(n as f64 / d as f64)
    }
}
fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { None } else { Some(sum / n as f64) }
}
fn aggregate<'a>(rows: impl Iterator<Item = &'a Value> + Clone) -> Value {
    let rate = |name: &str| {
        mean(rows.clone().filter_map(|r| {
            r["mechanical"][name]
                .as_bool()
                .map(|v| if v { 1.0 } else { 0.0 })
        }))
    };
    let sum = |name: &str| {
        rows.clone()
            .filter_map(|r| r["verification"][name].as_u64())
            .sum::<u64>()
    };
    let semantic = |name: &str| {
        mean(
            rows.clone()
                .filter_map(|r| r["human_semantic"][name].as_f64()),
        )
    };
    let reviewed = rows
        .clone()
        .filter(|r| !r["human_semantic"].is_null())
        .count();
    let supported = rows
        .clone()
        .filter_map(|r| r["human_semantic"]["supported_claims"].as_u64())
        .sum();
    let claims = rows
        .clone()
        .filter_map(|r| r["human_semantic"]["total_claims"].as_u64())
        .sum();
    json!({"cases":rows.clone().count(),"errors":rows.clone().filter(|r| !r["error"].is_null()).count(),
        "answerable_coverage":rate("answerable_coverage"),"false_abstention":rate("false_abstention"),
        "correct_abstention":rate("correct_abstention"),"citation_resolution":ratio(sum("citations_resolved"),sum("citations_submitted")),
        "quote_pass":ratio(sum("quotes_verified"),sum("quotes_submitted")),"claims_submitted":sum("claims_submitted"),
        "claims_verified":sum("claims_verified"),"claims_dropped":sum("claims_dropped"),
        "citations_submitted":sum("citations_submitted"),"citations_resolved":sum("citations_resolved"),
        "quotes_submitted":sum("quotes_submitted"),"quotes_verified":sum("quotes_verified"),
        "human_semantic":{"reviewed_cases":reviewed,"supported_claim_rate":ratio(supported,claims),
            "answer_relevance":semantic("answer_relevance"),"conflict_handling":semantic("conflict_handling"),
            "correct_abstention":semantic("correct_abstention"),"abstention_accuracy":semantic("abstention_accuracy")}})
}
/// Stable, reviewable generated payload: excludes timing and resolution timestamps.
fn reviewed_payload(data: &Value) -> Value {
    let claims: Vec<Value> = data["claims"].as_array().into_iter().flatten().map(|claim| {
        let citations: Vec<Value> = claim["citations"].as_array().into_iter().flatten().map(|c| json!({
            "evidence_id":c["evidence_id"],"quote":c["quote"],"document_key":c["provenance"]["document_key"],
            "chunk_key":c["provenance"]["chunk_key"],"source_sha256":c["provenance"]["source_sha256"]})).collect();
        json!({"text":claim["text"],"citations":citations})
    }).collect();
    json!({"abstained":data["abstained"],"abstain_reason":data["abstain_reason"],"claims":claims,
        "conflicts":data["conflicts"],"corpus_conflicts":data["corpus_conflicts"]})
}
fn semantic_review(
    case: &Case,
    data: &Value,
    hash: &str,
    evaluated_at: u64,
) -> (Value, Option<String>) {
    let Some(review) = &case.human_review else {
        return (Value::Null, Some("unreviewed".into()));
    };
    let count = data["claims"].as_array().map_or(0, Vec::len);
    if review.reviewer.trim().is_empty()
        || review.reviewed_at_unix_ms == 0
        || review.reviewed_at_unix_ms > evaluated_at
        || review.answer_sha256 != hash
        || review.total_claims != count
        || review.supported_claims > count
    {
        return (
            Value::Null,
            Some(
                "human review missing identity/time or not bound to this exact answer/claim count"
                    .into(),
            ),
        );
    }
    (
        json!({"reviewer":review.reviewer,"reviewed_at_unix_ms":review.reviewed_at_unix_ms,"supported_claims":review.supported_claims,
        "total_claims":count,"supported_claim_rate":ratio(review.supported_claims as u64,count as u64),
        "answer_relevance":if case.category == "unanswerable" {None} else {Some(if review.answer_relevant {1.0} else {0.0})},
        "conflict_handling":if case.category == "conflicting" {Some(if review.conflict_handling_correct {1.0} else {0.0})} else {None},
        "correct_abstention":if case.category == "unanswerable" {Some(if review.abstention_correct {1.0} else {0.0})} else {None},
        "abstention_accuracy":if review.abstention_correct {1.0} else {0.0}}),
        None,
    )
}
fn release(
    manifest: Option<&Path>,
    cases: &[Case],
    rows: &[Value],
    hashes: &Value,
    evaluated_at: u64,
) -> Result<Value> {
    let mut blocks = Vec::<String>::new();
    let Some(path) = manifest else {
        return Ok(
            json!({"release_acceptance":false,"manifest_sha256":null,"blocked_reasons":["No frozen owner-selected release manifest supplied" ]}),
        );
    };
    let raw = bytes(path)?;
    let manifest_hash = ingest::digest(&raw);
    let parsed = serde_json::from_slice::<ReleaseManifest>(&raw);
    let m = match parsed {
        Ok(m) => m,
        Err(e) => {
            return Ok(
                json!({"release_acceptance":false,"manifest_sha256":manifest_hash,"blocked_reasons":[format!("Invalid release manifest: {e}")]}),
            );
        }
    };
    if m.version != 1 || m.owner.trim().is_empty() || m.rationale.trim().is_empty() {
        blocks.push("Manifest requires version 1, owner, and threshold rationale".into());
    }
    if Some(m.fixture_sha256.as_str()) != hashes["fixture_sha256"].as_str()
        || Some(m.source_sha256.as_str()) != hashes["source_sha256"].as_str()
        || Some(m.qrels_sha256.as_str()) != hashes["qrels_sha256"].as_str()
    {
        blocks.push("Frozen fixture/source/qrel hashes do not match this evaluation".into());
    }
    if m.frozen_at_unix_ms == 0 || m.frozen_at_unix_ms >= evaluated_at {
        blocks.push("Manifest must be frozen strictly before evaluation starts".into());
    }
    let heldout: Vec<_> = cases.iter().filter(|c| c.split == "held_out").collect();
    for category in ["answerable", "unanswerable", "conflicting"] {
        if !heldout.iter().any(|c| c.category == category) {
            blocks.push(format!("Missing held-out {category} case"));
        }
    }
    for c in &heldout {
        if c.synthetic {
            blocks.push(format!("Case {} is synthetic; no V1 release claim", c.id));
        }
        if !m.human_reviewed_labels.get(&c.id).is_some_and(|r| {
            !r.reviewer.trim().is_empty()
                && r.reviewed_at_unix_ms > 0
                && r.reviewed_at_unix_ms <= m.frozen_at_unix_ms
        }) {
            blocks.push(format!(
                "Case {} lacks pre-freeze human-reviewed labels",
                c.id
            ));
        }
        let row = rows
            .iter()
            .find(|r| r["id"] == c.id)
            .expect("case result exists");
        if row["human_semantic"].is_null() {
            blocks.push(format!(
                "Case {} lacks actual human judgments bound to this answer",
                c.id
            ));
        }
        if !row["error"].is_null() {
            blocks.push(format!("Case {} failed answer execution", c.id));
        }
    }
    let t = &m.thresholds;
    let minima = [
        t.min_supported_claim_rate,
        t.min_answerable_coverage,
        t.min_correct_abstention,
        t.min_citation_resolution,
        t.min_quote_pass,
        t.min_answer_relevance,
        t.min_conflict_handling,
    ];
    if minima
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
        || !t.max_false_abstention.is_finite()
        || !(0.0..1.0).contains(&t.max_false_abstention)
    {
        blocks.push("Owner thresholds must be non-vacuous: minima in (0,1], false-abstention maximum in [0,1)".into());
    }
    let stats = aggregate(rows.iter().filter(|r| r["split"] == "held_out"));
    for (name, actual, threshold, minimum) in [
        (
            "supported_claim_rate",
            stats["human_semantic"]["supported_claim_rate"].as_f64(),
            t.min_supported_claim_rate,
            true,
        ),
        (
            "answerable_coverage",
            stats["answerable_coverage"].as_f64(),
            t.min_answerable_coverage,
            true,
        ),
        (
            "false_abstention",
            stats["false_abstention"].as_f64(),
            t.max_false_abstention,
            false,
        ),
        (
            "correct_abstention",
            stats["correct_abstention"].as_f64(),
            t.min_correct_abstention,
            true,
        ),
        (
            "human_correct_abstention",
            stats["human_semantic"]["correct_abstention"].as_f64(),
            t.min_correct_abstention,
            true,
        ),
        (
            "citation_resolution",
            stats["citation_resolution"].as_f64(),
            t.min_citation_resolution,
            true,
        ),
        (
            "quote_pass",
            stats["quote_pass"].as_f64(),
            t.min_quote_pass,
            true,
        ),
        (
            "answer_relevance",
            stats["human_semantic"]["answer_relevance"].as_f64(),
            t.min_answer_relevance,
            true,
        ),
        (
            "conflict_handling",
            stats["human_semantic"]["conflict_handling"].as_f64(),
            t.min_conflict_handling,
            true,
        ),
    ] {
        if !actual.is_some_and(|value| {
            if minimum {
                value >= threshold
            } else {
                value <= threshold
            }
        }) {
            blocks.push(format!(
                "Owner threshold {name} failed or metric unavailable"
            ));
        }
    }
    Ok(
        json!({"release_acceptance":blocks.is_empty(),"manifest_sha256":manifest_hash,"frozen_at_unix_ms":m.frozen_at_unix_ms,
        "owner":m.owner,"rationale":m.rationale,"thresholds":m.thresholds,"held_out_metrics":stats,"blocked_reasons":blocks}),
    )
}

pub fn run(
    config: &Config,
    fixture: &Path,
    release_manifest: Option<&Path>,
) -> Result<Envelope<Value>> {
    if !config.profile.is_full() {
        return Err(Error::new(
            "profile_requires_full",
            "Answer evaluation requires the full profile",
        ));
    }
    let evaluated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::new("invalid_clock", "Evaluation clock precedes Unix epoch"))?
        .as_millis() as u64;
    let raw = bytes(fixture)?;
    let cases = cases(&raw)?;
    let conn = db::open_reader(config)?;
    let generation = db::generation(&conn)?;
    let docs = db::load_documents(&conn)?;
    let mut sources: Vec<Value> = docs
        .iter()
        .map(|d| json!({"root":d.root_key,"path":d.path,"sha256":d.sha256}))
        .collect();
    sources.sort_by(|a, b| {
        (a["root"].as_str(), a["path"].as_str()).cmp(&(b["root"].as_str(), b["path"].as_str()))
    });
    let qrels: Vec<Value> = cases.iter().map(|c| json!({"id":c.id,"question":c.question,"filters":c.filters,"split":c.split,"category":c.category,"expected_sources":c.expected_sources,"expected_facts":c.expected_facts,"synthetic":c.synthetic})).collect();
    let hashes = json!({"fixture_sha256":ingest::digest(&raw),"source_sha256":ingest::digest(&serde_json::to_vec(&sources)?),"qrels_sha256":ingest::digest(&serde_json::to_vec(&qrels)?)});
    let mut rows = Vec::new();
    let mut report = config.envelope(generation, Value::Null);
    let mut contracts = None;
    for case in &cases {
        knowledge::mcp::checkpoint()?;
        let expected: BTreeSet<String> = case
            .expected_sources
            .iter()
            .map(|s| crate::eval::resolve_label(s, &docs).map(|d| d.key.clone()))
            .collect::<Result<_>>()?;
        let output = knowledge::answer::run(
            config,
            &AnswerRequest {
                question: case.question.clone(),
                filters: case.filters.clone(),
                limit: config.limits.max_results.min(10),
                budget_bytes: config.limits.max_context_bytes,
            },
        );
        let mut row = json!({"id":case.id,"question":case.question,"category":case.category,"split":case.split,"synthetic":case.synthetic,"expected_sources":case.expected_sources,"expected_facts":case.expected_facts,"error":null,"human_semantic":null});
        match output {
            Ok(result) => {
                if result.index_generation != generation {
                    return Err(Error::new(
                        "generation_changed",
                        "Answer evaluation requires one stable source generation",
                    ));
                }
                let identity = json!({"embedding":result.embedding_contract_id,"generator":result.generator_contract_id,"contracts":result.model_contracts,"fusion":result.fusion_method});
                if contracts.as_ref().is_some_and(|old: &Value| {
                    ["embedding", "generator", "fusion"]
                        .iter()
                        .any(|key| old[*key] != identity[*key])
                }) {
                    return Err(Error::new(
                        "model_contract_changed",
                        "Answer evaluation model contracts changed",
                    ));
                }
                row["execution"] = json!({"profile":result.profile,"retrieval_legs":result.retrieval_legs,"identity":identity});
                contracts = Some(identity);
                report.retrieval_legs = result.retrieval_legs;
                report.embedding_contract_id = result.embedding_contract_id;
                report.generator_contract_id = result.generator_contract_id;
                report.model_contracts = result.model_contracts;
                report.fusion_method = result.fusion_method;
                let data = result.data;
                let abstained = data["abstained"].as_bool().ok_or_else(|| {
                    Error::new("invalid_output", "Answer result lacks abstention status")
                })?;
                let claims = data["claims"]
                    .as_array()
                    .ok_or_else(|| Error::new("invalid_output", "Answer result lacks claims"))?;
                let cited: BTreeSet<String> = claims
                    .iter()
                    .flat_map(|c| c["citations"].as_array().into_iter().flatten())
                    .filter_map(|c| c["provenance"]["document_key"].as_str().map(str::to_owned))
                    .collect();
                let hash = ingest::digest(&serde_json::to_vec(
                    &json!({"question":case.question,"filters":case.filters,"expected_sources":case.expected_sources,"expected_facts":case.expected_facts,"answer":reviewed_payload(&data)}),
                )?);
                let (review, reason) = semantic_review(case, &data, &hash, evaluated_at);
                let v = &data["verification"];
                row["mechanical"] = json!({"answerable_coverage":if case.category == "answerable" {Some(!abstained && !claims.is_empty())} else {None},
                    "false_abstention":if case.category == "answerable" {Some(abstained)} else {None},
                    "correct_abstention":if case.category == "unanswerable" {Some(abstained)} else {None},
                    "citation_resolution":ratio(v["citations_resolved"].as_u64().unwrap_or(0),v["citations_submitted"].as_u64().unwrap_or(0)),
                    "quote_pass":ratio(v["quotes_verified"].as_u64().unwrap_or(0),v["quotes_submitted"].as_u64().unwrap_or(0)),
                    "expected_source_coverage":ratio(expected.intersection(&cited).count() as u64,expected.len() as u64),
                    "conflict_surfaced":if case.category == "conflicting" {Some(data["conflicts"].as_array().is_some_and(|a| !a.is_empty()) || data["corpus_conflicts"].as_array().is_some_and(|a| !a.is_empty()))} else {None}});
                row["verification"] = v.clone();
                row["answer_sha256"] = json!(hash);
                row["human_semantic"] = review;
                row["human_review_unavailable_reason"] = json!(reason);
                row["answer"] = data;
            }
            Err(error) => {
                row["error"] = json!({"code":error.code,"message":error.message});
                row["mechanical"] = json!({"answerable_coverage":if case.category == "answerable" {Some(false)} else {None},
                    "false_abstention":Value::Null,
                    "correct_abstention":if case.category == "unanswerable" {Some(false)} else {None}});
            }
        }
        rows.push(row);
    }
    let release = release(release_manifest, &cases, &rows, &hashes, evaluated_at)?;
    let by_category: BTreeMap<&str, Value> = ["answerable", "unanswerable", "conflicting"]
        .into_iter()
        .map(|category| {
            (
                category,
                aggregate(rows.iter().filter(|r| r["category"] == category)),
            )
        })
        .collect();
    let by_split: BTreeMap<&str, Value> = ["development", "held_out"]
        .into_iter()
        .map(|split| {
            (
                split,
                aggregate(rows.iter().filter(|r| r["split"] == split)),
            )
        })
        .collect();
    report.data = json!({"report_version":1,"evaluated_at_unix_ms":evaluated_at,"hashes":hashes,"aggregate":aggregate(rows.iter()),"by_category":by_category,"by_split":by_split,"rows":rows,
        "release_acceptance":release["release_acceptance"],"blocked_reasons":release["blocked_reasons"],"release":release,
        "metric_contract":{"mechanical":"Coverage is a non-abstained verified claim, not factual correctness; abstention uses fixture category; source overlap is not entailment.",
            "human_semantic":"Null unless supplied human identity/time and exact stable answer hash match; no model self-grading.","answer_hash":"SHA256 of canonical stable payload: question, filters, expected source/fact labels, answer abstention, claim text, evidence ID/quote/source+chunk identity/hash, conflicts; excludes clocks.",
            "zero_denominators":"Undefined rates are null, never a vacuous perfect score."},
        "limitations":["Synthetic/unreviewed fixtures cannot establish V1 answer-release acceptance.","Runtime provenance/quote verification cannot establish semantic support, relevance, or authority.","Human reviewer identities and pre-freeze timestamps are declarations, not cryptographically attested."]});
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_in_answers_include_each_heldout_category_without_release_claim() {
        let raw = include_bytes!("../tests/fixtures/answers.jsonl");
        let cases = cases(raw).unwrap();
        for category in ["answerable", "unanswerable", "conflicting"] {
            assert!(
                cases
                    .iter()
                    .any(|c| c.category == category && c.split == "held_out")
            );
        }
        assert!(
            cases
                .iter()
                .all(|c| c.synthetic && c.human_review.is_none())
        );
    }
    #[test]
    fn absent_manifest_and_unreviewed_semantics_cannot_pass() {
        let cases = cases(include_bytes!("../tests/fixtures/answers.jsonl")).unwrap();
        assert_eq!(
            release(None, &cases, &[], &Value::Null, 10).unwrap()["release_acceptance"],
            false
        );
        assert_eq!(
            aggregate([].iter())["human_semantic"]["supported_claim_rate"],
            Value::Null
        );
        assert_eq!(ratio(0, 0), None);
    }
    #[test]
    fn review_must_bind_exact_answer_and_claim_count() {
        let mut cases = cases(include_bytes!("../tests/fixtures/answers.jsonl")).unwrap();
        let c = &mut cases[0];
        c.human_review = Some(HumanReview {
            reviewer: "human".into(),
            reviewed_at_unix_ms: 5,
            answer_sha256: "wrong".into(),
            supported_claims: 1,
            total_claims: 1,
            answer_relevant: true,
            conflict_handling_correct: true,
            abstention_correct: true,
        });
        assert!(
            semantic_review(c, &json!({"claims":[{}]}), "actual", 10)
                .0
                .is_null()
        );
    }
    fn manifest_value(cases: &[Case]) -> Value {
        let labels: BTreeMap<&str, Value> = cases
            .iter()
            .filter(|c| c.split == "held_out")
            .map(|c| {
                (
                    c.id.as_str(),
                    json!({"reviewer":"fixture-human","reviewed_at_unix_ms":1}),
                )
            })
            .collect();
        json!({"version":1,"fixture_sha256":"fixture","source_sha256":"source","qrels_sha256":"qrels",
            "frozen_at_unix_ms":2,"owner":"test owner","rationale":"Owner-selected test thresholds, not production defaults",
            "human_reviewed_labels":labels,"thresholds":{"min_supported_claim_rate":0.8,"min_answerable_coverage":0.5,
                "max_false_abstention":0.5,"min_correct_abstention":0.8,"min_citation_resolution":0.8,
                "min_quote_pass":0.8,"min_answer_relevance":0.8,"min_conflict_handling":0.8}})
    }
    fn release_fixture(cases: &[Case], manifest: &Value, rows: &[Value]) -> Value {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("release.json");
        std::fs::write(&path, serde_json::to_vec(manifest).unwrap()).unwrap();
        release(
            Some(&path),
            cases,
            rows,
            &json!({"fixture_sha256":"fixture","source_sha256":"source","qrels_sha256":"qrels"}),
            10,
        )
        .unwrap()
    }
    #[test]
    fn manifest_rejects_vacuous_thresholds_postdated_freeze_and_missing_humans() {
        let mut cases = cases(include_bytes!("../tests/fixtures/answers.jsonl")).unwrap();
        for case in &mut cases {
            case.synthetic = false;
        }
        let rows: Vec<Value> = cases.iter().map(|c| json!({"id":c.id,"split":c.split,"error":null,
            "human_semantic":null,"mechanical":{"answerable_coverage":true,"false_abstention":false,"correct_abstention":true}})).collect();
        let mut manifest = manifest_value(&cases);
        manifest["thresholds"]["min_answerable_coverage"] = json!(0);
        manifest["thresholds"]["max_false_abstention"] = json!(1);
        manifest["frozen_at_unix_ms"] = json!(11);
        let result = release_fixture(&cases, &manifest, &rows);
        assert_eq!(result["release_acceptance"], false);
        let reasons = result["blocked_reasons"].as_array().unwrap();
        assert!(
            reasons
                .iter()
                .any(|r| r.as_str().unwrap().contains("non-vacuous"))
        );
        assert!(
            reasons
                .iter()
                .any(|r| r.as_str().unwrap().contains("strictly before"))
        );
        assert!(
            reasons
                .iter()
                .any(|r| r.as_str().unwrap().contains("actual human judgments"))
        );
    }
    #[test]
    fn universal_abstention_cannot_satisfy_release_even_with_claimed_human_reviews() {
        let mut cases = cases(include_bytes!("../tests/fixtures/answers.jsonl")).unwrap();
        for c in &mut cases {
            c.synthetic = false;
        }
        let rows: Vec<Value> = cases.iter().map(|c| json!({"id":c.id,"split":c.split,"error":null,
            "human_semantic":{"supported_claims":0,"total_claims":0,"answer_relevance":1.0,"conflict_handling":1.0},
            "mechanical":{"answerable_coverage":if c.category == "answerable" {Some(false)} else {None},
                "false_abstention":if c.category == "answerable" {Some(true)} else {None},
                "correct_abstention":if c.category == "unanswerable" {Some(true)} else {None}},
            "verification":{"citations_submitted":0,"citations_resolved":0,"quotes_submitted":0,"quotes_verified":0}})).collect();
        let result = release_fixture(&cases, &manifest_value(&cases), &rows);
        assert_eq!(result["release_acceptance"], false);
        let reasons = result["blocked_reasons"].as_array().unwrap();
        for metric in [
            "answerable_coverage",
            "false_abstention",
            "supported_claim_rate",
            "citation_resolution",
        ] {
            assert!(reasons.iter().any(|r| r.as_str().unwrap().contains(metric)));
        }
    }
}
