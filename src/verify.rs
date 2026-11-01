//! Mechanical grounding only: exact quoted spans and captured provenance, not entailment.
use crate::error::{Error, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{SystemTime, UNIX_EPOCH},
};

pub const ANSWER_SCHEMA_VERSION: u32 = 1;
pub const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_CLAIMS: usize = 64;
const MAX_CITATIONS: usize = 16;
const MAX_CONFLICTS: usize = 32;
const MAX_TEXT_BYTES: usize = 8192;
const MAX_QUOTE_BYTES: usize = 16384;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Generated {
    abstained: bool,
    #[serde(deserialize_with = "required_reason")]
    abstain_reason: Option<AbstainReason>,
    claims: Vec<Claim>,
    conflicts: Vec<Conflict>,
}
fn required_reason<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<AbstainReason>, D::Error> {
    Option::<AbstainReason>::deserialize(deserializer)
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AbstainReason {
    NoEvidence,
    InsufficientSupport,
    ConflictingEvidence,
    StaleIndex,
    ModelUnavailable,
    InvalidOutput,
}
impl AbstainReason {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoEvidence => "no_evidence",
            Self::InsufficientSupport => "insufficient_support",
            Self::ConflictingEvidence => "conflicting_evidence",
            Self::StaleIndex => "stale_index",
            Self::ModelUnavailable => "model_unavailable",
            Self::InvalidOutput => "invalid_output",
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    text: String,
    citations: Vec<Citation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Citation {
    evidence_id: String,
    quote: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Conflict {
    summary: String,
    evidence_ids: Vec<String>,
}

/// Schema identity versions the generated object without letting it supply provenance.
pub fn generation_schema() -> Value {
    json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"urn:scribe:grounded-answer:1", "title":"Scribe grounded answer v1",
        "type":"object", "additionalProperties":false,
        "required":["abstained","abstain_reason","claims","conflicts"],
        "properties":{
            "abstained":{"type":"boolean"},
            "abstain_reason":{"enum":[null,"no_evidence","insufficient_support","conflicting_evidence","stale_index","model_unavailable","invalid_output"]},
            "claims":{"type":"array","maxItems":MAX_CLAIMS,"items":{
                "type":"object","additionalProperties":false,"required":["text","citations"],
                "properties":{"text":{"type":"string","minLength":1,"maxLength":MAX_TEXT_BYTES},
                    "citations":{"type":"array","minItems":1,"maxItems":MAX_CITATIONS,"items":{
                        "type":"object","additionalProperties":false,"required":["evidence_id","quote"],
                        "properties":{"evidence_id":{"type":"string","minLength":1,"maxLength":64},
                            "quote":{"type":"string","minLength":1,"maxLength":MAX_QUOTE_BYTES}}
                    }}}
            }},
            "conflicts":{"type":"array","maxItems":MAX_CONFLICTS,"items":{
                "type":"object","additionalProperties":false,"required":["summary","evidence_ids"],
                "properties":{"summary":{"type":"string","minLength":1,"maxLength":MAX_TEXT_BYTES},
                    "evidence_ids":{"type":"array","minItems":2,"maxItems":MAX_CITATIONS,
                        "items":{"type":"string","minLength":1,"maxLength":64}}}
            }}
        }
    })
}

/// CR/LF and every Unicode whitespace run become one ASCII space; ends are trimmed.
/// No case folding, punctuation changes, Unicode normalization, or fuzzy matching.
pub fn normalize_quote(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(word);
    }
    normalized
}

fn invalid(message: &str) -> Error {
    Error::new("invalid_output", message)
}
fn bounded(text: &str, max: usize) -> bool {
    !text.trim().is_empty() && text.len() <= max
}

pub fn resolution_time_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::new("clock_error", "system clock predates Unix epoch"))?
        .as_millis();
    u64::try_from(millis).map_err(|_| Error::new("clock_error", "resolution timestamp overflow"))
}

/// Verifies bounded untrusted JSON against the exact selected pack. Caller must re-open
/// and hash its sources after generation and immediately before exposing this result.
pub fn verify(raw: &str, pack: &Value, generation: i64) -> Result<Value> {
    if raw.len() > MAX_OUTPUT_BYTES {
        return Err(invalid("generated JSON exceeds hard byte limit"));
    }
    let generated: Generated = serde_json::from_str(raw)
        .map_err(|_| invalid("generator did not return the strict answer JSON schema"))?;
    if generated.claims.len() > MAX_CLAIMS
        || generated.conflicts.len() > MAX_CONFLICTS
        || (generated.abstained != generated.abstain_reason.is_some())
        || (generated.abstained && !generated.claims.is_empty())
    {
        return Err(invalid(
            "answer counts or abstention state violate the schema",
        ));
    }
    for claim in &generated.claims {
        if !bounded(&claim.text, MAX_TEXT_BYTES)
            || claim.citations.len() > MAX_CITATIONS
            || claim
                .citations
                .iter()
                .any(|c| c.evidence_id.len() > 64 || c.quote.len() > MAX_QUOTE_BYTES)
        {
            return Err(invalid("generated claim exceeds schema bounds"));
        }
    }
    for conflict in &generated.conflicts {
        if !bounded(&conflict.summary, MAX_TEXT_BYTES)
            || conflict.evidence_ids.len() > MAX_CITATIONS
            || conflict.evidence_ids.iter().any(|id| id.len() > 64)
        {
            return Err(invalid("generated conflict exceeds schema bounds"));
        }
    }
    let items = pack["items"]
        .as_array()
        .ok_or_else(|| Error::new("invalid_evidence", "context pack has no items array"))?;
    let mut evidence = BTreeMap::new();
    for item in items {
        let id = item["evidence_id"]
            .as_str()
            .ok_or_else(|| Error::new("invalid_evidence", "missing evidence identity"))?;
        let body = item["body"]
            .as_str()
            .ok_or_else(|| Error::new("invalid_evidence", "missing evidence body"))?;
        if evidence.insert(id, (item, normalize_quote(body))).is_some() {
            return Err(Error::new(
                "invalid_evidence",
                "duplicate evidence identity",
            ));
        }
    }
    let mut claims = Vec::new();
    let mut submitted = 0usize;
    let mut resolved = 0usize;
    let mut verified = 0usize;
    let mut invalid_citations = 0usize;
    for claim in &generated.claims {
        let mut citations = Vec::new();
        let mut valid_claim = !claim.citations.is_empty();
        for citation in &claim.citations {
            submitted += 1;
            let Some((item, body)) = evidence.get(citation.evidence_id.as_str()) else {
                invalid_citations += 1;
                valid_claim = false;
                continue;
            };
            resolved += 1;
            let quote = normalize_quote(&citation.quote);
            let Some(offset) = (!quote.is_empty()).then(|| body.find(&quote)).flatten() else {
                invalid_citations += 1;
                valid_claim = false;
                continue;
            };
            verified += 1;
            let mut provenance = serde_json::Map::new();
            for field in [
                "root_key",
                "document_key",
                "chunk_key",
                "path",
                "heading",
                "line_start",
                "line_end",
                "source_sha256",
                "status",
                "role",
                "authority",
                "updated",
                "supersedes",
                "conflicts",
            ] {
                if let Some(value) = item.get(field) {
                    provenance.insert(field.into(), value.clone());
                }
            }
            provenance.insert("index_generation".into(), json!(generation));
            citations.push(json!({"evidence_id":citation.evidence_id,"quote":citation.quote,"normalized_quote":quote,
                "normalized_span_bytes":{"start":offset,"end":offset + quote.len()},"provenance":provenance}));
        }
        if valid_claim {
            claims.push(json!({"text":claim.text,"citations":citations}));
        }
    }
    let verified_claims = claims.len();
    let mut conflicts = Vec::new();
    for conflict in &generated.conflicts {
        let unique: BTreeSet<&str> = conflict.evidence_ids.iter().map(String::as_str).collect();
        if unique.len() < 2
            || unique.len() != conflict.evidence_ids.len()
            || unique.iter().any(|id| !evidence.contains_key(id))
        {
            continue;
        }
        conflicts.push(
            json!({"summary":conflict.summary,"evidence_ids":conflict.evidence_ids,
            "verification":"evidence_ids_only; disagreement is not semantically verified"}),
        );
    }
    let invalid_conflicts = generated.conflicts.len() - conflicts.len();
    let corpus_conflicts = pack.get("conflicts").cloned().unwrap_or_else(|| json!([]));
    let mut cited_projects = BTreeSet::new();
    let mut cited_source_conflict = false;
    for citation in claims
        .iter()
        .flat_map(|claim| claim["citations"].as_array().into_iter().flatten())
    {
        if let Some((item, _)) = citation["evidence_id"]
            .as_str()
            .and_then(|id| evidence.get(id))
        {
            cited_source_conflict |= item["conflicts"]
                .as_array()
                .is_some_and(|relations| !relations.is_empty());
            if let Some(project) = item["project"].as_str().filter(|id| !id.is_empty()) {
                cited_projects.insert(project);
            }
        }
    }
    // Relations can be one-sided. A selected source declaring a conflict with a
    // cited project still blocks synthesis, even when the cited source omits it.
    let incoming_conflict = items.iter().any(|item| {
        item["conflicts"].as_array().is_some_and(|relations| {
            relations.iter().any(|target| {
                target
                    .as_str()
                    .is_some_and(|id| cited_projects.contains(id))
            })
        })
    });
    let reason = if cited_source_conflict || incoming_conflict || !conflicts.is_empty() {
        Some("conflicting_evidence")
    } else if invalid_conflicts > 0 {
        Some("invalid_output")
    } else if generated.abstained {
        generated.abstain_reason.as_ref().map(AbstainReason::as_str)
    } else if claims.is_empty() {
        Some(if items.is_empty() {
            "no_evidence"
        } else if invalid_citations > 0 {
            "invalid_output"
        } else {
            "insufficient_support"
        })
    } else {
        None
    };
    let suppressed = if reason.is_some() {
        let n = claims.len();
        claims.clear();
        n
    } else {
        0
    };
    Ok(json!({
        "answer_schema_version":ANSWER_SCHEMA_VERSION,"abstained":reason.is_some(),"abstain_reason":reason,
        "conflict_policy":"Explicit conflicts involving cited sources, including incoming selected-source relations, or resolved model conflicts suppress synthesis; unrelated selected-source conflicts remain visible. Supersedes does not prove authority.",
        "claims":claims,"conflicts":conflicts,"corpus_conflicts":corpus_conflicts,
        "verification":{
            "claims_submitted":generated.claims.len(),"claims_verified":verified_claims,
            "claims_dropped":generated.claims.len()-verified_claims,"claims_suppressed_abstention":suppressed,
            "citations_submitted":submitted,"citations_resolved":resolved,"citations_verified":verified,
            "citations_invalid":invalid_citations,"quotes_submitted":submitted,"quotes_verified":verified,
            "conflicts_submitted":generated.conflicts.len(),"conflicts_verified":conflicts.len(),"conflicts_dropped":invalid_conflicts,
            "normalization":"Unicode whitespace runs -> ASCII space; trim; exact substring; no fuzzy matching",
            "semantic_entailment_verified":false,"relevance_verified":false,"source_authority_verified":false,
            "limitation":"Mechanical checks prove evidence identity, normalized exact quote resolution and captured source provenance only; not entailment, completeness, relevance, authority or currentness."
        },
        "pack_sha256":pack["pack_sha256"],"index_generation":generation,
        "source_consistency":"captured generation/hash; sources rehashed before output, not an atomic filesystem snapshot"
    }))
}
