use crate::{
    config::Config,
    error::{Error, Result},
    model::{ContextRequest, Envelope, SearchRequest},
    retrieve,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub fn pack(config: &Config, request: &ContextRequest) -> Result<Envelope<Value>> {
    if request.budget_bytes < 512 || request.budget_bytes > config.limits.max_context_bytes {
        return Err(Error::new(
            "invalid_arguments",
            "budget_bytes must be between 512 and the configured max_context_bytes; it is bytes, not tokens",
        ));
    }
    let search = retrieve::search(
        config,
        &SearchRequest {
            query: request.query.clone(),
            filters: request.filters.clone(),
            limit: request.limit,
        },
    )?;
    let candidates = search.data;
    let mut items: Vec<Value> = Vec::new();
    let mut spans: Vec<(&str, usize, usize)> = Vec::new();
    let mut budget_excluded = 0;
    let mut overlap_excluded = 0;
    for hit in &candidates {
        crate::mcp::checkpoint()?;
        if spans.iter().any(|(key, start, end)| {
            *key == hit.document_key && hit.line_start <= *end && hit.line_end >= *start
        }) {
            overlap_excluded += 1;
            continue;
        }
        let mut item = serde_json::to_value(hit)?;
        item.as_object_mut()
            .expect("serialized hit")
            .remove("snippet");
        item["evidence_id"] = json!(format!("S{}", items.len() + 1));
        items.push(item);
        let trial = payload(
            &items,
            request.budget_bytes,
            candidates.len(),
            budget_excluded,
            overlap_excluded,
        )?;
        if serde_json::to_vec(&trial)?.len() > request.budget_bytes {
            items.pop();
            budget_excluded += 1;
        } else {
            spans.push((&hit.document_key, hit.line_start, hit.line_end));
        }
    }
    let data = loop {
        let value = payload(
            &items,
            request.budget_bytes,
            candidates.len(),
            budget_excluded,
            overlap_excluded,
        )?;
        if serde_json::to_vec(&value)?.len() <= request.budget_bytes {
            break value;
        }
        if items.pop().is_none() {
            return Err(Error::new(
                "invalid_arguments",
                "budget cannot contain the evidence-pack manifest",
            ));
        }
        budget_excluded += 1;
    };
    // Revalidate selected sources immediately before returning the pack. This binds
    // citations to the captured generation/hash, not a filesystem snapshot.
    let mut checked = BTreeSet::new();
    for item in &items {
        if !checked.insert(item["document_key"].as_str().unwrap()) {
            continue;
        }
        crate::mcp::checkpoint()?;
        retrieve::read_source(
            config,
            item["root_key"].as_str().unwrap(),
            item["path"].as_str().unwrap(),
            item["source_sha256"].as_str().unwrap(),
        )?;
    }
    let mut result = config.envelope(search.index_generation, data);
    result.profile = search.profile;
    result.retrieval_legs = search.retrieval_legs;
    result.embedding_contract_id = search.embedding_contract_id;
    result.generator_contract_id = search.generator_contract_id;
    result.model_contracts = search.model_contracts;
    result.fusion_method = search.fusion_method;
    Ok(result)
}
fn payload(
    items: &[Value],
    budget: usize,
    candidates: usize,
    excluded_budget: usize,
    excluded_overlap: usize,
) -> Result<Value> {
    let conflicts: Vec<Value> = items.iter().filter(|v| v["conflicts"].as_array().is_some_and(|a| !a.is_empty()) || v["supersedes"].as_array().is_some_and(|a| !a.is_empty())).map(|v| json!({"evidence_id":v["evidence_id"],"supersedes":v["supersedes"],"conflicts":v["conflicts"]})).collect();
    let hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(items, &conflicts))?)
    );
    let mut data = json!({"budget_unit":"bytes","budget_bytes":budget,"used_bytes":0,"pack_sha256":hash,"items":items,"conflicts":conflicts,"truncation":{"candidate_count":candidates,"excluded_budget":excluded_budget,"excluded_overlap":excluded_overlap},"source_consistency":"captured_generation_hash; not an atomic filesystem snapshot"});
    loop {
        let size = serde_json::to_vec(&data)?.len();
        if data["used_bytes"].as_u64() == Some(size as u64) {
            break;
        }
        data["used_bytes"] = json!(size);
    }
    Ok(data)
}
