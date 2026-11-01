use crate::{
    config::Config,
    db,
    error::{Error, Result},
    ingest,
    model::{Envelope, GetRequest, Hit, Leg, SearchRequest},
};
#[cfg(feature = "full")]
use rusqlite::OptionalExtension;
use rusqlite::{Connection, types::Value as SqlValue};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Read};

pub(crate) fn read_source(config: &Config, root: &str, path: &str, hash: &str) -> Result<String> {
    crate::mcp::checkpoint()?;
    let canonical = config.resolve_source(root, path)?;
    let mut bytes = Vec::new();
    std::fs::File::open(canonical)?
        .take(config.limits.max_file_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > config.limits.max_file_bytes || ingest::digest(&bytes) != hash {
        return Err(Error::new(
            "stale_source",
            format!("{root}:{path} differs from the captured index generation; run kb index"),
        ));
    }
    ingest::normalized_source(&bytes)
}
pub(crate) fn source_excerpt(text: &str, start: usize, end: usize) -> Result<String> {
    if start == 0 || end < start {
        return Err(Error::new(
            "invalid_arguments",
            "line range is inclusive, one-based and ordered",
        ));
    }
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if end > lines.len() {
        return Err(Error::new("invalid_arguments", "line range exceeds source"));
    }
    Ok(lines[start - 1..end].concat())
}
fn validate_date(date: &str) -> bool {
    let b = date.as_bytes();
    if b.len() != 10
        || b[4] != b'-'
        || b[7] != b'-'
        || !b
            .iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
    {
        return false;
    }
    let year: u32 = date[..4].parse().unwrap_or(0);
    let month: usize = date[5..7].parse().unwrap_or(0);
    let day: u32 = date[8..].parse().unwrap_or(0);
    let days = [
        31,
        if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
            29
        } else {
            28
        },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    year > 0 && (1..=12).contains(&month) && day > 0 && day <= days[month - 1]
}
pub fn search(config: &Config, request: &SearchRequest) -> Result<Envelope<Vec<Hit>>> {
    let conn = db::open_reader(config)?;
    search_on(config, &conn, request)
}
pub(crate) fn search_on(
    config: &Config,
    conn: &Connection,
    request: &SearchRequest,
) -> Result<Envelope<Vec<Hit>>> {
    validate_request(config, request)?;
    if config.profile.is_full() {
        #[cfg(feature = "full")]
        {
            return hybrid_on(config, conn, request);
        }
        #[cfg(not(feature = "full"))]
        return Err(Error::new(
            "profile_unavailable",
            "rebuild with --features full",
        ));
    }
    let (exact_id, browse) = query_mode(conn, request)?;
    let hits = lexical_on(conn, request, exact_id, browse, request.limit, false)?;
    validate_hits(config, &hits)?;
    let result = config.envelope(db::generation(conn)?, hits);
    bounded_result(config, result)
}

fn validate_request(config: &Config, request: &SearchRequest) -> Result<()> {
    crate::mcp::checkpoint()?;
    if request.query.len() > config.limits.max_query_bytes
        || request.limit == 0
        || request.limit > config.limits.max_results
    {
        return Err(Error::new(
            "invalid_arguments",
            "query/result limit exceeds configured bounds",
        ));
    }
    let f = &request.filters;
    for value in [&f.root, &f.role, &f.status, &f.project, &f.doc_type, &f.tag] {
        if value
            .as_ref()
            .is_some_and(|v| v.len() > 512 || v.is_empty())
        {
            return Err(Error::new(
                "invalid_arguments",
                "filter must be nonempty and at most 512 bytes",
            ));
        }
    }
    if f.root
        .as_ref()
        .is_some_and(|v| !config.roots.contains_key(v))
    {
        return Err(Error::new("invalid_arguments", "unknown root filter"));
    }
    for date in [&f.after, &f.before].into_iter().flatten() {
        if !validate_date(date) {
            return Err(Error::new(
                "invalid_arguments",
                "date filters require a valid YYYY-MM-DD date",
            ));
        }
    }
    if matches!((&f.after,&f.before),(Some(a),Some(b)) if a>b) {
        return Err(Error::new("invalid_arguments", "after exceeds before"));
    }
    if request.query.trim().is_empty() && f.project.is_none() {
        return Err(Error::new(
            "invalid_arguments",
            "query must be nonempty unless an exact project filter is supplied",
        ));
    }
    Ok(())
}

fn query_mode(conn: &Connection, request: &SearchRequest) -> Result<(bool, bool)> {
    let query = request.query.trim();
    let exact_id = !query.is_empty()
        && conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM document_manifest WHERE project_id=?)",
            [query],
            |r| r.get(0),
        )?;
    Ok((exact_id, query.is_empty()))
}

// Both candidate legs use exactly these metadata predicates before ranking.
fn filter_sql(request: &SearchRequest) -> (Vec<&'static str>, Vec<SqlValue>) {
    let f = &request.filters;
    let mut conditions = Vec::new();
    let mut args = Vec::new();
    for (condition, value) in [
        ("m.root_key=?", &f.root),
        ("m.source_role=?", &f.role),
        ("m.source_status=?", &f.status),
        ("m.project_id=?", &f.project),
        ("m.doc_type=?", &f.doc_type),
        (
            "EXISTS(SELECT 1 FROM json_each(m.tags_json) WHERE value=?)",
            &f.tag,
        ),
        ("m.updated_at>=?", &f.after),
        ("m.updated_at<=?", &f.before),
    ] {
        if let Some(value) = value {
            conditions.push(condition);
            args.push(value.clone().into());
        }
    }
    (conditions, args)
}

const HIT_COLUMNS: &str = "m.root_key,m.document_key,p.chunk_key,m.path,p.heading_path,p.line_start,p.line_end,m.sha256,m.source_status,m.source_role,m.source_authority,m.project_id,m.doc_type,m.tags_json,m.updated_at,m.metadata_json";
const HIT_JOIN: &str = "chunks JOIN chunk_provenance p ON p.chunk_rowid=chunks.rowid JOIN document_manifest m ON m.document_key=p.document_key";

fn read_hit(row: &rusqlite::Row<'_>) -> rusqlite::Result<(Hit, String, String)> {
    Ok((
        Hit {
            root_key: row.get(0)?,
            document_key: row.get(1)?,
            chunk_key: row.get(2)?,
            path: row.get(3)?,
            heading: row.get(4)?,
            line_start: db::integer_column(row, 5)?,
            line_end: db::integer_column(row, 6)?,
            source_sha256: row.get(7)?,
            status: row.get(8)?,
            role: row.get(9)?,
            authority: row.get(10)?,
            project: row.get(11)?,
            doc_type: row.get(12)?,
            tags: Vec::new(),
            updated: row.get(14)?,
            supersedes: Vec::new(),
            conflicts: Vec::new(),
            snippet: row.get(16)?,
            body: row.get(17)?,
            legs: Vec::new(),
            fused_rank: None,
            fused_score: None,
        },
        row.get(13)?,
        row.get(15)?,
    ))
}

fn hydrate_hit(hit: &mut Hit, tags: &str, metadata: &str) -> Result<()> {
    hit.tags = serde_json::from_str(tags)?;
    let meta: crate::model::Metadata = serde_json::from_str(metadata)?;
    hit.supersedes = meta.supersedes;
    hit.conflicts = meta.conflicts;
    Ok(())
}

fn lexical_on(
    conn: &Connection,
    request: &SearchRequest,
    exact_id: bool,
    browse: bool,
    limit: usize,
    stable_keys: bool,
) -> Result<Vec<Hit>> {
    let (mut conditions, mut args) = filter_sql(request);
    if exact_id {
        conditions.push("m.project_id=?");
        args.push(request.query.trim().to_owned().into());
    } else if !browse {
        conditions.push("chunks MATCH ?");
        let escaped = request
            .query
            .split_whitespace()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" AND ");
        args.push(escaped.into());
    }
    if conditions.is_empty() {
        conditions.push("1");
    }
    args.push((limit as i64).into());
    let score = if exact_id || browse {
        "0.0"
    } else {
        "bm25(chunks)"
    };
    let snippet = if exact_id || browse {
        "substr(chunks.body,1,256)"
    } else {
        "snippet(chunks,2,'>>>','<<<','…',24)"
    };
    let order = if stable_keys {
        "rank,m.root_key,m.document_key,p.chunk_key"
    } else {
        "rank,m.root_key,m.path,p.line_start,p.chunk_key"
    };
    let sql = format!(
        "SELECT {HIT_COLUMNS},{snippet},chunks.body,{score} AS rank FROM {HIT_JOIN} WHERE {} ORDER BY {order} LIMIT ?",
        conditions.join(" AND ")
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(rusqlite::params_from_iter(args), |row| {
        let (hit, tags, metadata) = read_hit(row)?;
        Ok((hit, tags, metadata, row.get::<_, f64>(18)?))
    })?;
    let mut hits = Vec::new();
    for (index, row) in rows.enumerate() {
        crate::mcp::checkpoint()?;
        let (mut hit, tags, metadata, score) = row?;
        hydrate_hit(&mut hit, &tags, &metadata)?;
        hit.legs.push(Leg {
            name: "lexical".into(),
            rank: index + 1,
            raw_score: score,
            score_kind: if exact_id {
                "exact_id"
            } else if browse {
                "source_order"
            } else {
                "bm25_lower_is_better"
            }
            .into(),
        });
        hits.push(hit);
    }
    Ok(hits)
}

fn validate_hits(config: &Config, hits: &[Hit]) -> Result<()> {
    let mut sources = BTreeMap::new();
    for hit in hits {
        crate::mcp::checkpoint()?;
        let key = (&hit.root_key, &hit.path);
        if !sources.contains_key(&key) {
            sources.insert(
                key,
                read_source(config, &hit.root_key, &hit.path, &hit.source_sha256)?,
            );
        }
        if source_excerpt(&sources[&key], hit.line_start, hit.line_end)? != hit.body {
            return Err(Error::new(
                "index_corrupt",
                "chunk body does not match its recorded source lines",
            ));
        }
    }
    Ok(())
}

fn bounded_result<T: serde::Serialize>(
    config: &Config,
    result: Envelope<T>,
) -> Result<Envelope<T>> {
    if serde_json::to_vec(&result)?.len() > config.limits.max_response_bytes {
        return Err(Error::new("response_too_large", "reduce result count"));
    }
    Ok(result)
}

#[cfg(feature = "full")]
fn hybrid_on(
    config: &Config,
    conn: &Connection,
    request: &SearchRequest,
) -> Result<Envelope<Vec<Hit>>> {
    use crate::embed::{LocalModel, ModelPurpose};
    let settings = config.embedding()?;
    let retrieval = &config
        .full
        .as_ref()
        .ok_or_else(|| {
            Error::new(
                "embedding_unavailable",
                "full embedding configuration is required",
            )
        })?
        .retrieval;
    if !matches!(retrieval.fusion.as_str(), "rrf" | "weighted_score")
        || !retrieval.lexical_weight.is_finite()
        || !(0.0..=1.0).contains(&retrieval.lexical_weight)
        || retrieval.candidates == 0
        || retrieval.candidates > 10_000
        || retrieval.rrf_k == 0
        || retrieval.rrf_k > 1_000_000
    {
        return Err(Error::new(
            "invalid_retrieval_config",
            "fusion must be rrf or weighted_score; weight must be finite in [0,1], candidates in 1..=10000 and rrf_k in 1..=1000000",
        ));
    }
    let model = LocalModel::prepare(settings, ModelPurpose::Embedding)?;
    let generation = db::generation(conn)?;
    validate_contract(conn, &model.contract)?;
    let query = model.embed_query(request.query.trim())?;
    let limit = retrieval.candidates.max(request.limit);
    let (exact_id, browse) = query_mode(conn, request)?;
    let dense = dense_on(
        conn,
        request,
        exact_id,
        &model.contract.id,
        generation,
        &query,
        limit,
    )?;
    let lexical = lexical_on(conn, request, exact_id, browse, limit, true)?;
    let mut hits = fuse(lexical, dense, retrieval, exact_id);
    hits.truncate(request.limit);
    validate_hits(config, &hits)?;
    model.revalidate()?;
    let mut result = config.envelope(generation, hits);
    result.retrieval_legs = vec!["lexical".into(), "dense".into()];
    result.embedding_contract_id = Some(model.contract.id.clone());
    result.model_contracts = json!({
        "embedding": model.contract,
        "embedding_observations": model.observations(),
        "retrieval": {
            "fusion": retrieval,
            "exact_id_policy": "exact_id_first; exact IDs constrain both legs; all vectors still validated",
            "weighted_normalization": "per-leg minmax; lower BM25 inverted; constant leg=1; absent leg=0",
            "vector_validation": "all active corpus chunks, before metadata filtering"
        }
    });
    result.fusion_method = Some(retrieval.fusion.clone());
    bounded_result(config, result)
}

#[cfg(feature = "full")]
fn validate_contract(conn: &Connection, contract: &crate::embed::ModelContract) -> Result<()> {
    for table in ["embedding_contract", "chunk_embeddings"] {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
            [table],
            |r| r.get(0),
        )?;
        if !exists {
            return Err(Error::new(
                "embeddings_unavailable",
                "no full index vectors; run kb index --profile full",
            ));
        }
    }
    let schema: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='embedding_schema'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let active: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='embedding_contract_id'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if schema.as_deref() != Some("1") || active.as_deref() != Some(contract.id.as_str()) {
        return Err(Error::new(
            "embeddings_stale",
            "active vectors do not match the requested embedding contract; run full index",
        ));
    }
    let recorded: Option<String> = conn
        .query_row(
            "SELECT contract_json FROM embedding_contract WHERE contract_id=?",
            [&contract.id],
            |r| r.get(0),
        )
        .optional()?;
    let recorded = recorded.and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    let complete = serde_json::to_value(contract)?;
    if recorded.as_ref() != Some(&complete) {
        return Err(Error::new(
            "embeddings_stale",
            "persisted embedding contract is missing or differs from the actual model contract",
        ));
    }
    Ok(())
}

#[cfg(feature = "full")]
fn dense_on(
    conn: &Connection,
    request: &SearchRequest,
    exact_id: bool,
    contract_id: &str,
    generation: i64,
    query: &[f32],
    limit: usize,
) -> Result<Vec<Hit>> {
    // A restrictive filter must not hide a partly refreshed full generation.
    let broken: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM chunks c LEFT JOIN chunk_provenance p ON p.chunk_rowid=c.rowid LEFT JOIN document_manifest m ON m.document_key=p.document_key WHERE p.chunk_rowid IS NULL OR m.document_key IS NULL) OR EXISTS(SELECT 1 FROM chunk_provenance p LEFT JOIN chunks c ON c.rowid=p.chunk_rowid LEFT JOIN document_manifest m ON m.document_key=p.document_key WHERE c.rowid IS NULL OR m.document_key IS NULL) OR EXISTS(SELECT 1 FROM chunk_embeddings e WHERE e.contract_id=? AND NOT EXISTS(SELECT 1 FROM chunk_provenance p WHERE p.document_key=e.document_key AND p.chunk_key=e.chunk_key))",
        [contract_id], |r| r.get(0),
    )?;
    if broken {
        return Err(Error::new(
            "embeddings_stale",
            "vector/chunk identities do not match the captured corpus",
        ));
    }
    let (mut conditions, mut args) = filter_sql(request);
    if exact_id {
        conditions.push("m.project_id=?");
        args.push(request.query.trim().to_owned().into());
    }
    let eligible = if conditions.is_empty() {
        "1".to_owned()
    } else {
        conditions.join(" AND ")
    };
    args.push(contract_id.to_owned().into());
    let sql = format!(
        "SELECT chunks.rowid,m.sha256,m.generation,p.source_sha256,e.source_sha256,e.dimensions,e.generation,e.vector,({eligible}) FROM {HIT_JOIN} LEFT JOIN chunk_embeddings e ON e.document_key=p.document_key AND e.chunk_key=p.chunk_key AND e.contract_id=? ORDER BY m.root_key,m.document_key,p.chunk_key"
    );
    let mut statement = conn.prepare(&sql)?;
    let mut rows = statement.query(rusqlite::params_from_iter(args))?;
    let mut scores = Vec::new();
    while let Some(row) = rows.next()? {
        crate::mcp::checkpoint()?;
        let hash: String = row.get(1)?;
        let source_generation: i64 = row.get(2)?;
        let provenance_hash: String = row.get(3)?;
        let vector_hash: Option<String> = row.get(4)?;
        let dimensions: Option<i64> = row.get(5)?;
        let vector_generation: Option<i64> = row.get(6)?;
        if source_generation != generation
            || provenance_hash != hash
            || vector_hash.as_deref() != Some(hash.as_str())
            || dimensions != i64::try_from(query.len()).ok()
            || vector_generation != Some(generation)
        {
            return Err(Error::new(
                "embeddings_stale",
                "missing vector or mismatched source hash, dimension or generation; run full index",
            ));
        }
        let blob = row
            .get_ref(7)?
            .as_blob()
            .map_err(|_| Error::new("embeddings_stale", "vector is not a BLOB"))?;
        let vector = crate::embed::decode_vector(blob, query.len()).map_err(|error| {
            Error::new(
                "embeddings_stale",
                format!("invalid stored vector: {}", error.message),
            )
        })?;
        if row.get::<_, Option<bool>>(8)?.unwrap_or(false) {
            let score = query
                .iter()
                .zip(&vector)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            scores.push((score, row.get::<_, i64>(0)?));
        }
    }
    // Stable sorting retains the SQL root/document/chunk identity tie-break.
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    scores.truncate(limit);
    let mut statement = conn.prepare(&format!("SELECT {HIT_COLUMNS},substr(chunks.body,1,256),chunks.body FROM {HIT_JOIN} WHERE chunks.rowid=?"))?;
    let mut hits = Vec::with_capacity(scores.len());
    for (index, (score, rowid)) in scores.into_iter().enumerate() {
        crate::mcp::checkpoint()?;
        let (mut hit, tags, metadata) = statement.query_row([rowid], read_hit)?;
        hydrate_hit(&mut hit, &tags, &metadata)?;
        hit.legs.push(Leg {
            name: "dense".into(),
            rank: index + 1,
            raw_score: score,
            score_kind: "cosine_higher_is_better".into(),
        });
        hits.push(hit);
    }
    Ok(hits)
}

#[cfg(feature = "full")]
fn fuse(
    lexical: Vec<Hit>,
    dense: Vec<Hit>,
    retrieval: &crate::full_config::RetrievalConfig,
    exact_id: bool,
) -> Vec<Hit> {
    let bounds = |hits: &[Hit]| {
        hits.iter()
            .map(|h| h.legs[0].raw_score)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), score| {
                (min.min(score), max.max(score))
            })
    };
    let lexical_bounds = bounds(&lexical);
    let dense_bounds = bounds(&dense);
    let mut combined: BTreeMap<(String, String), Hit> = BTreeMap::new();
    for hit in lexical.into_iter().chain(dense) {
        let key = (hit.document_key.clone(), hit.chunk_key.clone());
        if let Some(existing) = combined.get_mut(&key) {
            existing.legs.extend(hit.legs);
        } else {
            combined.insert(key, hit);
        }
    }
    let mut hits: Vec<Hit> = combined.into_values().collect();
    for hit in &mut hits {
        let score = hit
            .legs
            .iter()
            .map(|leg| {
                if retrieval.fusion == "rrf" {
                    return 1.0 / (retrieval.rrf_k as f64 + leg.rank as f64);
                }
                let lexical = leg.name == "lexical";
                let (min, max) = if lexical {
                    lexical_bounds
                } else {
                    dense_bounds
                };
                let normalized = if min == max {
                    1.0
                } else if lexical {
                    (max - leg.raw_score) / (max - min)
                } else {
                    (leg.raw_score - min) / (max - min)
                };
                normalized
                    * if lexical {
                        retrieval.lexical_weight
                    } else {
                        1.0 - retrieval.lexical_weight
                    }
            })
            .sum();
        hit.fused_score = Some(score);
    }
    let pinned = |hit: &Hit| exact_id && hit.legs.iter().any(|leg| leg.score_kind == "exact_id");
    hits.sort_by(|a, b| {
        pinned(b)
            .cmp(&pinned(a))
            .then_with(|| b.fused_score.unwrap().total_cmp(&a.fused_score.unwrap()))
            .then_with(|| {
                (&a.root_key, &a.document_key, &a.chunk_key).cmp(&(
                    &b.root_key,
                    &b.document_key,
                    &b.chunk_key,
                ))
            })
    });
    for (index, hit) in hits.iter_mut().enumerate() {
        hit.fused_rank = Some(index + 1);
    }
    hits
}
pub fn get(config: &Config, request: &GetRequest) -> Result<Envelope<Value>> {
    let path = request
        .key
        .split_once(':')
        .map_or(request.key.as_str(), |(_, path)| path);
    if request.key.is_empty()
        || request.key.len() > 512
        || request.key.contains('\\')
        || path.starts_with('/')
        || path.split('/').any(|p| p == ".." || p == ".")
    {
        return Err(Error::new(
            "inaccessible_path",
            "invalid document/chunk key or relative source path",
        ));
    }
    let conn = db::open_reader(config)?;
    let mut statement = conn.prepare("SELECT DISTINCT m.root_key,m.path,m.document_key,m.sha256,m.metadata_json,p.chunk_key,p.heading_path,p.line_start,p.line_end FROM document_manifest m LEFT JOIN chunk_provenance p ON p.document_key=m.document_key AND p.chunk_key=?1 WHERE m.document_key=?1 OR m.project_id=?1 OR (m.root_key||':'||m.path)=?1 OR m.path=?1 OR p.chunk_key=?1 ORDER BY m.root_key,m.path")?;
    let rows = statement
        .query_map([&request.key], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, Option<String>>(6)?,
                db::optional_integer_column::<usize>(r, 7)?,
                db::optional_integer_column::<usize>(r, 8)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Err(Error::new(
            "not_found",
            "no indexed document or chunk matches key",
        ));
    }
    if rows.len() > 1 {
        return Err(Error::new(
            "ambiguous_identity",
            "key matches multiple documents; use document_key or root:path",
        ));
    }
    let (root, path, key, hash, meta, chunk, heading, start, end) = &rows[0];
    let source = read_source(config, root, path, hash)?;
    let total = source.split_inclusive('\n').count();
    let start = request.line_start.or(*start).unwrap_or(1);
    let end = request.line_end.or(*end).unwrap_or(total);
    if chunk.is_some() && (start < rows[0].7.unwrap() || end > rows[0].8.unwrap()) {
        return Err(Error::new(
            "invalid_arguments",
            "requested lines exceed chunk span",
        ));
    }
    let body = if total == 0 && request.line_start.is_none() && request.line_end.is_none() {
        String::new()
    } else {
        source_excerpt(&source, start, end)?
    };
    let mut result = config.envelope(
        db::generation(&conn)?,
        json!({"root_key":root,"path":path,"document_key":key,"chunk_key":chunk,"heading":heading,"line_start":start,"line_end":end,"source_sha256":hash,"metadata":serde_json::from_str::<Value>(meta)?,"body":body}),
    );
    result.retrieval_legs.clear();
    if serde_json::to_vec(&result)?.len() > config.limits.max_response_bytes {
        return Err(Error::new(
            "response_too_large",
            "request a smaller source line range",
        ));
    }
    crate::mcp::checkpoint()?;
    Ok(result)
}
