use crate::{
    config::Config,
    db,
    embed::{LocalModel, ModelPurpose, decode_vector, encode_vector},
    error::{Error, Result},
    model::Document,
};
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS embedding_contract(contract_id TEXT PRIMARY KEY,contract_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS chunk_embeddings(document_key TEXT NOT NULL,chunk_key TEXT NOT NULL,contract_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,dimensions INTEGER NOT NULL,generation INTEGER NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(document_key,chunk_key,contract_id)); CREATE INDEX IF NOT EXISTS chunk_embeddings_contract_idx ON chunk_embeddings(contract_id,document_key);";
struct Pending {
    document_key: String,
    chunk_key: String,
    source_sha256: String,
    vector: Vec<f32>,
}
pub(crate) struct Candidate {
    pub model: LocalModel,
    pub needs_publication: bool,
    pending: Vec<Pending>,
    dimensions: usize,
    started: Instant,
    timeout: Duration,
}
impl Candidate {
    pub fn check_deadline(&self) -> Result<()> {
        crate::mcp::checkpoint()?;
        if self.started.elapsed() >= self.timeout {
            return Err(Error::new(
                "index_timeout",
                "full index exceeded configured index_timeout_ms; prior generation retained",
            ));
        }
        Ok(())
    }
    pub fn publish(&self, conn: &Connection, generation: i64) -> Result<()> {
        self.check_deadline()?;
        conn.execute("INSERT INTO embedding_contract(contract_id,contract_json) VALUES(?,?) ON CONFLICT(contract_id) DO UPDATE SET contract_json=excluded.contract_json",params![self.model.contract.id,serde_json::to_string(&self.model.contract)?])?;
        conn.execute("DELETE FROM chunk_embeddings WHERE contract_id<>? OR NOT EXISTS(SELECT 1 FROM chunk_provenance p JOIN document_manifest m ON m.document_key=p.document_key WHERE p.document_key=chunk_embeddings.document_key AND p.chunk_key=chunk_embeddings.chunk_key AND m.sha256=chunk_embeddings.source_sha256)",[&self.model.contract.id])?;
        let mut statement=conn.prepare("INSERT INTO chunk_embeddings(document_key,chunk_key,contract_id,source_sha256,dimensions,generation,vector) VALUES(?,?,?,?,?,?,?) ON CONFLICT(document_key,chunk_key,contract_id) DO UPDATE SET source_sha256=excluded.source_sha256,dimensions=excluded.dimensions,generation=excluded.generation,vector=excluded.vector")?;
        for pending in &self.pending {
            self.check_deadline()?;
            statement.execute(params![
                pending.document_key,
                pending.chunk_key,
                self.model.contract.id,
                pending.source_sha256,
                self.dimensions as i64,
                generation,
                encode_vector(&pending.vector)?
            ])?;
        }
        conn.execute(
            "UPDATE chunk_embeddings SET generation=? WHERE contract_id=?",
            params![generation, self.model.contract.id],
        )?;
        for (key, value) in [
            ("embedding_schema", "1"),
            ("embedding_contract_id", self.model.contract.id.as_str()),
        ] {
            conn.execute("INSERT INTO kb_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[key,value])?;
        }
        let missing:i64=conn.query_row("SELECT count(*) FROM chunk_provenance p JOIN document_manifest m ON m.document_key=p.document_key LEFT JOIN chunk_embeddings e ON e.document_key=p.document_key AND e.chunk_key=p.chunk_key AND e.contract_id=? WHERE e.chunk_key IS NULL OR e.source_sha256<>m.sha256 OR e.generation<>?",params![self.model.contract.id,generation],|r|r.get(0))?;
        if missing != 0 {
            return Err(Error::new(
                "embeddings_stale",
                "candidate vector set is incomplete; prior generation retained",
            ));
        }
        self.model.revalidate()?;
        self.check_deadline()?;
        Ok(())
    }
}
pub(crate) fn prepare(
    config: &Config,
    conn: &Connection,
    docs: &[Document],
    force: bool,
    rebuild: bool,
    old_generation: i64,
) -> Result<Candidate> {
    let full = config.full.as_ref().ok_or_else(|| {
        Error::new(
            "embedding_unavailable",
            "configure full.embedding before full indexing",
        )
    })?;
    if !(1000..=3_600_000).contains(&full.index_timeout_ms) {
        return Err(Error::new(
            "invalid_config",
            "index_timeout_ms must be 1000..3600000",
        ));
    }
    let started = Instant::now();
    let settings = config.embedding()?;
    let model = LocalModel::prepare(settings, ModelPurpose::Embedding)?;
    let table_exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_embeddings')",
        [],
        |r| r.get(0),
    )?;
    let schema: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='embedding_schema'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if table_exists && schema.as_deref() != Some("1") {
        return Err(Error::new(
            "schema_unsupported",
            "existing embedding tables have an unsupported schema; rebuild a separate index",
        ));
    }
    conn.execute_batch(SCHEMA)?;
    let active: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='embedding_contract_id'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let contract_changed = active.as_deref() != Some(model.contract.id.as_str());
    let mut known = BTreeMap::new();
    if !force && !rebuild && !contract_changed {
        let mut stmt=conn.prepare("SELECT document_key,chunk_key,source_sha256,dimensions,generation,vector FROM chunk_embeddings WHERE contract_id=?")?;
        let rows = stmt.query_map([&model.contract.id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                db::integer_column::<usize>(r, 3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        for row in rows {
            let (doc, chunk, hash, dim, generation, blob) = row?;
            if dim == settings.dimensions
                && generation == old_generation
                && decode_vector(&blob, dim).is_ok()
            {
                known.insert((doc, chunk), hash);
            }
        }
    }
    let mut work = Vec::new();
    for doc in docs {
        for chunk in &doc.chunks {
            if known.get(&(doc.key.clone(), chunk.key.clone())) != Some(&doc.sha256) {
                work.push((doc, chunk));
            }
        }
    }
    let total_chunks: usize = docs.iter().map(|d| d.chunks.len()).sum();
    let current_count: i64 = conn.query_row(
        "SELECT count(*) FROM chunk_embeddings WHERE contract_id=?",
        [&model.contract.id],
        |r| r.get(0),
    )?;
    let mut candidate = Candidate {
        model,
        needs_publication: force
            || rebuild
            || contract_changed
            || !work.is_empty()
            || current_count != total_chunks as i64,
        pending: Vec::with_capacity(work.len()),
        dimensions: settings.dimensions,
        started,
        timeout: Duration::from_millis(full.index_timeout_ms),
    };
    for batch in work.chunks(16) {
        candidate.check_deadline()?;
        let texts: Vec<&str> = batch.iter().map(|(_, chunk)| chunk.body.as_str()).collect();
        let vectors = candidate.model.embed_documents(&texts)?;
        if vectors.len() != batch.len() {
            return Err(Error::new(
                "embedding_invalid",
                "provider returned wrong embedding count",
            ));
        }
        for ((doc, chunk), vector) in batch.iter().zip(vectors) {
            candidate.pending.push(Pending {
                document_key: doc.key.clone(),
                chunk_key: chunk.key.clone(),
                source_sha256: doc.sha256.clone(),
                vector,
            });
        }
    }
    candidate.check_deadline()?;
    Ok(candidate)
}
