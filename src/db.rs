use crate::{
    config::Config,
    error::{Error, Result},
    ingest,
    model::{Chunk, Document, Envelope},
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};

pub(crate) fn integer_column<T>(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<T>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    T::try_from(row.get::<_, i64>(index)?).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Integer,
            Box::new(e),
        )
    })
}
pub(crate) fn optional_integer_column<T>(
    row: &rusqlite::Row<'_>,
    index: usize,
) -> rusqlite::Result<Option<T>>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    row.get::<_, Option<i64>>(index)?
        .map(T::try_from)
        .transpose()
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                index,
                rusqlite::types::Type::Integer,
                Box::new(e),
            )
        })
}
fn sql_integer(value: impl TryInto<i64>) -> Result<i64> {
    value.try_into().map_err(|_| {
        Error::new(
            "invalid_index_integer",
            "value exceeds SQLite's signed integer range",
        )
    })
}

pub const SCHEMA_VERSION: i64 = 1;
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,mtime REAL NOT NULL,size INTEGER NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(path UNINDEXED,heading,body,project UNINDEXED,type UNINDEXED,status UNINDEXED,tags UNINDEXED,line_start UNINDEXED,tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS kb_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS generations(id INTEGER PRIMARY KEY,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS root_registry(root_key TEXT PRIMARY KEY,display_name TEXT NOT NULL,canonical_path TEXT NOT NULL,policy_version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_manifest(
 root_key TEXT NOT NULL,path TEXT NOT NULL,document_key TEXT NOT NULL UNIQUE,sha256 TEXT NOT NULL,
 parser_version TEXT NOT NULL,chunker_version TEXT NOT NULL,source_status TEXT NOT NULL,source_role TEXT NOT NULL,
 source_authority TEXT NOT NULL,relations_json TEXT NOT NULL DEFAULT '[]',project_id TEXT NOT NULL,
 doc_type TEXT NOT NULL,tags_json TEXT NOT NULL,updated_at TEXT,generation INTEGER NOT NULL,
 metadata_json TEXT NOT NULL,links_json TEXT NOT NULL,mtime REAL NOT NULL,size INTEGER NOT NULL,
 PRIMARY KEY(root_key,path));
CREATE INDEX IF NOT EXISTS document_filter_idx ON document_manifest(root_key,source_role,source_status,project_id,doc_type);
CREATE TABLE IF NOT EXISTS chunk_provenance(
 chunk_rowid INTEGER PRIMARY KEY,document_key TEXT NOT NULL,chunk_key TEXT NOT NULL,parent_key TEXT,
 heading_path TEXT NOT NULL,line_start INTEGER NOT NULL,line_end INTEGER NOT NULL,source_sha256 TEXT NOT NULL,
 UNIQUE(document_key,chunk_key));
"#;
fn version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?)
}
fn reject_newer(conn: &Connection) -> Result<i64> {
    let v = version(conn)?;
    if v != 0 && v != SCHEMA_VERSION {
        return Err(Error::new(
            "schema_unsupported",
            format!(
                "schema {v}, supported {SCHEMA_VERSION}; use a matching binary or rebuild a separate index"
            ),
        ));
    }
    Ok(v)
}
fn verify_legacy(conn: &Connection) -> Result<()> {
    for (table, expected) in [
        ("files", vec!["path", "sha256", "mtime", "size"]),
        (
            "chunks",
            vec![
                "path",
                "heading",
                "body",
                "project",
                "type",
                "status",
                "tags",
                "line_start",
            ],
        ),
    ] {
        let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = statement
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if !names.is_empty() && names != expected {
            return Err(Error::new(
                "schema_unsupported",
                format!(
                    "unexpected legacy {table} shape; source rebuild into a separate database required"
                ),
            ));
        }
    }
    Ok(())
}
pub fn open_reader(config: &Config) -> Result<Connection> {
    if !config.database.is_file() {
        return Err(Error::new(
            "index_unavailable",
            "index is missing; run kb index with the same configuration",
        ));
    }
    let conn = Connection::open_with_flags(
        &config.database,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    crate::mcp::checkpoint()?;
    crate::mcp::observe_connection(&conn);
    conn.busy_timeout(Duration::from_millis(2000))?;
    let v = reject_newer(&conn)?;
    if v == 0 {
        return Err(Error::new(
            "migration_required",
            "legacy index must be rebuilt by kb index; first use a private copy",
        ));
    }
    // Pin policy and data in one snapshot. Callers finish this transaction by dropping the connection.
    conn.execute_batch("PRAGMA query_only=ON; BEGIN DEFERRED;")?;
    let fingerprint: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='config_fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if fingerprint.as_deref() != Some(config.fingerprint().as_str()) {
        return Err(Error::new(
            "index_configuration_changed",
            "roots/parser/chunker/limits changed; run kb index before reading",
        ));
    }
    generation(&conn)?;
    Ok(conn)
}
pub fn generation(conn: &Connection) -> Result<i64> {
    let value: Option<String> = conn
        .query_row(
            "SELECT value FROM kb_meta WHERE key='active_generation'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    value
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| Error::new("index_unavailable", "no committed index generation"))
}
pub fn load_documents(conn: &Connection) -> Result<Vec<Document>> {
    let mut statement = conn.prepare("SELECT root_key,path,document_key,sha256,mtime,size,metadata_json,links_json FROM document_manifest ORDER BY root_key,path")?;
    let rows = statement.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, f64>(4)?,
            integer_column::<u64>(r, 5)?,
            r.get::<_, String>(6)?,
            r.get::<_, String>(7)?,
        ))
    })?;
    let mut docs = Vec::new();
    for row in rows {
        let (root_key, path, key, sha256, mtime, size, meta, links) = row?;
        let mut chunks_stmt = conn.prepare("SELECT p.chunk_key,p.parent_key,p.heading_path,p.line_start,p.line_end,c.body FROM chunk_provenance p JOIN chunks c ON c.rowid=p.chunk_rowid WHERE p.document_key=? ORDER BY p.line_start,p.chunk_key")?;
        let chunks = chunks_stmt
            .query_map([&key], |r| {
                Ok(Chunk {
                    key: r.get(0)?,
                    parent_key: r.get(1)?,
                    heading: r.get(2)?,
                    line_start: integer_column(r, 3)?,
                    line_end: integer_column(r, 4)?,
                    body: r.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        docs.push(Document {
            root_key,
            path,
            key,
            sha256,
            mtime,
            size,
            meta: serde_json::from_str(&meta)?,
            links: serde_json::from_str(&links)?,
            chunks,
        });
    }
    Ok(docs)
}
pub fn index(config: &Config, force: bool) -> Result<Envelope<Value>> {
    crate::config::validate_profile(config.profile.as_str())?;
    if let Some(parent) = config.database.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(&config.database)?;
    conn.busy_timeout(Duration::from_millis(2000))?;
    let old_version = reject_newer(&conn)?;
    verify_legacy(&conn)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "FULL")?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch(SCHEMA)?;
    let fingerprint = config.fingerprint();
    let previous_fingerprint: Option<String> = tx
        .query_row(
            "SELECT value FROM kb_meta WHERE key='config_fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let old_generation = if old_version == 0 {
        0
    } else {
        generation(&tx)?
    };
    let previous = if old_version == SCHEMA_VERSION {
        load_documents(&tx)?
    } else {
        Vec::new()
    };
    let docs = if !force && previous_fingerprint.as_ref() == Some(&fingerprint) {
        ingest::scan_cached(config, &previous)?
    } else {
        ingest::scan(config)?
    };
    let before: BTreeMap<_, _> = previous
        .iter()
        .map(|d| ((d.root_key.as_str(), d.path.as_str()), d.sha256.as_str()))
        .collect();
    let after: BTreeMap<_, _> = docs
        .iter()
        .map(|d| ((d.root_key.as_str(), d.path.as_str()), d.sha256.as_str()))
        .collect();
    let added = after.keys().filter(|k| !before.contains_key(*k)).count();
    let removed = before.keys().filter(|k| !after.contains_key(*k)).count();
    let updated = after
        .iter()
        .filter(|(k, h)| before.get(*k).is_some_and(|v| v != *h))
        .count();
    let unchanged = docs.len() - added - updated;
    let chunk_count: usize = docs.iter().map(|d| d.chunks.len()).sum();
    #[cfg(feature = "full")]
    let vectors = if config.profile.is_full() {
        Some(crate::vector_index::prepare(
            config,
            &tx,
            &docs,
            force,
            previous_fingerprint.as_ref() != Some(&fingerprint),
            old_generation,
        )?)
    } else {
        None
    };
    let output = |generation, data| {
        let result = config.envelope(generation, data);
        #[cfg(feature = "full")]
        let result = {
            let mut result = result;
            if let Some(candidate) = &vectors {
                result.embedding_contract_id = Some(candidate.model.contract.id.clone());
                result.model_contracts = json!({"embedding":candidate.model.contract,"embedding_observations":candidate.model.observations()});
            }
            result
        };
        result
    };
    let changed = force
        || before != after
        || previous_fingerprint.as_ref() != Some(&fingerprint)
        || old_generation == 0
        || {
            #[cfg(feature = "full")]
            {
                vectors
                    .as_ref()
                    .is_some_and(|candidate| candidate.needs_publication)
            }
            #[cfg(not(feature = "full"))]
            {
                false
            }
        };
    if !changed {
        tx.commit()?;
        return Ok(output(
            old_generation,
            json!({"added":0,"updated":0,"removed":0,"unchanged":unchanged,"files":docs.len(),"chunks":chunk_count,"published":false}),
        ));
    }
    let next = old_generation
        .checked_add(1)
        .ok_or_else(|| Error::new("generation_exhausted", "index generation exhausted"))?;
    // Candidate replacement is invisible until COMMIT. WAL retains the previous complete
    // rows for pinned readers; no filesystem swap or cross-resource snapshot is implied.
    let rebuild = force || old_version == 0 || previous_fingerprint.as_ref() != Some(&fingerprint);
    let has_vectors: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_embeddings')",
        [],
        |r| r.get(0),
    )?;
    if rebuild {
        tx.execute_batch("DELETE FROM chunk_provenance; DELETE FROM chunks; DELETE FROM document_manifest; DELETE FROM files;")?;
        if has_vectors {
            tx.execute("DELETE FROM chunk_embeddings", [])?;
        }
    } else {
        for doc in &previous {
            if after.get(&(doc.root_key.as_str(), doc.path.as_str())) == Some(&doc.sha256.as_str())
            {
                continue;
            }
            tx.execute("DELETE FROM chunks WHERE rowid IN (SELECT chunk_rowid FROM chunk_provenance WHERE document_key=?)", [&doc.key])?;
            tx.execute(
                "DELETE FROM chunk_provenance WHERE document_key=?",
                [&doc.key],
            )?;
            tx.execute(
                "DELETE FROM document_manifest WHERE document_key=?",
                [&doc.key],
            )?;
            let path = if config.roots.len() == 1 {
                doc.path.clone()
            } else {
                format!("{}:{}", doc.root_key, doc.path)
            };
            tx.execute("DELETE FROM files WHERE path=?", [&path])?;
            if has_vectors {
                tx.execute(
                    "DELETE FROM chunk_embeddings WHERE document_key=?",
                    [&doc.key],
                )?;
            }
        }
    }
    tx.execute("DELETE FROM root_registry", [])?;
    for (key, root) in &config.roots {
        tx.execute(
            "INSERT INTO root_registry VALUES(?,?,?,?)",
            params![key, key, root.path.to_string_lossy(), fingerprint],
        )?;
    }
    {
        let mut file_stmt = tx.prepare("INSERT INTO files VALUES(?,?,?,?)")?;
        let mut doc_stmt = tx.prepare(
            "INSERT INTO document_manifest VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        )?;
        let mut fts_stmt = tx.prepare("INSERT INTO chunks(path,heading,body,project,type,status,tags,line_start) VALUES(?,?,?,?,?,?,?,?)")?;
        let mut provenance_stmt =
            tx.prepare("INSERT INTO chunk_provenance VALUES(?,?,?,?,?,?,?,?)")?;
        for doc in &docs {
            if !rebuild
                && before.get(&(doc.root_key.as_str(), doc.path.as_str()))
                    == Some(&doc.sha256.as_str())
            {
                continue;
            }
            let root = &config.roots[&doc.root_key];
            let legacy_path = if config.roots.len() == 1 {
                doc.path.clone()
            } else {
                format!("{}:{}", doc.root_key, doc.path)
            };
            file_stmt.execute(params![
                legacy_path,
                doc.sha256,
                doc.mtime,
                sql_integer(doc.size)?
            ])?;
            doc_stmt.execute(params![
                doc.root_key,
                doc.path,
                doc.key,
                doc.sha256,
                ingest::PARSER_VERSION,
                ingest::CHUNKER_VERSION,
                doc.meta.status,
                root.role,
                root.trust,
                serde_json::to_string(
                    &json!({"supersedes":doc.meta.supersedes,"conflicts":doc.meta.conflicts})
                )?,
                doc.meta.id,
                doc.meta.doc_type,
                serde_json::to_string(&doc.meta.tags)?,
                doc.meta.updated,
                next,
                serde_json::to_string(&doc.meta)?,
                serde_json::to_string(&doc.links)?,
                doc.mtime,
                sql_integer(doc.size)?
            ])?;
            for chunk in &doc.chunks {
                fts_stmt.execute(params![
                    legacy_path,
                    chunk.heading,
                    chunk.body,
                    doc.meta.id,
                    doc.meta.doc_type,
                    doc.meta.status,
                    doc.meta.tags.join(", "),
                    sql_integer(chunk.line_start)?
                ])?;
                provenance_stmt.execute(params![
                    tx.last_insert_rowid(),
                    doc.key,
                    chunk.key,
                    chunk.parent_key,
                    chunk.heading,
                    sql_integer(chunk.line_start)?,
                    sql_integer(chunk.line_end)?,
                    doc.sha256
                ])?;
            }
        }
    }
    if has_vectors {
        tx.execute("UPDATE chunk_embeddings SET generation=?", [next])?;
    }
    #[cfg(feature = "full")]
    if let Some(candidate) = &vectors {
        candidate.publish(&tx, next)?;
    }
    tx.execute("UPDATE document_manifest SET generation=?", [next])?;
    tx.execute("INSERT INTO generations(id) VALUES(?)", [next])?;
    for (key, value) in [
        ("active_generation", next.to_string()),
        ("config_fingerprint", fingerprint),
    ] {
        tx.execute(
            "INSERT INTO kb_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [key, &value],
        )?;
    }
    tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    tx.execute("INSERT INTO chunks(chunks) VALUES('integrity-check')", [])?;
    let integrity: String = tx.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    if integrity != "ok" {
        return Err(Error::new("index_corrupt", integrity));
    }
    tx.commit()?;
    Ok(output(
        next,
        json!({"added":added,"updated":updated,"removed":removed,"unchanged":unchanged,"files":docs.len(),"chunks":chunk_count,"published":true}),
    ))
}
pub fn stats(config: &Config) -> Result<Envelope<Value>> {
    let conn = open_reader(config)?;
    let (files, bytes): (i64, i64) = conn.query_row(
        "SELECT count(*),coalesce(sum(size),0) FROM files",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let chunks: i64 = conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
    let ids: i64 = conn.query_row(
        "SELECT count(DISTINCT project_id) FROM document_manifest WHERE project_id<>''",
        [],
        |r| r.get(0),
    )?;
    let mut result = config.envelope(
        generation(&conn)?,
        json!({"files":files,"bytes":bytes,"chunks":chunks,"projects":ids,"schema_epoch":SCHEMA_VERSION,"sqlite_version":rusqlite::version(),"database":config.database}),
    );
    result.retrieval_legs.clear();
    Ok(result)
}
