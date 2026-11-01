use kb::{
    config::{Config, Limits, Root},
    db,
    error::Result,
    model::{ContextRequest, Filters, GetRequest, SearchRequest},
    retrieve,
};
use std::collections::BTreeMap;
use tempfile::TempDir;

fn setup() -> (TempDir, Config) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("notes");
    std::fs::create_dir(&root).unwrap();
    let mut config = Config {
        database: temp.path().join("index.sqlite"),
        roots: BTreeMap::from([(
            "notes".into(),
            Root {
                path: root,
                role: "canonical".into(),
                trust: "operator".into(),
                ignore_dirs: vec!["ignored".into()],
            },
        )]),
        limits: Limits::default(),
        profile: Default::default(),
        full: None,
    };
    config.validate().unwrap();
    (temp, config)
}
fn write(config: &Config, path: &str, body: &str) {
    std::fs::write(config.roots["notes"].path.join(path), body).unwrap();
}
fn search(config: &Config, query: &str) -> Result<kb::model::Envelope<Vec<kb::model::Hit>>> {
    retrieve::search(
        config,
        &SearchRequest {
            query: query.into(),
            ..Default::default()
        },
    )
}
#[test]
fn edit_rename_delete_and_failed_candidate_preserve_complete_generation() {
    let (_temp, config) = setup();
    write(
        &config,
        "one.md",
        "---\nid: P-ONE\ntags: [rust, local]\n---\n# Recovery\nalpha recovery evidence\n",
    );
    let first = db::index(&config, false).unwrap();
    assert_eq!(search(&config, "alpha").unwrap().data[0].path, "one.md");
    assert!(
        !db::index(&config, false).unwrap().data["published"]
            .as_bool()
            .unwrap()
    );
    write(
        &config,
        "one.md",
        "---\nid: P-ONE\n---\n# Recovery\nbeta recovery evidence\n",
    );
    assert_eq!(search(&config, "alpha").unwrap_err().code, "stale_source");
    let second = db::index(&config, false).unwrap();
    assert!(second.index_generation > first.index_generation);
    assert!(search(&config, "alpha").unwrap().data.is_empty());
    assert_eq!(search(&config, "beta").unwrap().data.len(), 1);
    write(
        &config,
        "invalid.md",
        "---\nid: [not-a-string]\n---\n# Invalid\nevidence\n",
    );
    assert!(db::index(&config, false).is_err());
    assert_eq!(
        search(&config, "beta").unwrap().index_generation,
        second.index_generation
    );
    std::fs::remove_file(config.roots["notes"].path.join("invalid.md")).unwrap();
    std::fs::rename(
        config.roots["notes"].path.join("one.md"),
        config.roots["notes"].path.join("renamed.md"),
    )
    .unwrap();
    db::index(&config, false).unwrap();
    assert_eq!(search(&config, "P-ONE").unwrap().data[0].path, "renamed.md");
    std::fs::remove_file(config.roots["notes"].path.join("renamed.md")).unwrap();
    db::index(&config, false).unwrap();
    assert!(search(&config, "beta").unwrap().data.is_empty());
}
#[test]
fn wal_reader_keeps_prior_snapshot_while_writer_publishes() {
    let (_temp, config) = setup();
    write(&config, "one.md", "# State\nbeforeword\n");
    db::index(&config, false).unwrap();
    let pinned = db::open_reader(&config).unwrap();
    let old = db::generation(&pinned).unwrap();
    write(&config, "one.md", "# State\nafterword\n");
    db::index(&config, false).unwrap();
    assert_eq!(db::generation(&pinned).unwrap(), old);
    let body: String = pinned
        .query_row("SELECT body FROM chunks", [], |r| r.get(0))
        .unwrap();
    assert!(body.contains("beforeword"));
    assert!(search(&config, "afterword").unwrap().index_generation > old);
    assert!(pinned.execute("DELETE FROM chunks", []).is_err());
}
#[test]
fn get_and_context_validate_citations_and_serialized_byte_budget() {
    let (_temp, config) = setup();
    write(
        &config,
        "one.md",
        "---\nid: P-ONE\n---\n# Evidence\nUnicode café recovery.\n\n## More\nrecovery details here.\n",
    );
    db::index(&config, false).unwrap();
    let hits = search(&config, "recovery").unwrap();
    let hit = &hits.data[0];
    let got = retrieve::get(
        &config,
        &GetRequest {
            key: hit.chunk_key.clone(),
            line_start: None,
            line_end: None,
        },
    )
    .unwrap();
    assert_eq!(got.data["body"], hit.body);
    let pack = kb::context::pack(
        &config,
        &ContextRequest {
            query: "recovery".into(),
            filters: Filters::default(),
            limit: 10,
            budget_bytes: 2048,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&pack.data).unwrap().len();
    assert!(bytes <= 2048);
    assert_eq!(pack.data["used_bytes"].as_u64().unwrap() as usize, bytes);
    assert!(!pack.data["items"].as_array().unwrap().is_empty());
    write(&config, "one.md", "# Evidence\nmodified recovery\n");
    assert_eq!(
        retrieve::get(
            &config,
            &GetRequest {
                key: hit.chunk_key.clone(),
                line_start: None,
                line_end: None
            }
        )
        .unwrap_err()
        .code,
        "stale_source"
    );
    assert_eq!(
        retrieve::get(
            &config,
            &GetRequest {
                key: "../secret.md".into(),
                line_start: None,
                line_end: None
            }
        )
        .unwrap_err()
        .code,
        "inaccessible_path"
    );
}
#[test]
fn filters_precede_limit_and_tags_are_exact_not_substrings() {
    let (_temp, config) = setup();
    write(
        &config,
        "a.md",
        "---\nid: P-A\nstatus: old\ntags: [rustacean]\nupdated: 2025-01-01\n---\n# State\nrecovery recovery recovery\n",
    );
    write(
        &config,
        "b.md",
        "---\nid: P-B\nstatus: active\ntags: [rust]\nupdated: 2026-01-01\n---\n# State\nrecovery\n",
    );
    db::index(&config, false).unwrap();
    let result = retrieve::search(
        &config,
        &SearchRequest {
            query: "recovery".into(),
            limit: 1,
            filters: Filters {
                tag: Some("rust".into()),
                status: Some("active".into()),
                after: Some("2026-01-01".into()),
                ..Default::default()
            },
        },
    )
    .unwrap();
    assert_eq!(result.data[0].project, "P-B");
    assert!(
        search(&config, "recovery OR notthere")
            .unwrap()
            .data
            .is_empty()
    );
}
#[test]
fn legacy_schema_migration_is_additive_and_future_epoch_fails_closed() {
    let (_temp, config) = setup();
    write(&config, "one.md", "# Recovery\nnew source evidence\n");
    let conn = rusqlite::Connection::open(&config.database).unwrap();
    conn.execute_batch("CREATE TABLE files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,mtime REAL NOT NULL,size INTEGER NOT NULL); CREATE VIRTUAL TABLE chunks USING fts5(path UNINDEXED,heading,body,project UNINDEXED,type UNINDEXED,status UNINDEXED,tags UNINDEXED,line_start UNINDEXED,tokenize='porter unicode61'); INSERT INTO files VALUES('old.md','old',0,0);").unwrap();
    assert_eq!(
        db::open_reader(&config).unwrap_err().code,
        "migration_required"
    );
    db::index(&config, false).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT count(*) FROM chunks WHERE chunks MATCH 'recovery'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
    conn.pragma_update(None, "user_version", 99).unwrap();
    assert_eq!(
        db::index(&config, false).unwrap_err().code,
        "schema_unsupported"
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM files", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}
#[test]
fn duplicate_ids_are_ambiguous_not_silently_overwritten() {
    let (_temp, config) = setup();
    write(
        &config,
        "a.md",
        "---\nid: P-DUP\n---\n# One\nfirst evidence\n",
    );
    write(
        &config,
        "b.md",
        "---\nid: P-DUP\n---\n# Two\nsecond evidence\n",
    );
    db::index(&config, false).unwrap();
    assert_eq!(search(&config, "P-DUP").unwrap().data.len(), 2);
    assert_eq!(
        retrieve::get(
            &config,
            &GetRequest {
                key: "P-DUP".into(),
                line_start: None,
                line_end: None
            }
        )
        .unwrap_err()
        .code,
        "ambiguous_identity"
    );
    assert!(
        !kb::graph::check(&config).unwrap().data["errors"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
