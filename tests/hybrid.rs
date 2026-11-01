#![cfg(feature = "full")]

mod support;

use kb::{
    config::{Config, Limits, Profile, Root},
    context, db,
    embed::encode_vector,
    full_config::FullConfig,
    model::{ContextRequest, Filters, GetRequest, Hit, SearchRequest},
    retrieve,
};
use rusqlite::Connection;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use support::{OllamaServer, Reply, standard_reply};
use tempfile::TempDir;

// This is a synthetic loopback provider, not evidence of real-model quality.
// Production retrieval still uses the provider adapter and persisted SQLite vectors.
struct Fixture {
    _temp: TempDir,
    server: OllamaServer,
    config: Config,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let server = OllamaServer::start(|request| {
            if request.path != "/api/embed" {
                return standard_reply(request);
            }
            let inputs = request.body["input"].as_array().unwrap();
            if inputs
                .iter()
                .any(|input| input.as_str().unwrap().contains("FAIL_EMBEDDING_MARKER"))
            {
                return Reply::raw(503, b"synthetic embedding outage".to_vec());
            }
            let vectors: Vec<Vec<f32>> = inputs
                .iter()
                .map(|input| {
                    let text = input.as_str().unwrap();
                    if text.contains("LEXICAL_MARKER") {
                        vec![0.0, 1.0, 0.0]
                    } else if text.contains("OTHER_MARKER") {
                        vec![0.0, 0.0, 1.0]
                    } else {
                        vec![1.0, 0.0, 0.0]
                    }
                })
                .collect();
            let count: usize = inputs
                .iter()
                .map(|v| v.as_str().unwrap().split_whitespace().count())
                .sum();
            Reply::json(
                json!({"model":"fixture:latest","embeddings":vectors,"prompt_eval_count":count}),
            )
        });
        let mut roots = BTreeMap::new();
        for (name, role) in [("notes", "canonical"), ("other", "external")] {
            let path = temp.path().join(name);
            std::fs::create_dir(&path).unwrap();
            roots.insert(
                name.into(),
                Root {
                    path,
                    role: role.into(),
                    trust: "operator".into(),
                    ignore_dirs: vec![],
                },
            );
        }
        let mut config = Config {
            database: temp.path().join("index.sqlite"),
            roots,
            limits: Limits::default(),
            profile: Profile::Full,
            full: Some(FullConfig {
                embedding: Some(server.model_config(temp.path())),
                ..Default::default()
            }),
        };
        config.validate().unwrap();
        for (root, path, body) in [
            (
                "notes",
                "a.md",
                "---\nid: P-LEX\ntype: product\nstatus: active\ntags: [rust]\nupdated: 2025-01-01\n---\n# Repairs\nrepair LEXICAL_MARKER local evidence.\n",
            ),
            (
                "notes",
                "b.md",
                "---\nid: P-SEM\ntype: decision\nstatus: active\ntags: [rust]\nupdated: 2024-01-01\n---\n# Restoration\nRecovery without the lexical query word.\n",
            ),
            (
                "notes",
                "c.md",
                "---\nid: P-OLD\ntype: product\nstatus: archived\ntags: [secret]\nupdated: 2023-01-01\n---\n# Old\nrepair repair archived evidence.\n",
            ),
            (
                "other",
                "d.md",
                "---\nid: P-OTHER\ntype: product\nstatus: active\ntags: [rust]\nupdated: 2025-01-01\n---\n# Other\nrepair OTHER_MARKER external evidence.\n",
            ),
        ] {
            std::fs::write(config.roots[root].path.join(path), body).unwrap();
        }
        db::index(&config, false).unwrap();
        Self {
            _temp: temp,
            server,
            config,
        }
    }
    fn search(&self, query: &str) -> kb::error::Result<kb::model::Envelope<Vec<Hit>>> {
        retrieve::search(
            &self.config,
            &SearchRequest {
                query: query.into(),
                limit: 10,
                ..Default::default()
            },
        )
    }
    fn writer(&self) -> Connection {
        Connection::open(&self.config.database).unwrap()
    }
}

#[test]
fn failed_embedding_batch_rolls_back_and_light_edits_require_full_refresh() {
    let fixture = Fixture::new();
    let notes = &fixture.config.roots["notes"].path;
    for i in 0..16 {
        std::fs::write(
            notes.join(format!("added-{i:02}.md")),
            format!("---\nid: NEW-{i}\n---\n# Candidate\nCandidate document {i}.\n"),
        )
        .unwrap();
    }
    let failed_source = notes.join("zz-failed.md");
    std::fs::write(
        &failed_source,
        "---\nid: FAIL-CANDIDATE\n---\n# Candidate\nFAIL_EMBEDDING_MARKER\n",
    )
    .unwrap();
    assert_eq!(
        db::index(&fixture.config, false).unwrap_err().code,
        "provider_http_error"
    );
    let previous = fixture.search("P-LEX").unwrap();
    assert_eq!(previous.index_generation, 1);
    assert_eq!(previous.data[0].path, "a.md");
    assert!(
        previous.data[0]
            .body
            .contains("repair LEXICAL_MARKER local evidence.")
    );

    std::fs::remove_file(failed_source).unwrap();
    assert_eq!(
        db::index(&fixture.config, false).unwrap().index_generation,
        2
    );
    let published = fixture.search("NEW-15").unwrap();
    assert_eq!(published.index_generation, 2);
    assert_eq!(published.data[0].path, "added-15.md");
    assert!(published.data[0].body.contains("Candidate document 15."));
    let old_key = fixture
        .search("P-SEM")
        .unwrap()
        .data
        .into_iter()
        .next()
        .unwrap()
        .document_key;

    let original = std::fs::read_to_string(notes.join("a.md")).unwrap();
    std::fs::write(
        notes.join("a.md"),
        original.replace("local evidence.", "updated source evidence."),
    )
    .unwrap();
    std::fs::rename(notes.join("b.md"), notes.join("moved.md")).unwrap();
    std::fs::remove_file(notes.join("c.md")).unwrap();
    let mut light = fixture.config.clone();
    light.profile = Profile::Light;
    assert_eq!(db::index(&light, false).unwrap().index_generation, 3);
    assert_eq!(
        fixture.search("P-OTHER").unwrap_err().code,
        "embeddings_stale"
    );

    assert_eq!(
        db::index(&fixture.config, false).unwrap().index_generation,
        4
    );
    let renamed = fixture.search("P-SEM").unwrap();
    assert_eq!(renamed.index_generation, 4);
    assert_eq!(renamed.data[0].path, "moved.md");
    assert_ne!(renamed.data[0].document_key, old_key);
    let updated = fixture.search("P-LEX").unwrap();
    assert!(updated.data[0].body.contains("updated source evidence."));
    assert_eq!(
        retrieve::get(
            &fixture.config,
            &GetRequest {
                key: "notes:c.md".into(),
                line_start: None,
                line_end: None,
            },
        )
        .unwrap_err()
        .code,
        "not_found"
    );
}

fn paths(hits: &[Hit]) -> BTreeSet<&str> {
    hits.iter().map(|h| h.path.as_str()).collect()
}

#[test]
fn rrf_preserves_raw_legs_and_is_deterministic_with_real_sqlite_vectors() {
    let fixture = Fixture::new();
    let result = fixture.search("repair").unwrap();
    assert_eq!(result.profile, "full");
    assert_eq!(result.fusion_method.as_deref(), Some("rrf"));
    assert_eq!(result.retrieval_legs, ["lexical", "dense"]);
    assert!(result.embedding_contract_id.is_some());
    assert_eq!(
        result.model_contracts["embedding"]["id"],
        json!(result.embedding_contract_id)
    );
    assert!(
        result
            .data
            .iter()
            .any(|h| h.path == "b.md" && h.legs.iter().all(|l| l.name == "dense"))
    );
    for (index, hit) in result.data.iter().enumerate() {
        assert_eq!(hit.fused_rank, Some(index + 1));
        let expected: f64 = hit.legs.iter().map(|l| 1.0 / (60.0 + l.rank as f64)).sum();
        assert!((hit.fused_score.unwrap() - expected).abs() < 1e-12);
        assert!(hit.legs.iter().all(|l| matches!(
            l.score_kind.as_str(),
            "cosine_higher_is_better" | "bm25_lower_is_better"
        )));
    }
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::to_value(fixture.search("repair").unwrap()).unwrap()
    );
    let conn = fixture.writer();
    let count: i64 = conn
        .query_row("SELECT count(*) FROM chunk_embeddings", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 4);
    let embeddings = fixture
        .server
        .requests()
        .into_iter()
        .filter(|r| r.path == "/api/embed")
        .collect::<Vec<_>>();
    assert!(embeddings.iter().any(|r| {
        r.body["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "repair")
    }));
}

#[test]
fn every_metadata_filter_precedes_both_legs_and_candidate_limits() {
    let mut fixture = Fixture::new();
    fixture.config.full.as_mut().unwrap().retrieval.candidates = 1;
    let cases = [
        (
            Filters {
                root: Some("notes".into()),
                ..Default::default()
            },
            vec!["a.md", "b.md", "c.md"],
        ),
        (
            Filters {
                role: Some("canonical".into()),
                ..Default::default()
            },
            vec!["a.md", "b.md", "c.md"],
        ),
        (
            Filters {
                status: Some("active".into()),
                ..Default::default()
            },
            vec!["a.md", "b.md", "d.md"],
        ),
        (
            Filters {
                project: Some("P-LEX".into()),
                ..Default::default()
            },
            vec!["a.md"],
        ),
        (
            Filters {
                doc_type: Some("decision".into()),
                ..Default::default()
            },
            vec!["b.md"],
        ),
        (
            Filters {
                tag: Some("rust".into()),
                ..Default::default()
            },
            vec!["a.md", "b.md", "d.md"],
        ),
        (
            Filters {
                after: Some("2025-01-01".into()),
                ..Default::default()
            },
            vec!["a.md", "d.md"],
        ),
        (
            Filters {
                before: Some("2023-01-01".into()),
                ..Default::default()
            },
            vec!["c.md"],
        ),
    ];
    for (filters, expected) in cases {
        let result = retrieve::search(
            &fixture.config,
            &SearchRequest {
                query: "repair".into(),
                filters: filters.clone(),
                limit: 10,
            },
        )
        .unwrap();
        assert_eq!(paths(&result.data), expected.iter().copied().collect());
        // With a one-result request the first available filtered dense candidate
        // must still have rank one, not its rank in the unfiltered corpus.
        let limited = retrieve::search(
            &fixture.config,
            &SearchRequest {
                query: "repair".into(),
                filters,
                limit: 1,
            },
        )
        .unwrap();
        assert_eq!(limited.data.len(), 1);
        assert!(expected.contains(&limited.data[0].path.as_str()));
        assert!(limited.data[0].legs.iter().all(|leg| leg.rank == 1));
    }
}

#[test]
fn weighted_fusion_normalizes_each_leg_before_combining_scores() {
    let mut fixture = Fixture::new();
    let retrieval = &mut fixture.config.full.as_mut().unwrap().retrieval;
    retrieval.fusion = "weighted_score".into();
    retrieval.lexical_weight = 0.3;
    let result = fixture.search("repair").unwrap();
    assert_eq!(result.fusion_method.as_deref(), Some("weighted_score"));
    let bounds = |name: &str| {
        result
            .data
            .iter()
            .flat_map(|h| &h.legs)
            .filter(|l| l.name == name)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), leg| {
                (min.min(leg.raw_score), max.max(leg.raw_score))
            })
    };
    for hit in &result.data {
        let expected: f64 = hit
            .legs
            .iter()
            .map(|leg| {
                let (min, max) = bounds(&leg.name);
                let normalized = if min == max {
                    1.0
                } else if leg.name == "lexical" {
                    (max - leg.raw_score) / (max - min)
                } else {
                    (leg.raw_score - min) / (max - min)
                };
                normalized * if leg.name == "lexical" { 0.3 } else { 0.7 }
            })
            .sum();
        assert!((hit.fused_score.unwrap() - expected).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&hit.fused_score.unwrap()));
    }
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .retrieval
        .lexical_weight = 0.0;
    let dense_only_weight = fixture.search("repair").unwrap();
    assert_eq!(
        dense_only_weight.data[0]
            .legs
            .iter()
            .find(|l| l.name == "dense")
            .unwrap()
            .raw_score,
        1.0
    );
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .retrieval
        .lexical_weight = 1.0;
    let lexical_only_weight = fixture.search("repair").unwrap();
    assert!(
        lexical_only_weight.data[0]
            .legs
            .iter()
            .any(|l| l.name == "lexical" && l.rank == 1)
    );
}

#[test]
fn exact_identity_is_pinned_but_never_bypasses_model_or_vector_validation() {
    let fixture = Fixture::new();
    let result = retrieve::search(
        &fixture.config,
        &SearchRequest {
            query: "P-LEX".into(),
            limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.data[0].path, "a.md");
    assert!(
        result.data[0]
            .legs
            .iter()
            .any(|l| l.score_kind == "exact_id")
    );
    fixture.writer().execute("DELETE FROM chunk_embeddings WHERE document_key IN (SELECT document_key FROM document_manifest WHERE path='b.md')", []).unwrap();
    assert_eq!(
        fixture.search("P-LEX").unwrap_err().code,
        "embeddings_stale"
    );
}

#[test]
fn filtered_out_exact_id_never_falls_back_to_unrelated_dense_results() {
    let fixture = Fixture::new();
    let request = SearchRequest {
        query: "P-LEX".into(),
        filters: Filters {
            root: Some("other".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let result = retrieve::search(&fixture.config, &request).unwrap();
    assert!(
        result.data.is_empty(),
        "the known ID is outside the requested root"
    );

    fixture.writer().execute(
        "DELETE FROM chunk_embeddings WHERE document_key IN (SELECT document_key FROM document_manifest WHERE path='a.md')",
        [],
    ).unwrap();
    assert_eq!(
        retrieve::search(&fixture.config, &request)
            .unwrap_err()
            .code,
        "embeddings_stale",
    );
}

#[test]
fn invalid_vectors_and_contracts_fail_closed_even_outside_filters() {
    let fixture = Fixture::new();
    let conn = fixture.writer();
    let key: (String, String, Vec<u8>) = conn.query_row("SELECT document_key,chunk_key,vector FROM chunk_embeddings WHERE document_key IN (SELECT document_key FROM document_manifest WHERE path='c.md')", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
    let request = SearchRequest {
        query: "repair".into(),
        filters: Filters {
            project: Some("P-LEX".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    conn.execute_batch("CREATE TEMP TABLE original_embeddings AS SELECT * FROM chunk_embeddings; CREATE TEMP TABLE original_meta AS SELECT * FROM kb_meta; CREATE TEMP TABLE original_contract AS SELECT * FROM embedding_contract; CREATE TEMP TABLE original_manifest AS SELECT * FROM document_manifest; CREATE TEMP TABLE original_provenance AS SELECT * FROM chunk_provenance;").unwrap();
    let mutations = [
        "UPDATE chunk_embeddings SET vector=X'00' WHERE document_key=?1",
        "UPDATE chunk_embeddings SET vector=zeroblob(12) WHERE document_key=?1",
        "UPDATE chunk_embeddings SET dimensions=4 WHERE document_key=?1",
        "UPDATE chunk_embeddings SET source_sha256='wrong' WHERE document_key=?1",
        "UPDATE chunk_embeddings SET generation=generation+1 WHERE document_key=?1",
        "UPDATE chunk_embeddings SET chunk_key='unrelated' WHERE document_key=?1",
        "UPDATE chunk_embeddings SET contract_id='wrong' WHERE document_key=?1",
        "DELETE FROM chunk_embeddings WHERE document_key=?1",
    ];
    for sql in mutations {
        conn.execute(sql, [&key.0]).unwrap();
        assert_eq!(
            retrieve::search(&fixture.config, &request)
                .unwrap_err()
                .code,
            "embeddings_stale",
            "{sql}"
        );
        conn.execute_batch("DELETE FROM chunk_embeddings; INSERT INTO chunk_embeddings SELECT * FROM original_embeddings;").unwrap();
    }
    for vector in [
        vec![f32::NAN, 0.0, 0.0],
        vec![f32::INFINITY, 0.0, 0.0],
        vec![2.0, 0.0, 0.0],
    ] {
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        conn.execute(
            "UPDATE chunk_embeddings SET vector=? WHERE document_key=?",
            rusqlite::params![bytes, key.0],
        )
        .unwrap();
        assert_eq!(
            retrieve::search(&fixture.config, &request)
                .unwrap_err()
                .code,
            "embeddings_stale"
        );
    }
    conn.execute(
        "UPDATE chunk_embeddings SET vector=? WHERE document_key=?",
        rusqlite::params![key.2, key.0],
    )
    .unwrap();
    for sql in [
        "UPDATE kb_meta SET value='wrong' WHERE key='embedding_contract_id'",
        "UPDATE kb_meta SET value='2' WHERE key='embedding_schema'",
        "UPDATE embedding_contract SET contract_json='{}'",
        "UPDATE document_manifest SET generation=generation+1 WHERE path='c.md'",
        "UPDATE chunk_provenance SET source_sha256='wrong' WHERE document_key IN (SELECT document_key FROM document_manifest WHERE path='c.md')",
    ] {
        conn.execute_batch(sql).unwrap();
        assert_eq!(
            retrieve::search(&fixture.config, &request)
                .unwrap_err()
                .code,
            "embeddings_stale",
            "{sql}"
        );
        conn.execute_batch("DELETE FROM kb_meta; INSERT INTO kb_meta SELECT * FROM original_meta; DELETE FROM embedding_contract; INSERT INTO embedding_contract SELECT * FROM original_contract; DELETE FROM document_manifest; INSERT INTO document_manifest SELECT * FROM original_manifest; DELETE FROM chunk_provenance; INSERT INTO chunk_provenance SELECT * FROM original_provenance;").unwrap();
    }
    // A valid different vector is used, proving production reads SQLite bytes.
    conn.execute(
        "UPDATE chunk_embeddings SET vector=? WHERE document_key=?",
        rusqlite::params![encode_vector(&[0.0, 1.0, 0.0]).unwrap(), key.0],
    )
    .unwrap();
    assert!(
        fixture
            .search("repair")
            .unwrap()
            .data
            .iter()
            .find(|h| h.path == "c.md")
            .unwrap()
            .legs
            .iter()
            .any(|l| l.name == "dense" && l.raw_score == 0.0)
    );
}

#[test]
fn explicit_context_byte_budget_preserves_full_metadata_and_source_checks() {
    let fixture = Fixture::new();
    let pack = context::pack(
        &fixture.config,
        &ContextRequest {
            query: "repair".into(),
            filters: Filters::default(),
            limit: 10,
            budget_bytes: 16_384,
        },
    )
    .unwrap();
    assert_eq!(pack.profile, "full");
    assert_eq!(pack.retrieval_legs, ["lexical", "dense"]);
    assert_eq!(pack.fusion_method.as_deref(), Some("rrf"));
    assert!(pack.embedding_contract_id.is_some());
    assert!(pack.model_contracts["embedding"]["details"].is_object());
    assert_eq!(pack.data["budget_unit"], "bytes");
    assert_eq!(
        pack.data["used_bytes"].as_u64().unwrap() as usize,
        serde_json::to_vec(&pack.data).unwrap().len()
    );
    assert!(serde_json::to_vec(&pack.data).unwrap().len() <= 16_384);
    assert!(
        pack.data["items"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["fused_rank"].is_number() && v["evidence_id"].is_string())
    );
    std::fs::write(fixture.config.roots["notes"].path.join("a.md"), "changed\n").unwrap();
    assert_eq!(fixture.search("P-LEX").unwrap_err().code, "stale_source");
}

#[test]
fn light_and_metadata_get_never_touch_provider_and_full_prerequisites_are_typed() {
    let mut fixture = Fixture::new();
    let before = fixture.server.requests().len();
    let get = retrieve::get(
        &fixture.config,
        &GetRequest {
            key: "P-LEX".into(),
            line_start: None,
            line_end: None,
        },
    )
    .unwrap();
    assert_eq!(get.profile, "full");
    assert!(get.retrieval_legs.is_empty());
    assert!(get.embedding_contract_id.is_none());
    assert_eq!(fixture.server.requests().len(), before);
    fixture.config.profile = Profile::Light;
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .embedding
        .as_mut()
        .unwrap()
        .base_url = "https://example.invalid:443".into();
    let light = fixture.search("repair").unwrap();
    assert_eq!(light.profile, "light");
    assert_eq!(light.retrieval_legs, ["lexical"]);
    assert!(
        light
            .data
            .iter()
            .all(|h| h.fused_rank.is_none() && h.fused_score.is_none())
    );
    assert_eq!(fixture.server.requests().len(), before);
    fixture.config.profile = Profile::Full;
    fixture.config.full = None;
    assert_eq!(
        fixture.search("repair").unwrap_err().code,
        "embedding_unavailable"
    );
}

#[test]
fn missing_vector_tables_and_invalid_fusion_are_not_lexical_fallbacks() {
    let mut fixture = Fixture::new();
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .embedding
        .as_mut()
        .unwrap()
        .query_prefix = "query: ".into();
    assert_eq!(
        fixture.search("P-LEX").unwrap_err().code,
        "embeddings_stale"
    );
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .embedding
        .as_mut()
        .unwrap()
        .query_prefix
        .clear();
    let tokenizer = fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .embedding
        .as_mut()
        .unwrap()
        .tokenizer_path
        .take();
    assert_eq!(
        fixture.search("P-LEX").unwrap_err().code,
        "tokenizer_unavailable"
    );
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .embedding
        .as_mut()
        .unwrap()
        .tokenizer_path = tokenizer;
    fixture
        .writer()
        .execute_batch("DROP TABLE chunk_embeddings; DROP TABLE embedding_contract;")
        .unwrap();
    assert_eq!(
        fixture.search("repair").unwrap_err().code,
        "embeddings_unavailable"
    );
    fixture.config.full.as_mut().unwrap().retrieval.fusion = "raw_score_sum".into();
    assert_eq!(
        fixture.search("repair").unwrap_err().code,
        "invalid_retrieval_config"
    );
    fixture.config.full.as_mut().unwrap().retrieval.fusion = "rrf".into();
    fixture
        .config
        .full
        .as_mut()
        .unwrap()
        .retrieval
        .lexical_weight = f64::NAN;
    assert_eq!(
        fixture.search("repair").unwrap_err().code,
        "invalid_retrieval_config"
    );
}
