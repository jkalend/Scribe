#![cfg(feature = "full")]
//! Exercise the production rmcp SDK over real stdio with a synthetic loopback
//! provider. These are protocol/engine tests, not real-model quality results.
mod support;
use kb::{
    config::{Config, Limits, Profile, Root},
    full_config::FullConfig,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use support::{OllamaServer, Reply, Request, standard_reply};

struct Fixture {
    temp: tempfile::TempDir,
    config: Config,
    file: PathBuf,
}
#[derive(serde::Serialize)]
struct FileSettings<'a> {
    profile: &'a str,
    index: IndexSettings<'a>,
    roots: &'a BTreeMap<String, Root>,
    limits: &'a Limits,
    full: Option<&'a FullConfig>,
}
#[derive(serde::Serialize)]
struct IndexSettings<'a> {
    database: &'a Path,
}
impl Fixture {
    fn new(server: &OllamaServer) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("docs");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("guide.md"), "---\nid: GUIDE\ntitle: Synthetic guide\ntype: policy\nstatus: accepted\n---\n# Synthetic guide\nBackups run nightly.\n").unwrap();
        let mut model = server.model_config(temp.path());
        model.timeout_ms = 2000;
        let config = Config {
            database: temp.path().join("index.sqlite"),
            profile: Profile::Full,
            full: Some(FullConfig {
                embedding: Some(model.clone()),
                generator: Some(model),
                ..Default::default()
            }),
            roots: BTreeMap::from([(
                "docs".into(),
                Root {
                    path: root.canonicalize().unwrap(),
                    role: "spec".into(),
                    trust: "synthetic".into(),
                    ignore_dirs: vec![],
                },
            )]),
            limits: Limits::default(),
        };
        kb::db::index(&config, false).unwrap();
        let file = temp.path().join("kb.toml");
        let result = Self { temp, config, file };
        result.save();
        result
    }
    fn save(&self) {
        let raw = FileSettings {
            profile: "full",
            index: IndexSettings {
                database: &self.config.database,
            },
            roots: &self.config.roots,
            limits: &self.config.limits,
            full: self.config.full.as_ref(),
        };
        std::fs::write(&self.file, toml::to_string(&raw).unwrap()).unwrap();
    }
    fn launch(&self, profile: &str) -> Client {
        Client::launch(&self.file, profile)
    }
}
struct Client {
    child: Child,
    input: Option<ChildStdin>,
    messages: Receiver<Value>,
    buffered: Vec<Value>,
}
impl Client {
    fn launch(path: &Path, profile: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_kb"))
            .arg("--config")
            .arg(path)
            .args(["--profile", profile, "mcp"])
            .env_remove("KB_CONFIG")
            .env_remove("KB_PROFILE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let output = child.stdout.take().unwrap();
        let (tx, messages) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let value = serde_json::from_str(&line.unwrap()).expect("protocol-only stdout");
                if tx.send(value).is_err() {
                    break;
                }
            }
        });
        let mut result = Self {
            child,
            input,
            messages,
            buffered: vec![],
        };
        result.request(1,"initialize",json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"full-synthetic-protocol","version":"1"}}));
        let initialized = result.response(1);
        assert!(initialized.get("error").is_none(), "{initialized}");
        assert_eq!(initialized["result"]["serverInfo"]["name"], "scribe");
        result.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
        result
    }
    fn send(&mut self, value: Value) {
        let input = self.input.as_mut().unwrap();
        serde_json::to_writer(&mut *input, &value).unwrap();
        input.write_all(b"\n").unwrap();
        input.flush().unwrap();
    }
    fn request(&mut self, id: u64, method: &str, params: Value) {
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
    }
    fn response(&mut self, id: u64) -> Value {
        if let Some(i) = self.buffered.iter().position(|v| v["id"] == id) {
            return self.buffered.remove(i);
        }
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            let value = self
                .messages
                .recv_timeout(until.saturating_duration_since(Instant::now()))
                .expect("bounded protocol reply");
            if value["id"] == id {
                return value;
            }
            self.buffered.push(value);
        }
    }
    fn call(&mut self, id: u64, name: &str, arguments: Value) -> Value {
        self.request(id, "tools/call", json!({"name":name,"arguments":arguments}));
        self.response(id)
    }
    fn list(&mut self) -> Vec<Value> {
        self.request(2, "tools/list", json!({}));
        self.response(2)["result"]["tools"]
            .as_array()
            .unwrap()
            .clone()
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn answer_reply(request: &Request) -> Reply {
    if request.path != "/api/generate" {
        return standard_reply(request);
    }
    let response = json!({"abstained":false,"abstain_reason":null,"claims":[{"text":"Backups run nightly.","citations":[{"evidence_id":"S1","quote":"Backups run nightly."}]}],"conflicts":[]});
    Reply::json(
        json!({"model":"fixture:latest","response":response.to_string(),"done":true,"done_reason":"stop","prompt_eval_count":request.body["prompt"].as_str().unwrap().split_whitespace().count(),"eval_count":10}),
    )
}
fn answer_args() -> Value {
    json!({"question":"When do backups run?","budget_bytes":16384})
}

#[test]
fn startup_profile_selects_read_only_surface_even_in_full_binary() {
    let server = OllamaServer::start(answer_reply);
    let fixture = Fixture::new(&server);
    let before = server.requests().len();
    let mut light = fixture.launch("light");
    let tools = light.list();
    let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["search", "get", "context"]);
    for t in &tools {
        assert_eq!(t["annotations"]["readOnlyHint"], true);
    }
    let rejected = light.call(3, "answer", answer_args());
    assert!(rejected.get("error").is_some(), "{rejected}");
    let search = light.call(4, "search", json!({"query":"backups"}));
    assert_eq!(search["result"]["structuredContent"]["profile"], "light");
    assert_eq!(
        before,
        server.requests().len(),
        "light tools never touch configured models"
    );
    let mut full = fixture.launch("full");
    let tools = full.list();
    let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["search", "get", "context", "answer"]);
    for t in &tools {
        assert_eq!(t["annotations"]["readOnlyHint"], true);
    }
    let answer = tools.iter().find(|t| t["name"] == "answer").unwrap();
    assert_eq!(
        answer["outputSchema"]["anyOf"][0]["properties"]["profile"]["const"],
        "full"
    );
    assert!(
        !names
            .iter()
            .any(|n| ["index", "eval", "download", "settings"].contains(n))
    );
}
#[test]
fn full_answer_dispatch_is_grounded_and_arguments_cannot_change_scope() {
    let server = OllamaServer::start(answer_reply);
    let fixture = Fixture::new(&server);
    let mut client = fixture.launch("full");
    let before = server.requests().len();
    for (i, args) in [
        json!({"question":"backups","budget_bytes":511}),
        json!({"question":"backups","budget_bytes":16384,"profile":"light"}),
        json!({"question":"backups","budget_bytes":16384,"filters":{"root":"docs","unknown":"x"}}),
        json!({"question":"backups","budget_bytes":16384,"limit":0}),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            client
                .call(10 + i as u64, "answer", args)
                .get("error")
                .is_some()
        );
    }
    assert_eq!(
        before,
        server.requests().len(),
        "invalid arguments must fail before model calls"
    );
    let response = client.call(20, "answer", answer_args());
    assert_eq!(response["result"]["isError"], false, "{response}");
    let result = &response["result"]["structuredContent"];
    assert_eq!(result["profile"], "full");
    assert!(result["embedding_contract_id"].is_string());
    assert!(result["generator_contract_id"].is_string());
    assert_eq!(result["data"]["abstained"], false);
    assert_eq!(result["data"]["verification"]["citations_verified"], 1);
    assert_eq!(
        result["data"]["claims"][0]["citations"][0]["provenance"]["path"],
        "guide.md"
    );
    assert!(
        serde_json::to_vec(&response).unwrap().len() <= fixture.config.limits.max_response_bytes
    );
    let calls = server.requests();
    let generation = calls.iter().find(|r| r.path == "/api/generate").unwrap();
    assert_eq!(generation.body["raw"], true);
    assert_eq!(generation.body["stream"], false);
    assert!(generation.body["format"].is_object());
}
#[test]
fn full_answer_errors_report_selected_profile_without_lexical_fallback() {
    let server = OllamaServer::start(standard_reply);
    let mut fixture = Fixture::new(&server);
    fixture.config.full.as_mut().unwrap().generator = None;
    fixture.save();
    let mut client = fixture.launch("full");
    let response = client.call(3, "answer", answer_args());
    assert_eq!(response["result"]["isError"], true, "{response}");
    let result = &response["result"]["structuredContent"];
    assert_eq!(result["profile"], "full");
    assert!(
        result["error"]["code"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
    assert!(result.get("data").is_none());
}
#[test]
fn answer_timeout_and_cancellation_remain_bounded() {
    let server = OllamaServer::start(|r| {
        if r.path == "/api/generate" {
            thread::sleep(Duration::from_millis(800));
        }
        answer_reply(r)
    });
    let mut fixture = Fixture::new(&server);
    fixture.config.limits.timeout_ms = 250;
    // Limits are part of the captured source/index policy. Publish that policy
    // before testing a model-call deadline, rather than exercising stale config.
    kb::db::index(&fixture.config, false).unwrap();
    fixture.save();
    let mut client = fixture.launch("full");
    let start = Instant::now();
    let response = client.call(3, "answer", answer_args());
    assert_eq!(response["result"]["structuredContent"]["profile"], "full");
    assert_eq!(
        response["result"]["structuredContent"]["error"]["code"], "timeout",
        "{response}"
    );
    assert!(start.elapsed() < Duration::from_secs(3));
    drop(client);
    // Cancellation can be sent before expensive work starts; the SDK may cancel
    // the request future without returning a tool response. Session remains live.
    let mut client = fixture.launch("full");
    client.request(
        4,
        "tools/call",
        json!({"name":"answer","arguments":answer_args()}),
    );
    client.send(json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":4,"reason":"synthetic cancellation"}}));
    assert_eq!(client.list().len(), 4);
}
#[test]
fn evaluation_reports_both_profiles_and_mechanical_answer_rows() {
    let server = OllamaServer::start(answer_reply);
    let fixture = Fixture::new(&server);
    let qrels = fixture.temp.path().join("qrels.jsonl");
    std::fs::write(&qrels,"{\"id\":\"guide\",\"query\":\"backups\",\"relevant\":[\"GUIDE\"],\"split\":\"held_out\",\"category\":\"term\"}\n").unwrap();
    let report = kb::eval::compare(&fixture.config, &qrels).unwrap();
    assert_eq!(report.data["light"]["profile"], "light");
    assert_eq!(report.profile, "full");
    assert_eq!(report.data["rows"].as_array().unwrap().len(), 1);
    assert_eq!(
        report.data["rows"][0]["full_execution"]["retrieval_legs"],
        json!(["lexical", "dense"])
    );
    let answers = fixture.temp.path().join("answers.jsonl");
    std::fs::write(&answers,"{\"id\":\"guide\",\"question\":\"When do backups run?\",\"split\":\"held_out\",\"category\":\"answerable\",\"expected_sources\":[\"GUIDE\"],\"expected_facts\":[\"Backups run nightly.\"],\"synthetic\":true}\n").unwrap();
    let report = kb::answer_eval::run(&fixture.config, &answers, None).unwrap();
    assert_eq!(report.data["aggregate"]["answerable_coverage"], 1.0);
    assert_eq!(report.data["aggregate"]["false_abstention"], 0.0);
    assert_eq!(report.data["aggregate"]["citation_resolution"], 1.0);
    assert_eq!(report.data["rows"][0]["human_semantic"], Value::Null);
    assert_eq!(report.data["release_acceptance"], false);
}
