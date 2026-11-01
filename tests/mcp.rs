//! Real subprocess protocol tests. The test client writes JSON-RPC messages;
//! the production server delegates transport, negotiation and cancellation to rmcp.
use kb::config::{Config, Limits};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

struct Fixture {
    _temp: TempDir,
    config: Config,
    file: PathBuf,
    source: PathBuf,
}
impl Fixture {
    fn new(text: &str, limits: Limits) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("docs");
        std::fs::create_dir(&root).unwrap();
        let source = root.join("guide.md");
        std::fs::write(&source, text).unwrap();
        let file = temp.path().join("kb.toml");
        let database = temp.path().join("index.sqlite3");
        let quote =
            |path: &std::path::Path| serde_json::to_string(&path.to_string_lossy()).unwrap();
        let text = format!(
            "profile = \"light\"\n[index]\ndatabase = {}\n[roots.docs]\npath = {}\nrole = \"spec\"\ntrust = \"user-configured\"\n[limits]\nmax_query_bytes = {}\nmax_response_bytes = {}\nmax_context_bytes = {}\ntimeout_ms = {}\n",
            quote(&database),
            quote(&root),
            limits.max_query_bytes,
            limits.max_response_bytes,
            limits.max_context_bytes,
            limits.timeout_ms
        );
        std::fs::write(&file, text).unwrap();
        let config = Config::load(Some(&file), None, &[], Some("light".into())).unwrap();
        kb::db::index(&config, false).unwrap();
        Self {
            _temp: temp,
            config,
            file,
            source,
        }
    }
    fn small() -> Self {
        Self::new(
            "---\nid: P-MCP\ntitle: MCP retrieval guide\ntype: product\nstatus: active\ntags: [retrieval]\n---\n# MCP retrieval\nRetrieval evidence is local and source verified.\n\n## Safety\nOnly configured roots are readable.\n",
            Limits::default(),
        )
    }
    fn launch(&self) -> Client {
        Client::launch(&self.file)
    }
    fn exclusive_lock(&self) -> rusqlite::Connection {
        let conn = rusqlite::Connection::open(&self.config.database).unwrap();
        // Rollback-journal exclusive locking deterministically blocks a new
        // reader; no sleeps or artificial production hooks simulate retrieval.
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE; PRAGMA locking_mode=EXCLUSIVE; BEGIN EXCLUSIVE;").unwrap();
        conn
    }
}

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    messages: Receiver<std::result::Result<Value, String>>,
    buffered: Vec<Value>,
    stderr: Option<thread::JoinHandle<String>>,
}
impl Client {
    fn launch(config: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_kb"))
            .args(["--config"])
            .arg(config)
            .args(["--profile", "light", "mcp"])
            .env_remove("KB_CONFIG")
            .env_remove("KB_PROFILE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let output = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let stderr = thread::spawn(move || {
            let mut text = String::new();
            stderr.read_to_string(&mut text).unwrap();
            text
        });
        let (sender, messages) = mpsc::channel();
        thread::spawn(move || {
            let mut output = BufReader::new(output);
            loop {
                let mut line = String::new();
                match output.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let parsed = serde_json::from_str::<Value>(&line)
                            .map_err(|error| format!("non-protocol stdout: {error}: {line}"));
                        if sender.send(parsed).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        });
        Self {
            child,
            input,
            messages,
            buffered: vec![],
            stderr: Some(stderr),
        }
    }
    fn send(&mut self, message: Value) {
        let input = self.input.as_mut().unwrap();
        serde_json::to_writer(&mut *input, &message).unwrap();
        input.write_all(b"\n").unwrap();
        input.flush().unwrap();
    }
    fn request(&mut self, id: u64, method: &str, params: Value) {
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
    }
    fn response(&mut self, id: u64) -> Value {
        if let Some(index) = self.buffered.iter().position(|message| message["id"] == id) {
            return self.buffered.remove(index);
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let message = self
                .messages
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("server response before deadline")
                .expect("stdout contains JSON only");
            assert_eq!(message["jsonrpc"], "2.0");
            if message["id"] == id {
                return message;
            }
            self.buffered.push(message);
        }
    }
    fn initialize(&mut self) {
        self.request(1, "initialize", json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"scribe-subprocess-test","version":"1"}}));
        let response = self.response(1);
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(response["result"]["serverInfo"]["name"], "scribe");
        assert!(response["result"]["capabilities"].get("tools").is_some());
        assert!(
            response["result"]["capabilities"]
                .get("resources")
                .is_none()
        );
        self.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    }
    fn call(&mut self, id: u64, tool: &str, arguments: Value) -> Value {
        self.request(id, "tools/call", json!({"name":tool,"arguments":arguments}));
        self.response(id)
    }
    fn finish(mut self) -> String {
        self.input.take();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "server failed at EOF: {status}");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "server did not exit at stdin EOF"
            );
            thread::sleep(Duration::from_millis(10));
        }
        self.stderr.take().unwrap().join().unwrap()
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        self.input.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}
fn structured(response: &Value) -> &Value {
    assert!(response.get("error").is_none(), "{response}");
    assert_eq!(response["result"]["isError"], false, "{response}");
    let value = &response["result"]["structuredContent"];
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["profile"], "light");
    assert!(value["index_generation"].as_i64().unwrap() > 0);
    value
}
fn tool_code(response: &Value) -> &str {
    assert_eq!(response["result"]["isError"], true, "{response}");
    response["result"]["structuredContent"]["error"]["code"]
        .as_str()
        .expect("typed tool error")
}

#[test]
fn initialize_discovery_and_actual_retrieval_are_read_only() {
    let fixture = Fixture::small();
    let before = std::fs::read(&fixture.config.database).unwrap();
    let source = std::fs::read(&fixture.source).unwrap();
    let mut client = fixture.launch();
    client.initialize();
    client.request(2, "tools/list", json!({}));
    let response = client.response(2);
    let tools = response["result"]["tools"].as_array().unwrap();
    let mut names: Vec<_> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["context", "get", "search"]);
    for tool in tools {
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
        assert!(tool["outputSchema"].is_object());
        assert_eq!(tool["annotations"]["readOnlyHint"], true);
    }
    let response = client.call(
        3,
        "search",
        json!({"query":"retrieval","filters":{"root":"docs","status":"active"}}),
    );
    let search = structured(&response);
    let hit = &search["data"][0];
    assert_eq!(hit["path"], "guide.md");
    assert_eq!(hit["root_key"], "docs");
    assert!(
        hit["body"].as_str().unwrap().contains("retrieval")
            || hit["body"].as_str().unwrap().contains("Retrieval")
    );
    assert_eq!(hit["source_sha256"].as_str().unwrap().len(), 64);
    assert!(hit["line_start"].as_u64().unwrap() > 0);
    let response = client.call(4, "get", json!({"key":hit["chunk_key"]}));
    let get = structured(&response);
    assert_eq!(get["index_generation"], search["index_generation"]);
    assert_eq!(get["data"]["source_sha256"], hit["source_sha256"]);
    assert!(
        get["data"]["body"].as_str().unwrap().contains("retrieval")
            || get["data"]["body"].as_str().unwrap().contains("Retrieval")
    );
    let response = client.call(
        5,
        "context",
        json!({"query":"retrieval","budget_bytes":4096}),
    );
    let context = structured(&response);
    assert_eq!(context["data"]["budget_unit"], "bytes");
    assert!(!context["data"]["items"].as_array().unwrap().is_empty());
    assert!(context["data"]["used_bytes"].as_u64().unwrap() <= 4096);
    assert_eq!(client.finish(), "");
    assert_eq!(std::fs::read(&fixture.config.database).unwrap(), before);
    assert_eq!(std::fs::read(&fixture.source).unwrap(), source);
}

#[test]
fn rejects_unknown_malformed_and_oversized_arguments_without_escalation() {
    let mut limits = Limits::default();
    limits.max_query_bytes = 16;
    let fixture = Fixture::new("# Retrieval\nretrieval evidence\n", limits);
    let mut client = fixture.launch();
    client.initialize();
    let cases = [
        ("search", json!({"query":"retrieval","profile":"full"})),
        ("search", json!({"query":"retrieval","root_path":"/"})),
        (
            "search",
            json!({"query":"retrieval","filters":{"trust":"trusted"}}),
        ),
        ("search", json!({"query":17})),
        ("search", json!({"query":"x".repeat(17)})),
        ("search", json!({"query":"retrieval","limit":101})),
        ("search", json!({"query":"retrieval","filters":[]})),
        ("get", json!({"key":"P-MCP","line_start":0})),
        ("context", json!({"query":"retrieval","budget_bytes":65537})),
        ("answer", json!({"query":"retrieval"})),
        ("index", json!({})),
    ];
    for (offset, (tool, arguments)) in cases.into_iter().enumerate() {
        let response = client.call(10 + offset as u64, tool, arguments);
        assert_eq!(response["error"]["code"], -32602, "{response}");
        assert_eq!(response["error"]["data"]["code"], "invalid_arguments");
    }
    // Exceed the SDK's configured line cap; it discards the line without
    // buffering forever, and the subsequent request must still work.
    client.request(
        50,
        "tools/call",
        json!({"name":"search","arguments":{"query":"x".repeat(20_000)}}),
    );
    client.request(51, "tools/list", json!({}));
    assert_eq!(
        client.response(51)["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(client.finish(), "");
}

#[test]
fn stale_sources_and_path_escape_are_typed_tool_errors() {
    let fixture = Fixture::small();
    let mut client = fixture.launch();
    client.initialize();
    let response = client.call(2, "get", json!({"key":"docs:../outside.md"}));
    assert_eq!(tool_code(&response), "inaccessible_path");
    std::fs::write(&fixture.source, "# Changed\nretrieval evidence changed\n").unwrap();
    let response = client.call(3, "get", json!({"key":"docs:guide.md"}));
    assert_eq!(tool_code(&response), "stale_source");
    let response = client.call(
        4,
        "context",
        json!({"query":"retrieval","budget_bytes":4096}),
    );
    assert_eq!(tool_code(&response), "stale_source");
    assert_eq!(client.finish(), "");
}

#[test]
fn unavailable_index_and_aggregate_response_cap_are_visible() {
    let mut limits = Limits::default();
    limits.max_response_bytes = 32768;
    limits.max_context_bytes = 8192;
    let text = format!(
        "# Retrieval\n{}",
        "retrieval evidence with bounded source lines and repeated searchable material.\n"
            .repeat(1000)
    );
    let fixture = Fixture::new(&text, limits);
    let mut client = fixture.launch();
    client.initialize();
    let response = client.call(2, "get", json!({"key":"docs:guide.md"}));
    assert_eq!(tool_code(&response), "response_too_large");
    assert!(
        serde_json::to_vec(&response).unwrap().len() <= fixture.config.limits.max_response_bytes
    );
    assert_eq!(client.finish(), "");
    std::fs::remove_file(&fixture.config.database).unwrap();
    let mut client = fixture.launch();
    client.initialize();
    let response = client.call(2, "search", json!({"query":"retrieval"}));
    assert_eq!(tool_code(&response), "index_unavailable");
    assert_eq!(client.finish(), "");
}

#[test]
fn cancellation_stops_publication_and_discovery_remains_responsive() {
    let fixture = Fixture::small();
    let mut client = fixture.launch();
    client.initialize();
    let lock = fixture.exclusive_lock();
    client.request(
        2,
        "tools/call",
        json!({"name":"search","arguments":{"query":"retrieval"}}),
    );
    thread::sleep(Duration::from_millis(100));
    client.send(json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":2,"reason":"test cancellation"}}));
    client.request(3, "tools/list", json!({}));
    assert_eq!(
        client.response(3)["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    lock.execute_batch("ROLLBACK;").unwrap();
    drop(lock);
    // Cancellation does not poison subsequent real retrieval operations.
    let response = client.call(4, "search", json!({"query":"retrieval"}));
    assert!(!structured(&response)["data"].as_array().unwrap().is_empty());
    // The SDK may suppress a cancelled request's response entirely. If one was
    // delivered before the subsequent successful request, it must not publish evidence.
    assert!(
        !client
            .buffered
            .iter()
            .any(|message| message["id"] == 2 && message["result"]["isError"] == false)
    );
    assert_eq!(client.finish(), "");
}

#[test]
fn wall_clock_timeout_and_eof_with_pending_work_are_bounded() {
    let mut limits = Limits::default();
    limits.timeout_ms = 100;
    let fixture = Fixture::new("# Retrieval\nretrieval evidence\n", limits);
    let mut client = fixture.launch();
    client.initialize();
    let lock = fixture.exclusive_lock();
    let started = Instant::now();
    let response = client.call(2, "search", json!({"query":"retrieval"}));
    assert_eq!(tool_code(&response), "timeout");
    assert!(started.elapsed() < Duration::from_secs(2));
    client.request(
        3,
        "tools/call",
        json!({"name":"search","arguments":{"query":"retrieval"}}),
    );
    assert_eq!(client.finish(), "");
    lock.execute_batch("ROLLBACK;").unwrap();
}

#[test]
fn concurrency_admission_bounds_blocking_reads_and_supports_cancellation() {
    let fixture = Fixture::small();
    let mut client = fixture.launch();
    client.initialize();
    let lock = fixture.exclusive_lock();
    for id in 10..14 {
        client.request(
            id,
            "tools/call",
            json!({"name":"search","arguments":{"query":"retrieval"}}),
        );
    }
    thread::sleep(Duration::from_millis(100));
    let response = client.call(20, "search", json!({"query":"retrieval"}));
    assert_eq!(tool_code(&response), "server_busy");
    for id in 10..14 {
        client.send(json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":id,"reason":"release waiter"}}));
    }
    // Busy SQLite readers may remain inside a blocking busy handler even after
    // interrupt. Their permits must not be reused until their workers return.
    // No hard cancellation claim is made; release the real lock before exit.
    lock.execute_batch("ROLLBACK;").unwrap();
    drop(lock);
    client.request(21, "tools/list", json!({}));
    assert_eq!(
        client.response(21)["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(!client.buffered.iter().any(|message| {
        message["id"]
            .as_u64()
            .is_some_and(|id| (10..14).contains(&id))
            && message["result"]["isError"] == false
    }));
    assert_eq!(client.finish(), "");
}

#[test]
fn current_protocol_uses_sdk_discovery_and_per_request_metadata() {
    let fixture = Fixture::small();
    let mut client = fixture.launch();
    let meta = json!({
        "io.modelcontextprotocol/protocolVersion":"2026-07-28",
        "io.modelcontextprotocol/clientInfo":{"name":"scribe-subprocess-test","version":"1"},
        "io.modelcontextprotocol/clientCapabilities":{}
    });
    client.request(1, "server/discover", json!({"_meta":meta}));
    let response = client.response(1);
    assert!(response.get("error").is_none(), "{response}");
    assert!(
        response["result"]["supportedVersions"]
            .as_array()
            .unwrap()
            .contains(&json!("2026-07-28"))
    );
    client.request(2, "tools/list", json!({"_meta":meta}));
    assert_eq!(
        client.response(2)["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    client.request(
        3,
        "tools/call",
        json!({"_meta":meta,"name":"search","arguments":{"query":"retrieval"}}),
    );
    let response = client.response(3);
    assert!(!structured(&response)["data"].as_array().unwrap().is_empty());
    assert_eq!(client.finish(), "");
}
