#![allow(dead_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Value,
    pub headers: Vec<(String, String)>,
}

pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl Reply {
    pub fn json(value: Value) -> Self {
        let mut reply = Self::raw(200, serde_json::to_vec(&value).unwrap());
        reply
            .headers
            .push(("Content-Type".into(), "application/json".into()));
        reply
    }
    pub fn raw(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }
}

pub struct OllamaServer {
    pub url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl OllamaServer {
    pub fn start(handler: impl Fn(&Request) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&requests);
        let stopped = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // Windows inherits the listener's nonblocking mode. This
                        // request parser requires blocking reads on accepted sockets.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        if let Some(request) = read_request(&mut stream) {
                            seen.lock().unwrap().push(request.clone());
                            let reply = handler(&request);
                            let mut headers = format!(
                                "HTTP/1.1 {} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n",
                                reply.status,
                                reply.body.len()
                            );
                            for (key, value) in reply.headers {
                                headers.push_str(&format!("{key}: {value}\r\n"));
                            }
                            headers.push_str("\r\n");
                            let _ = stream.write_all(headers.as_bytes());
                            let _ = stream.write_all(&reply.body);
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            thread: Some(thread),
        }
    }
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
    pub fn model_config(&self, dir: &Path) -> kb::full_config::ModelConfig {
        let tokenizer = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        let path = dir.join("fixture-tokenizer.json");
        std::fs::write(&path, tokenizer).unwrap();
        kb::full_config::ModelConfig {
            base_url: self.url.clone(),
            model: "fixture:latest".into(),
            license: "synthetic-test-only".into(),
            tokenizer_path: Some(path),
            tokenizer_sha256: Some(format!("{:x}", Sha256::digest(tokenizer))),
            dimensions: 3,
            max_input_tokens: 4096,
            context_window: 8192,
            max_output_tokens: 1024,
            timeout_ms: 3000,
            max_response_bytes: 1024 * 1024,
            ..Default::default()
        }
    }
}
impl Drop for OllamaServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0; 4096];
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break index + 4;
        }
        if bytes.len() > 64 * 1024 {
            return None;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end]).ok()?;
    let mut first = header.lines().next()?.split_whitespace();
    let method = first.next()?.to_owned();
    let path = first.next()?.to_owned();
    let length = header
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let headers = header
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    if length > 8 * 1024 * 1024 {
        return None;
    }
    while bytes.len() - header_end < length {
        let mut chunk = [0; 4096];
        let count = stream.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let body = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[header_end..header_end + length]).ok()?
    };
    Some(Request {
        method,
        path,
        body,
        headers,
    })
}

pub fn standard_reply(request: &Request) -> Reply {
    match request.path.as_str() {
        "/api/version" => Reply::json(json!({"version":"fixture-1"})),
        "/api/tags" => Reply::json(
            json!({"models":[{"name":"fixture:latest","model":"fixture:latest","digest":"a".repeat(64),"size":3,"details":{"format":"synthetic","quantization_level":"F32"}}]}),
        ),
        "/api/embed" => {
            let inputs = request.body["input"].as_array().unwrap();
            let count: usize = inputs
                .iter()
                .map(|text| text.as_str().unwrap().split_whitespace().count())
                .sum();
            Reply::json(
                json!({"model":"fixture:latest","embeddings":inputs.iter().map(|_| vec![1.0,0.0,0.0]).collect::<Vec<_>>(),"prompt_eval_count":count}),
            )
        }
        "/api/generate" => {
            let text = r#"{"abstained":true,"abstain_reason":"insufficient_support","claims":[],"conflicts":[]}"#;
            Reply::json(
                json!({"model":"fixture:latest","response":text,"done":true,"done_reason":"stop","prompt_eval_count":request.body["prompt"].as_str().unwrap().split_whitespace().count(),"eval_count":1}),
            )
        }
        _ => Reply::raw(404, b"not found".to_vec()),
    }
}
