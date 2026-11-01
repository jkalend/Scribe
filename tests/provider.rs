#![cfg(feature = "full")]

mod support;

use kb::embed::{LocalModel, ModelPurpose, decode_vector, encode_vector, validate_vector};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use support::{OllamaServer, Reply, Request, standard_reply};

fn error<T>(result: kb::error::Result<T>) -> String {
    result.err().expect("operation must fail").code
}
fn generated(request: &Request, response: &str) -> Value {
    json!({"model":"fixture:latest","response":response,"done":true,"done_reason":"stop",
        "prompt_eval_count":request.body["prompt"].as_str().unwrap().split_whitespace().count(),"eval_count":1})
}

#[test]
fn exact_protocol_prefixes_counts_normalization_and_contract() {
    let server = OllamaServer::start(|request| {
        if request.path == "/api/embed" {
            let mut reply: Value = serde_json::from_slice(&standard_reply(request).body).unwrap();
            for vector in reply["embeddings"].as_array_mut().unwrap() {
                *vector = json!([3.0, 4.0, 0.0]);
            }
            Reply::json(reply)
        } else {
            standard_reply(request)
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let mut config = server.model_config(dir.path());
    config.provider_reported_tag_digest = Some("a".repeat(64));
    config.query_prefix = "query: ".into();
    config.document_prefix = "passage: ".into();
    let model = LocalModel::prepare(&config, ModelPurpose::Embedding).unwrap();
    let second = LocalModel::prepare(&config, ModelPurpose::Embedding).unwrap();
    assert_eq!(model.contract, second.contract);
    assert_eq!(model.contract.id.len(), 64);
    assert_eq!(
        model.contract.details["artifact_verification_status"],
        "provider_reported_only"
    );
    assert!(model.contract.details["independent_artifact_sha256"].is_null());
    assert_eq!(
        model.contract.details["transforms"]["query"]["prefix"],
        "query: "
    );
    assert_eq!(model.count_tokens("one two three").unwrap(), 3);
    let vector = model.embed_query("one two").unwrap();
    assert!((vector[0] - 0.6).abs() < 1e-6 && (vector[1] - 0.8).abs() < 1e-6);
    assert_eq!(
        decode_vector(&encode_vector(&vector).unwrap(), 3).unwrap(),
        vector
    );
    assert_eq!(
        model.embed_documents(&["one", "two three"]).unwrap().len(),
        2
    );
    let requests = server.requests();
    let embeds: Vec<_> = requests.iter().filter(|r| r.path == "/api/embed").collect();
    assert_eq!(embeds[0].body["input"], json!(["query: one two"]));
    assert_eq!(
        embeds[1].body["input"],
        json!(["passage: one", "passage: two three"])
    );
    for request in embeds {
        assert_eq!(request.method, "POST");
        assert_eq!(request.body["truncate"], false);
        assert_eq!(request.body["dimensions"], 3);
        assert_eq!(request.body["options"]["num_ctx"], config.context_window);
    }
    config.query_prefix.push('!');
    assert_ne!(
        model.contract.id,
        LocalModel::prepare(&config, ModelPurpose::Embedding)
            .unwrap()
            .contract
            .id
    );
}

#[test]
fn endpoint_policy_rejects_nonliteral_nonlocal_and_ambiguous_urls_before_network() {
    let server = OllamaServer::start(standard_reply);
    let dir = tempfile::tempdir().unwrap();
    let original = server.model_config(dir.path());
    for endpoint in [
        "http://localhost:11434",
        "http://example.com:11434",
        "https://ollama.com:443",
        "http://127.0.0.2:11434",
        "http://127.1:11434",
        "http://2130706433:11434",
        "http://127.0.0.1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://user@127.0.0.1:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434/?secret=x",
        "http://127.0.0.1:11434/#fragment",
        "file://127.0.0.1:11434",
        "http://[::2]:11434",
        "http://127.0.0.1:11434\\evil",
    ] {
        let mut config = original.clone();
        config.base_url = endpoint.into();
        assert_eq!(
            error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
            "provider_endpoint_rejected",
            "{endpoint}"
        );
    }
    assert!(server.requests().is_empty());
}

#[test]
fn redirects_never_reach_destination_and_no_implicit_retries() {
    let destination = OllamaServer::start(standard_reply);
    let target = destination.url.clone();
    let source = OllamaServer::start(move |_| {
        let mut reply = Reply::raw(307, Vec::new());
        reply
            .headers
            .push(("Location".into(), format!("{target}/api/version")));
        reply
    });
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        error(LocalModel::prepare(
            &source.model_config(dir.path()),
            ModelPurpose::Embedding
        )),
        "provider_redirect_rejected"
    );
    assert!(destination.requests().is_empty());
    assert_eq!(source.requests().len(), 1);
    let failures =
        OllamaServer::start(|_| Reply::raw(503, b"private prompt must not escape".to_vec()));
    let failure = LocalModel::prepare(&failures.model_config(dir.path()), ModelPurpose::Embedding)
        .err()
        .unwrap();
    assert_eq!(failure.code, "provider_http_error");
    assert!(!failure.message.contains("private"));
    assert_eq!(failures.requests().len(), 1);
}

#[test]
fn configured_digest_and_independent_actual_file_hash_are_distinct() {
    let server = OllamaServer::start(standard_reply);
    let dir = tempfile::tempdir().unwrap();
    let mut config = server.model_config(dir.path());
    config.provider_reported_tag_digest = Some("b".repeat(64));
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "model_identity_mismatch"
    );
    config.provider_reported_tag_digest = Some("sha256:".to_owned() + &"a".repeat(64));
    config.require_independent_artifact_sha256 = true;
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "artifact_verification_required"
    );
    let path = dir.path().join("independent.bin");
    std::fs::write(&path, b"independently hashed synthetic bytes").unwrap();
    config.independent_artifact_path = Some(path.clone());
    config.independent_artifact_sha256 = Some("a".repeat(64));
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "artifact_identity_mismatch"
    );
    let actual = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    config.independent_artifact_sha256 = Some(actual.clone());
    let model = LocalModel::prepare(&config, ModelPurpose::Embedding).unwrap();
    assert_eq!(
        model.contract.details["independent_artifact_sha256"],
        actual
    );
    assert_ne!(
        model.contract.details["independent_artifact_sha256"],
        model.contract.details["provider_reported_tag_digest"]
    );
    assert_eq!(
        model.contract.details["artifact_verification_status"],
        "independent_file_sha256_verified_runtime_binding_unattested"
    );
}

#[test]
fn tokenizer_hash_is_required_and_file_truncation_cannot_hide_overflow() {
    let server = OllamaServer::start(standard_reply);
    let dir = tempfile::tempdir().unwrap();
    let mut config = server.model_config(dir.path());
    let original = config.tokenizer_sha256.clone();
    config.tokenizer_sha256 = None;
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "tokenizer_unavailable"
    );
    config.tokenizer_sha256 = Some("b".repeat(64));
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "tokenizer_mismatch"
    );
    config.tokenizer_sha256 = original;
    let path = config.tokenizer_path.as_ref().unwrap();
    let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    tokenizer["truncation"] =
        json!({"direction":"Right","max_length":1,"strategy":"LongestFirst","stride":0});
    tokenizer["padding"] = json!({"strategy":{"Fixed":20},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[UNK]"});
    let bytes = serde_json::to_vec(&tokenizer).unwrap();
    std::fs::write(path, &bytes).unwrap();
    config.tokenizer_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
    config.max_input_tokens = 2;
    config.document_prefix = "prefix ".into();
    let model = LocalModel::prepare(&config, ModelPurpose::Embedding).unwrap();
    assert_eq!(model.count_tokens("one two three").unwrap(), 3);
    assert_eq!(
        error(model.embed_documents(&["one two"])),
        "token_limit_exceeded"
    );
    assert_eq!(model.embed_documents(&["one"]).unwrap().len(), 1);
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(|r| r.path == "/api/embed")
            .count(),
        1
    );
}

#[test]
fn invalid_embedding_shapes_values_counts_and_runtime_truncation_fail() {
    for (embeddings, token_count, expected) in [
        (json!([]), 1, "provider_invalid_response"),
        (json!([[1, 0]]), 1, "vector_invalid"),
        (json!([[0, 0, 0]]), 1, "vector_invalid"),
        (json!([[1e100, 0, 0]]), 1, "vector_invalid"),
        (json!([[null, 0, 0]]), 1, "vector_invalid"),
        (
            json!([[1, 0, 0], [1, 0, 0]]),
            1,
            "provider_invalid_response",
        ),
        (json!([[1, 0, 0]]), 0, "tokenizer_mismatch"),
    ] {
        let server = OllamaServer::start(move |request| {
            if request.path == "/api/embed" {
                Reply::json(
                    json!({"model":"fixture:latest","embeddings":embeddings,"prompt_eval_count":token_count}),
                )
            } else {
                standard_reply(request)
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let model =
            LocalModel::prepare(&server.model_config(dir.path()), ModelPurpose::Embedding).unwrap();
        assert_eq!(error(model.embed_query("one")), expected);
    }
}

#[test]
fn vector_codec_checks_length_endianness_finite_nonzero_and_normalization() {
    assert_eq!(
        encode_vector(&[1.0, 0.0]).unwrap(),
        vec![0, 0, 128, 63, 0, 0, 0, 0]
    );
    for vector in [
        vec![],
        vec![0.0],
        vec![2.0],
        vec![f32::NAN],
        vec![f32::INFINITY],
    ] {
        assert_eq!(error(encode_vector(&vector)), "vector_invalid");
        assert_eq!(error(validate_vector(&vector)), "vector_invalid");
    }
    for (blob, dimensions) in [
        (vec![], 0),
        (vec![0; 4], 1),
        (vec![0; 3], 1),
        (vec![0; 8], 1),
        (vec![], usize::MAX),
    ] {
        assert_eq!(error(decode_vector(&blob, dimensions)), "vector_invalid");
    }
    for value in [f32::NAN, f32::INFINITY, 2.0] {
        assert_eq!(
            error(decode_vector(&value.to_le_bytes(), 1)),
            "vector_invalid"
        );
    }
}

#[test]
fn generator_sends_pinned_raw_structured_protocol_with_exact_reservation() {
    let server = OllamaServer::start(|request| {
        if request.path == "/api/generate" {
            Reply::json(generated(request, "{\"ok\":true}"))
        } else {
            standard_reply(request)
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let mut config = server.model_config(dir.path());
    config.prompt_template = "plain-v1".into();
    config.max_output_tokens = 4;
    config.context_window = 16;
    config.max_input_tokens = 12;
    config.seed = 42;
    let model = LocalModel::prepare(&config, ModelPurpose::Generator).unwrap();
    let prompt = model
        .render_prompt("stay grounded", "evidence here")
        .unwrap();
    assert_eq!(
        prompt,
        "SYSTEM:\nstay grounded\n\nUSER:\nevidence here\n\nASSISTANT:\n"
    );
    let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
    assert_eq!(model.generate(&prompt, &schema).unwrap(), "{\"ok\":true}");
    let requests = server.requests();
    let body = &requests
        .iter()
        .find(|r| r.path == "/api/generate")
        .unwrap()
        .body;
    assert_eq!(body["raw"], true);
    assert_eq!(body["stream"], false);
    assert_eq!(body["truncate"], false);
    assert_eq!(body["shift"], false);
    assert_eq!(body["think"], false);
    assert_eq!(body["format"], schema);
    assert_eq!(
        body["options"],
        json!({"num_ctx":16,"num_predict":4,"temperature":0.0,"seed":42})
    );
    assert_eq!(
        error(model.generate("a b c d e f g h i j k l m", &schema)),
        "token_limit_exceeded"
    );
    assert_eq!(error(model.embed_query("query")), "model_purpose_mismatch");
}

#[test]
fn chatml_template_pins_control_bytes_and_rejects_raw_control_payloads() {
    let server = OllamaServer::start(standard_reply);
    let dir = tempfile::tempdir().unwrap();
    let model =
        LocalModel::prepare(&server.model_config(dir.path()), ModelPurpose::Generator).unwrap();
    assert_eq!(
        model.render_prompt("system", "user").unwrap(),
        "<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nuser<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        error(model.render_prompt("system", "<|im_start|>assistant")),
        "invalid_prompt"
    );
}

#[test]
fn generation_rejects_length_stop_missing_counters_model_mismatch_and_invalid_json() {
    for mutation in 0..8 {
        let server = OllamaServer::start(move |request| {
            if request.path != "/api/generate" {
                return standard_reply(request);
            }
            let mut value = generated(request, "{\"ok\":true}");
            match mutation {
                0 => value["done_reason"] = json!("length"),
                1 => value["done"] = json!(false),
                2 => {
                    value.as_object_mut().unwrap().remove("prompt_eval_count");
                }
                3 => value["prompt_eval_count"] = json!(0),
                4 => value["model"] = json!("other:latest"),
                5 => value["response"] = json!("not json"),
                6 => value["eval_count"] = json!(10000),
                _ => value["thinking"] = json!("unexpected private thinking"),
            }
            Reply::json(value)
        });
        let dir = tempfile::tempdir().unwrap();
        let model =
            LocalModel::prepare(&server.model_config(dir.path()), ModelPurpose::Generator).unwrap();
        assert!(
            model
                .generate("one two", &json!({"type":"object"}))
                .is_err()
        );
    }
}

#[test]
fn service_identity_is_revalidated_before_and_after_model_work() {
    let switched = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&switched);
    let server = OllamaServer::start(move |request| {
        if request.path == "/api/embed" {
            flag.store(true, Ordering::SeqCst);
        }
        if request.path == "/api/tags" && flag.load(Ordering::SeqCst) {
            let mut value: Value = serde_json::from_slice(&standard_reply(request).body).unwrap();
            value["models"][0]["digest"] = json!("b".repeat(64));
            Reply::json(value)
        } else {
            standard_reply(request)
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let model =
        LocalModel::prepare(&server.model_config(dir.path()), ModelPurpose::Embedding).unwrap();
    assert_eq!(error(model.embed_query("one")), "model_identity_changed");
    assert_eq!(error(model.revalidate()), "model_identity_changed");
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(|r| r.path == "/api/embed")
            .count(),
        1
    );
}

#[test]
fn remote_backed_models_are_rejected_even_at_loopback_endpoint() {
    for field in ["remote_model", "remote_host"] {
        let server = OllamaServer::start(move |request| {
            if request.path == "/api/tags" {
                let mut value: Value =
                    serde_json::from_slice(&standard_reply(request).body).unwrap();
                value["models"][0][field] = json!("cloud");
                Reply::json(value)
            } else {
                standard_reply(request)
            }
        });
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            error(LocalModel::prepare(
                &server.model_config(dir.path()),
                ModelPurpose::Embedding
            )),
            "provider_endpoint_rejected"
        );
    }
}

#[test]
fn malformed_oversized_http_errors_and_timeouts_are_bounded() {
    for reply in 0..3 {
        let server = OllamaServer::start(move |_| match reply {
            0 => Reply::raw(200, b"not-json".to_vec()),
            1 => Reply::raw(200, vec![b'x'; 1025]),
            _ => {
                std::thread::sleep(std::time::Duration::from_millis(150));
                Reply::json(json!({"version":"fixture-1"}))
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let mut config = server.model_config(dir.path());
        config.max_response_bytes = 1024;
        if reply == 2 {
            config.timeout_ms = 30;
        }
        let result = error(LocalModel::prepare(&config, ModelPurpose::Embedding));
        assert_eq!(
            result,
            match reply {
                0 => "provider_invalid_response",
                1 => "provider_response_too_large",
                _ => "provider_timeout",
            }
        );
    }
}

#[test]
fn missing_models_fail_without_pull_or_start_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let server = OllamaServer::start(move |request| {
        seen.fetch_add(1, Ordering::SeqCst);
        if request.path == "/api/tags" {
            Reply::json(json!({"models":[]}))
        } else {
            standard_reply(request)
        }
    });
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        error(LocalModel::prepare(
            &server.model_config(dir.path()),
            ModelPurpose::Embedding
        )),
        "model_unavailable"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(server.requests().iter().all(|r| r.method == "GET"));
}

#[test]
fn ambient_proxies_are_ignored_in_isolated_child() {
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "proxy_child", "--ignored"])
        .env("SCRIBE_PROXY_TEST_CHILD", "1")
        .env("HTTP_PROXY", "http://127.0.0.1:1")
        .env("http_proxy", "http://127.0.0.1:1")
        .env("HTTPS_PROXY", "http://127.0.0.1:1")
        .env("https_proxy", "http://127.0.0.1:1")
        .env("ALL_PROXY", "http://127.0.0.1:1")
        .env("all_proxy", "http://127.0.0.1:1")
        .env("NO_PROXY", "")
        .env("no_proxy", "")
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
#[ignore = "invoked by ambient_proxies_are_ignored_in_isolated_child with isolated environment"]
fn proxy_child() {
    assert_eq!(std::env::var("SCRIBE_PROXY_TEST_CHILD").unwrap(), "1");
    let server = OllamaServer::start(standard_reply);
    let dir = tempfile::tempdir().unwrap();
    let model =
        LocalModel::prepare(&server.model_config(dir.path()), ModelPurpose::Embedding).unwrap();
    assert_eq!(model.embed_query("one").unwrap(), vec![1.0, 0.0, 0.0]);
}

#[test]
fn litellm_protocol_and_safety_regressions_in_isolated_child() {
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "litellm_child", "--ignored"])
        .env("SCRIBE_LITELLM_TEST_KEY", "synthetic-not-a-real-key")
        .env_remove("SCRIBE_LITELLM_MISSING_KEY")
        .env("HTTP_PROXY", "http://127.0.0.1:1")
        .env("http_proxy", "http://127.0.0.1:1")
        .env("HTTPS_PROXY", "http://127.0.0.1:1")
        .env("https_proxy", "http://127.0.0.1:1")
        .env("ALL_PROXY", "http://127.0.0.1:1")
        .env("all_proxy", "http://127.0.0.1:1")
        .env("NO_PROXY", "")
        .env("no_proxy", "")
        .status()
        .unwrap();
    assert!(status.success());
}

fn remote_reply(request: &Request) -> Reply {
    match request.path.as_str() {
        "/v1/models" => Reply::json(json!({"data":[{"id":"remote-alias","owned_by":"synthetic"}]})),
        "/v1/embeddings" => {
            let count = request.body["input"].as_array().unwrap().len();
            Reply::json(
                json!({"model":"observable-embedding-backend","data":(0..count).rev()
                .map(|index|json!({"index":index,"embedding":if index==0{vec![3.0,4.0,0.0]}else{vec![0.0,0.0,1.0]}})).collect::<Vec<_>>(),
                "usage":{"prompt_tokens":200,"total_tokens":200}}),
            )
        }
        "/v1/chat/completions" => Reply::json(json!({"model":"observable-generator-backend",
            "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"ok\":true}"}}],
            "usage":{"prompt_tokens":200,"completion_tokens":3,"total_tokens":203}})),
        _ => Reply::raw(404, Vec::new()),
    }
}

fn remote_config(
    server: &OllamaServer,
    directory: &std::path::Path,
) -> kb::full_config::ModelConfig {
    let mut config = server.model_config(directory);
    config.provider = "litellm".into();
    config.base_url = format!("{}/v1", server.url);
    config.model = "remote-alias".into();
    config.api_key_env = Some("SCRIBE_LITELLM_TEST_KEY".into());
    config.tokenizer_path = None;
    config.tokenizer_sha256 = None;
    config.prompt_template = "openai-chat-v1".into();
    config.max_input_bytes = 4096;
    config.max_output_tokens = 32;
    config
}

#[test]
#[ignore = "invoked by litellm_protocol_and_safety_regressions_in_isolated_child with synthetic auth env"]
fn litellm_child() {
    let server = OllamaServer::start(remote_reply);
    let dir = tempfile::tempdir().unwrap();
    let mut config = remote_config(&server, dir.path());
    config.query_prefix = "query: ".into();
    config.document_prefix = "passage: ".into();
    let embeddings = LocalModel::prepare(&config, ModelPurpose::Embedding).unwrap();
    assert!(!embeddings.has_exact_tokenizer());
    assert_eq!(
        error(embeddings.count_tokens("one two")),
        "tokenizer_unavailable"
    );
    assert_eq!(
        embeddings.contract.details["artifact_verification_status"],
        "provider_identity_only"
    );
    assert!(embeddings.contract.details["runtime"].is_null());
    assert!(embeddings.contract.details["independent_artifact_sha256"].is_null());
    let contract_before = embeddings.contract.clone();
    let vectors = embeddings.embed_documents(&["one", "two"]).unwrap();
    assert!((vectors[0][0] - 0.6).abs() < 1e-6);
    assert_eq!(vectors[1], vec![0.0, 0.0, 1.0]);
    assert_eq!(embeddings.contract, contract_before);
    assert_eq!(
        embeddings.observations()["response_model_identity"],
        "observable-embedding-backend"
    );
    assert_eq!(embeddings.observations()["usage"]["prompt_tokens"], 200);
    assert_eq!(embeddings.embed_query("three").unwrap().len(), 3);
    assert_eq!(
        embeddings.observations()["aggregate_usage"]["prompt_tokens"],
        400
    );
    let generator = LocalModel::prepare(&config, ModelPurpose::Generator).unwrap();
    let prompt = generator
        .render_prompt("system instruction", "untrusted evidence")
        .unwrap();
    let messages: Value = serde_json::from_str(&prompt).unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    generator.validate_prompt_budget(&prompt).unwrap();
    let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
    assert_eq!(
        generator.generate(&prompt, &schema).unwrap(),
        "{\"ok\":true}"
    );
    assert_eq!(
        generator.observations()["response_model_identity"],
        "observable-generator-backend"
    );
    assert_eq!(generator.observations()["usage"]["completion_tokens"], 3);
    assert_eq!(
        generator.observations()["upstream_chat_framing_counted"],
        false
    );
    assert_eq!(
        error(generator.validate_prompt_budget(&"x".repeat(4097))),
        "prompt_too_large"
    );
    assert_eq!(
        error(embeddings.embed_query(&"x".repeat(4097))),
        "provider_request_too_large"
    );
    let requests = server.requests();
    for request in &requests {
        assert!(
            request
                .headers
                .iter()
                .any(|(name, value)| name == "authorization"
                    && value == "Bearer synthetic-not-a-real-key")
        );
    }
    let chat = requests
        .iter()
        .find(|request| request.path == "/v1/chat/completions")
        .unwrap();
    assert_eq!(chat.body["messages"], messages);
    assert_eq!(chat.body["stream"], false);
    assert_eq!(
        chat.body["response_format"],
        json!({"type":"json_schema","json_schema":{"name":"scribe_answer","schema":schema,"strict":true}})
    );
    assert_eq!(chat.body["max_tokens"], 32);
    assert!(chat.body.get("tools").is_none());
    let embed = requests
        .iter()
        .find(|request| request.path == "/v1/embeddings")
        .unwrap();
    assert_eq!(embed.body["input"], json!(["passage: one", "passage: two"]));
    assert_eq!(embed.body["encoding_format"], "float");
    let serialized = serde_json::to_string(&embeddings.contract).unwrap();
    assert!(!serialized.contains("synthetic-not-a-real-key"));

    config.api_key_env = Some("SCRIBE_LITELLM_MISSING_KEY".into());
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "provider_auth_unavailable"
    );
    config.api_key_env = Some("SCRIBE_LITELLM_TEST_KEY".into());
    config.require_independent_artifact_sha256 = true;
    assert_eq!(
        error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
        "artifact_verification_required"
    );
    config.require_independent_artifact_sha256 = false;
    for endpoint in [
        "http://example.com/v1",
        "http://localhost:4000/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/v1#fragment",
    ] {
        config.base_url = endpoint.into();
        assert_eq!(
            error(LocalModel::prepare(&config, ModelPurpose::Embedding)),
            "provider_endpoint_rejected"
        );
    }

    for mutation in 0..9 {
        let broken = OllamaServer::start(move |request| {
            let mut value: Value = serde_json::from_slice(&remote_reply(request).body).unwrap();
            if request.path == "/v1/embeddings" {
                match mutation {
                    0 => value["data"][0]["index"] = json!(10),
                    1 => value["data"][0]["embedding"] = json!([0, 0, 0]),
                    2 => value["data"][0]["embedding"] = json!([1e100, 0, 0]),
                    3 => value["data"][0]["embedding"] = json!([1, 0]),
                    4 => value["model"] = Value::Null,
                    5 => value["usage"]["total_tokens"] = json!(0),
                    6 => {
                        value.as_object_mut().unwrap().remove("usage");
                    }
                    7 => value["data"] = json!([]),
                    _ => {
                        value["data"][1]["index"] = value["data"][0]["index"].clone();
                    }
                }
            }
            Reply::json(value)
        });
        let model =
            LocalModel::prepare(&remote_config(&broken, dir.path()), ModelPurpose::Embedding)
                .unwrap();
        assert!(model.embed_documents(&["one", "two"]).is_err());
    }
    for mutation in 0..8 {
        let broken = OllamaServer::start(move |request| {
            let mut value: Value = serde_json::from_slice(&remote_reply(request).body).unwrap();
            if request.path == "/v1/chat/completions" {
                match mutation {
                    0 => value["choices"][0]["finish_reason"] = json!("length"),
                    1 => value["choices"][0]["message"]["content"] = json!("not JSON"),
                    2 => value["usage"]["completion_tokens"] = json!(10000),
                    3 => {
                        value["choices"][0]["message"]["tool_calls"] =
                            json!([{"function":{"name":"unsafe"}}])
                    }
                    4 => value["choices"][0]["message"]["refusal"] = json!("refused"),
                    5 => value["choices"][0]["message"]["role"] = json!("tool"),
                    6 => {
                        value.as_object_mut().unwrap().remove("usage");
                    }
                    _ => value["choices"] = json!([]),
                }
            }
            Reply::json(value)
        });
        let model =
            LocalModel::prepare(&remote_config(&broken, dir.path()), ModelPurpose::Generator)
                .unwrap();
        assert!(
            model
                .generate(&model.render_prompt("system", "question").unwrap(), &schema)
                .is_err()
        );
    }
    let changed = Arc::new(AtomicBool::new(false));
    let state = Arc::clone(&changed);
    let drift = OllamaServer::start(move |request| {
        let mut value: Value = serde_json::from_slice(&remote_reply(request).body).unwrap();
        if request.path == "/v1/embeddings" && state.swap(true, Ordering::SeqCst) {
            value["model"] = json!("different-backend");
        }
        Reply::json(value)
    });
    let model =
        LocalModel::prepare(&remote_config(&drift, dir.path()), ModelPurpose::Embedding).unwrap();
    model.embed_query("first").unwrap();
    assert_eq!(error(model.embed_query("second")), "model_identity_changed");
}
