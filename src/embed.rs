//! Opt-in bounded Ollama and explicitly authorized LiteLLM adapters. Never downloads models.
//! Blocking requests cannot be forcibly cancelled; cancellation is checked between reads/requests.
//! Ollama: https://docs.ollama.com/api/{embed,generate,tags}; LiteLLM:
//! https://docs.litellm.ai/docs/{proxy/user_keys,embedding/supported_embedding,completion/json_mode}.
//! reqwest 0.12.24, tokenizers 0.23. Ollama tokenizer bytes must match runtime tokenization/BOS.

#![cfg(feature = "full")]

use crate::error::{Error, Result};
use crate::full_config::ModelConfig;
use reqwest::blocking::Client;
use reqwest::{Method, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_DIMENSIONS: usize = 65_536;
const MAX_TOKENS: usize = 1_048_576;
const NORM_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelPurpose {
    Embedding,
    Generator,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelContract {
    pub id: String,
    pub details: Value,
}

pub struct LocalModel {
    pub contract: ModelContract,
    settings: ModelConfig,
    purpose: ModelPurpose,
    client: Client,
    base: Url,
    tokenizer: Option<Tokenizer>,
    identity: Value,
    observations: std::sync::Mutex<Value>,
}

impl LocalModel {
    pub fn prepare(settings: &ModelConfig, purpose: ModelPurpose) -> Result<Self> {
        crate::mcp::checkpoint()?;
        validate_settings(settings, purpose)?;
        if settings.provider == "litellm" {
            return Self::prepare_litellm(settings, purpose);
        }
        let base = local_endpoint(&settings.base_url)?;
        let deadline = deadline(settings.timeout_ms);
        let path = settings.tokenizer_path.as_ref().ok_or_else(|| {
            Error::new(
                "tokenizer_unavailable",
                "a local pinned tokenizer_path is required",
            )
        })?;
        let expected = settings
            .tokenizer_sha256
            .as_deref()
            .ok_or_else(|| Error::new("tokenizer_unavailable", "tokenizer_sha256 is required"))?;
        validate_hash(expected, "tokenizer_sha256")?;
        let bytes = read_bounded_file(path, MAX_TOKENIZER_BYTES, "tokenizer_unavailable")?;
        let tokenizer_hash = sha256(&bytes);
        if !tokenizer_hash.eq_ignore_ascii_case(expected) {
            return Err(Error::new(
                "tokenizer_mismatch",
                "tokenizer bytes do not match the pinned SHA-256",
            ));
        }
        let mut tokenizer = Tokenizer::from_bytes(&bytes)
            .map_err(|_| Error::new("tokenizer_invalid", "cannot parse the pinned tokenizer"))?;
        // File-supplied padding/truncation must never hide an over-budget input.
        tokenizer
            .with_truncation(None)
            .map_err(|_| Error::new("tokenizer_invalid", "cannot disable tokenizer truncation"))?;
        tokenizer.with_padding(None);
        let artifact_hash = independent_artifact(settings, deadline)?;
        let timeout = remaining(deadline)?;
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .referer(false)
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .connect_timeout(timeout.min(Duration::from_secs(5)))
            .timeout(timeout)
            .build()
            .map_err(|_| {
                Error::new("provider_unavailable", "cannot construct local HTTP client")
            })?;
        let mut adapter = Self {
            contract: ModelContract {
                id: String::new(),
                details: Value::Null,
            },
            settings: settings.clone(),
            purpose,
            client,
            base,
            tokenizer: Some(tokenizer),
            identity: Value::Null,
            observations: std::sync::Mutex::new(Value::Null),
        };
        adapter.identity = adapter.read_identity(deadline)?;
        let details = json!({
            "contract_version": 1,
            "provider": "ollama", "endpoint": adapter.base.as_str(),
            "purpose": match purpose { ModelPurpose::Embedding => "embedding", ModelPurpose::Generator => "generator" },
            "runtime": adapter.identity,
            "model": settings.model,
            "provider_reported_tag_digest": adapter.identity["model"]["digest"],
            "provider_digest_expected": settings.provider_reported_tag_digest.as_ref().map(|value| normalized_digest(value)),
            "provider_digest_verification": if settings.provider_reported_tag_digest.is_some() { "matched_configured_provider_digest" } else { "provider_reported_unpinned" },
            "independent_artifact_sha256": artifact_hash,
            "artifact_verification_status": if artifact_hash.is_some() { "independent_file_sha256_verified_runtime_binding_unattested" } else { "provider_reported_only" },
            "artifact_attestation_limitation": "Hashing an independently selected local file does not attest that the runtime loaded those bytes. Service tags are provider-reported manifest identity, not an independent artifact hash.",
            "require_independent_artifact_sha256": settings.require_independent_artifact_sha256,
            "license": settings.license,
            "tokenizer": {"sha256":tokenizer_hash,"implementation":"tokenizers-0.23","add_special_tokens":true,"truncation":false,"padding":false,"runtime_prompt_count":"must_equal_pinned_tokenizer_count","runtime_tokenizer_binding":"operator_selected_not_attested"},
            "transforms": {
                "query":{"version":"prepend-utf8-v1","prefix":settings.query_prefix},
                "document":{"version":"prepend-utf8-v1","prefix":settings.document_prefix}
            },
            "dimensions":settings.dimensions,"normalization":"l2-f64-accumulator-f32-output-v1","dtype":"f32-le",
            "chunker":crate::ingest::CHUNKER_VERSION,"parser":crate::ingest::PARSER_VERSION,
            "max_input_tokens":settings.max_input_tokens,"max_input_bytes":settings.max_input_bytes,"context_window":settings.context_window,
            "max_output_tokens":settings.max_output_tokens,"prompt_template":settings.prompt_template,
            "prompt_template_bytes": template_contract(&settings.prompt_template),
            "temperature":settings.temperature,"seed":settings.seed,
            "timeout_ms":settings.timeout_ms,"max_response_bytes":settings.max_response_bytes,
            "network_policy":"literal-loopback-no-proxy-no-redirect-no-retry-v1",
            "runtime_cloud_policy":"reject_remote_model_metadata; operator must configure OLLAMA_NO_CLOUD=1 and loopback bind",
            "cancellation":"cooperative_between_reads_and_requests; started_blocking_request_bounded_not_forcibly_aborted"
        });
        let canonical = canonical_json(&details)?;
        adapter.contract = ModelContract {
            id: sha256(&canonical),
            details,
        };
        remaining(deadline)?;
        Ok(adapter)
    }

    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts, &self.settings.document_prefix)
    }
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(&[text], &self.settings.query_prefix)?
            .pop()
            .ok_or_else(|| Error::new("provider_invalid_response", "missing query embedding"))
    }
    fn embed(&self, texts: &[&str], prefix: &str) -> Result<Vec<Vec<f32>>> {
        if self.purpose != ModelPurpose::Embedding {
            return Err(Error::new(
                "model_purpose_mismatch",
                "embedding requires an embedding model contract",
            ));
        }
        if self.settings.provider == "litellm" {
            return self.embed_litellm(texts, prefix);
        }
        let end = deadline(self.settings.timeout_ms);
        self.revalidate_until(end)?;
        let mut inputs = Vec::with_capacity(texts.len().min(4096));
        let mut total_bytes = 0usize;
        let mut total_tokens = 0usize;
        for text in texts {
            crate::mcp::checkpoint()?;
            total_bytes = total_bytes
                .checked_add(prefix.len())
                .and_then(|n| n.checked_add(text.len()))
                .ok_or_else(|| {
                    Error::new(
                        "provider_request_too_large",
                        "embedding request size overflow",
                    )
                })?;
            if total_bytes > MAX_REQUEST_BYTES || inputs.len() >= 4096 {
                return Err(Error::new(
                    "provider_request_too_large",
                    "embedding batch exceeds byte or input count limit",
                ));
            }
            let input = format!("{prefix}{text}");
            let tokens = self.count_tokens(&input)?;
            if tokens > self.settings.max_input_tokens || tokens > self.settings.context_window {
                return Err(Error::new(
                    "token_limit_exceeded",
                    "embedding input including task prefix exceeds token limit",
                ));
            }
            total_tokens = total_tokens.checked_add(tokens).ok_or_else(|| {
                Error::new("token_limit_exceeded", "embedding token count overflow")
            })?;
            inputs.push(input);
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let response = self.request(Method::POST, "api/embed", Some(&json!({
            "model":self.settings.model,"input":inputs,"truncate":false,
            "dimensions":self.settings.dimensions,"options":{"num_ctx":self.settings.context_window}
        })), end)?;
        self.check_response_model(&response)?;
        require_count(&response, "prompt_eval_count", total_tokens, true)?;
        let rows = response["embeddings"]
            .as_array()
            .ok_or_else(|| invalid("missing embeddings array"))?;
        if rows.len() != texts.len() {
            return Err(invalid("embedding response count differs from request"));
        }
        let mut vectors = Vec::with_capacity(rows.len());
        for row in rows {
            let values = row
                .as_array()
                .ok_or_else(|| invalid("embedding is not an array"))?;
            if values.len() != self.settings.dimensions {
                return Err(Error::new(
                    "vector_invalid",
                    "embedding dimension differs from contract",
                ));
            }
            let mut vector = Vec::with_capacity(values.len());
            for value in values {
                let value = value.as_f64().ok_or_else(|| {
                    Error::new("vector_invalid", "embedding contains a nonnumeric value")
                })? as f32;
                if !value.is_finite() {
                    return Err(Error::new(
                        "vector_invalid",
                        "embedding contains a nonfinite f32 value",
                    ));
                }
                vector.push(value);
            }
            normalize(&mut vector)?;
            vectors.push(vector);
        }
        self.revalidate_until(end)?;
        Ok(vectors)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        crate::mcp::checkpoint()?;
        if text.len() > MAX_REQUEST_BYTES {
            return Err(Error::new(
                "token_limit_exceeded",
                "tokenizer input exceeds byte limit",
            ));
        }
        self.tokenizer.as_ref().ok_or_else(|| Error::new("tokenizer_unavailable", "no pinned tokenizer; this provider uses an explicitly labeled byte input budget"))?.encode(text, true).map(|encoding| encoding.len())
            .map_err(|_| Error::new("tokenizer_invalid", "pinned tokenizer could not encode input"))
    }

    pub fn render_prompt(&self, system: &str, user_payload: &str) -> Result<String> {
        if system
            .len()
            .checked_add(user_payload.len())
            .is_none_or(|n| n > MAX_REQUEST_BYTES - 256)
        {
            return Err(Error::new(
                "provider_request_too_large",
                "prompt exceeds byte limit",
            ));
        }
        if self.settings.provider == "litellm" {
            return Ok(serde_json::to_string(
                &json!([{"role":"system","content":system},{"role":"user","content":user_payload}]),
            )?);
        }
        match self.settings.prompt_template.as_str() {
            "chatml-v1" => {
                if [system, user_payload]
                    .iter()
                    .any(|text| text.contains("<|im_start|>") || text.contains("<|im_end|>"))
                {
                    return Err(Error::new(
                        "invalid_prompt",
                        "payload contains a reserved prompt control token",
                    ));
                }
                Ok(format!(
                    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_payload}<|im_end|>\n<|im_start|>assistant\n"
                ))
            }
            "plain-v1" => Ok(format!(
                "SYSTEM:\n{system}\n\nUSER:\n{user_payload}\n\nASSISTANT:\n"
            )),
            _ => Err(Error::new(
                "model_config_invalid",
                "unsupported pinned prompt template",
            )),
        }
    }

    pub fn generate(&self, prompt: &str, schema: &Value) -> Result<String> {
        if self.purpose != ModelPurpose::Generator {
            return Err(Error::new(
                "model_purpose_mismatch",
                "generation requires a generator contract",
            ));
        }
        if !schema.is_object() {
            return Err(Error::new(
                "invalid_arguments",
                "generation format must be a JSON schema object",
            ));
        }
        if self.settings.provider == "litellm" {
            return self.generate_litellm(prompt, schema);
        }
        self.validate_prompt_budget(prompt)?;
        let end = deadline(self.settings.timeout_ms);
        let tokens = self.count_tokens(prompt)?;
        if tokens > self.settings.max_input_tokens
            || tokens
                .checked_add(self.settings.max_output_tokens)
                .is_none_or(|n| n > self.settings.context_window)
        {
            return Err(Error::new(
                "token_limit_exceeded",
                "full rendered prompt plus reserved output exceeds token limit",
            ));
        }
        self.revalidate_until(end)?;
        let response = self.request(Method::POST, "api/generate", Some(&json!({
            "model":self.settings.model,"prompt":prompt,"raw":true,"stream":false,
            "format":schema,"think":false,"truncate":false,"shift":false,
            "options":{"num_ctx":self.settings.context_window,"num_predict":self.settings.max_output_tokens,
                "temperature":self.settings.temperature,"seed":self.settings.seed}
        })), end)?;
        self.check_response_model(&response)?;
        if response["done"] != true || response["done_reason"] != "stop" {
            return Err(invalid(
                "generation did not complete normally; truncated output is not accepted",
            ));
        }
        if response
            .get("thinking")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
        {
            return Err(invalid("unexpected hidden thinking output"));
        }
        require_count(&response, "prompt_eval_count", tokens, true)?;
        require_count(
            &response,
            "eval_count",
            self.settings.max_output_tokens,
            false,
        )?;
        let text = response["response"]
            .as_str()
            .ok_or_else(|| invalid("missing generated response string"))?;
        if text.is_empty() || self.count_tokens(text)? > self.settings.max_output_tokens {
            return Err(invalid(
                "generated response is empty or exceeds output token limit",
            ));
        }
        // Schema-specific validation belongs to the answer verifier, but malformed JSON never escapes.
        serde_json::from_str::<Value>(text)
            .map_err(|_| invalid("generated structured response is not JSON"))?;
        self.revalidate_until(end)?;
        Ok(text.to_owned())
    }

    pub fn revalidate(&self) -> Result<()> {
        self.revalidate_until(deadline(self.settings.timeout_ms))
    }
    fn revalidate_until(&self, end: Instant) -> Result<()> {
        if self.read_identity(end)? != self.identity {
            return Err(Error::new(
                "model_identity_changed",
                "runtime version or installed model identity changed during operation",
            ));
        }
        Ok(())
    }
    fn read_identity(&self, end: Instant) -> Result<Value> {
        if self.settings.provider == "litellm" {
            let catalog = self.request(Method::GET, "models", None, end)?;
            let entries = catalog["data"]
                .as_array()
                .ok_or_else(|| invalid("missing provider model catalog"))?;
            let mut matches = entries
                .iter()
                .filter(|entry| entry["id"].as_str() == Some(self.settings.model.as_str()));
            let model = matches.next().ok_or_else(|| {
                Error::new(
                    "model_unavailable",
                    "configured alias is absent from authenticated provider catalog",
                )
            })?;
            if matches.next().is_some() {
                return Err(invalid("provider catalog contains duplicate model aliases"));
            }
            return Ok(json!({"catalog_model_alias":model["id"],"owned_by":model["owned_by"]}));
        }
        let version = self.request(Method::GET, "api/version", None, end)?;
        let version = version["version"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 256)
            .ok_or_else(|| invalid("missing bounded runtime version"))?;
        let tags = self.request(Method::GET, "api/tags", None, end)?;
        let models = tags["models"]
            .as_array()
            .ok_or_else(|| invalid("missing installed models array"))?;
        let mut matches = models
            .iter()
            .filter(|model| model["name"].as_str() == Some(self.settings.model.as_str()));
        let model = matches.next().ok_or_else(|| {
            Error::new(
                "model_unavailable",
                "configured exact model name is not installed locally",
            )
        })?;
        if matches.next().is_some() {
            return Err(invalid("duplicate installed model name"));
        }
        reject_remote(model)?;
        if model
            .get("model")
            .is_some_and(|name| name.as_str() != Some(self.settings.model.as_str()))
        {
            return Err(invalid("installed model name fields disagree"));
        }
        let digest = model["digest"]
            .as_str()
            .ok_or_else(|| invalid("missing installed model digest"))?;
        let digest = normalized_digest(digest);
        validate_hash(&digest, "provider reported digest")?;
        if self
            .settings
            .provider_reported_tag_digest
            .as_deref()
            .is_some_and(|expected| normalized_digest(expected) != digest)
        {
            return Err(Error::new(
                "model_identity_mismatch",
                "installed model digest differs from configured provider expectation",
            ));
        }
        Ok(
            json!({"version":version,"model":{"name":self.settings.model,"digest":digest,"size":model["size"],"details":model["details"]}}),
        )
    }
    fn check_response_model(&self, response: &Value) -> Result<()> {
        reject_remote(response)?;
        if response["model"].as_str() != Some(self.settings.model.as_str()) {
            return Err(invalid(
                "provider response model differs from pinned installed identity",
            ));
        }
        Ok(())
    }
    fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        end: Instant,
    ) -> Result<Value> {
        crate::mcp::checkpoint()?;
        let url = self
            .base
            .join(path)
            .map_err(|_| Error::new("model_config_invalid", "invalid provider API path"))?;
        let mut request = self.client.request(method, url).timeout(remaining(end)?);
        if let Some(body) = body {
            let bytes = serde_json::to_vec(body)?;
            if bytes.len() > MAX_REQUEST_BYTES {
                return Err(Error::new(
                    "provider_request_too_large",
                    "serialized model request exceeds byte limit",
                ));
            }
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(bytes);
        }
        let mut response = request.send().map_err(http_error)?;
        if response.status().is_redirection() {
            return Err(Error::new(
                "provider_redirect_rejected",
                "model provider API redirects are forbidden",
            ));
        }
        if !response.status().is_success() {
            // Do not echo an error body: model servers can include source/prompt bytes in it.
            return Err(Error::new(
                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    "model_unavailable"
                } else {
                    "provider_http_error"
                },
                format!(
                    "model provider API returned HTTP {}",
                    response.status().as_u16()
                ),
            ));
        }
        if response
            .content_length()
            .is_some_and(|n| n > self.settings.max_response_bytes as u64)
        {
            return Err(Error::new(
                "provider_response_too_large",
                "model response exceeds byte limit",
            ));
        }
        let mut bytes = Vec::new();
        let mut chunk = [0; 8192];
        loop {
            crate::mcp::checkpoint()?;
            remaining(end)?;
            let room = self
                .settings
                .max_response_bytes
                .saturating_sub(bytes.len())
                .saturating_add(1)
                .min(chunk.len());
            let count = response.read(&mut chunk[..room]).map_err(|error| {
                let timed_out =
                    error.kind() == std::io::ErrorKind::TimedOut || Instant::now() >= end;
                Error::new(
                    if timed_out {
                        "provider_timeout"
                    } else {
                        "provider_unavailable"
                    },
                    "bounded model response read failed; no response content is logged",
                )
            })?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..count]);
            if bytes.len() > self.settings.max_response_bytes {
                return Err(Error::new(
                    "provider_response_too_large",
                    "model response exceeds byte limit",
                ));
            }
        }
        remaining(end)?;
        crate::mcp::checkpoint()?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| invalid("model API returned malformed JSON"))?;
        if !value.is_object() || value.get("error").is_some() {
            return Err(invalid(
                "model API returned an error or non-object response",
            ));
        }
        Ok(value)
    }
}

fn validate_settings(settings: &ModelConfig, purpose: ModelPurpose) -> Result<()> {
    if !matches!(settings.provider.as_str(), "ollama" | "litellm") {
        return Err(Error::new(
            "provider_unavailable",
            "only explicitly configured Ollama or LiteLLM providers are supported",
        ));
    }
    if settings.model.is_empty()
        || settings.model.len() > 512
        || settings.model.chars().any(char::is_whitespace)
        || (settings.provider == "ollama"
            && (settings.model.contains(":cloud") || settings.model.ends_with("-cloud")))
        || settings.model.contains("://")
    {
        return Err(Error::new(
            "model_config_invalid",
            "a bounded exact local model name is required",
        ));
    }
    if settings.license.trim().is_empty() {
        return Err(Error::new(
            "model_config_invalid",
            "model license must be explicitly recorded",
        ));
    }
    if settings.timeout_ms == 0
        || settings.timeout_ms > 3_600_000
        || settings.max_response_bytes == 0
        || settings.max_response_bytes > MAX_RESPONSE_BYTES
        || settings.max_input_tokens == 0
        || settings.max_input_tokens > MAX_TOKENS
        || settings.context_window == 0
        || settings.context_window > MAX_TOKENS
        || settings.max_input_bytes == 0
        || settings.max_input_bytes > MAX_REQUEST_BYTES
        || !settings.temperature.is_finite()
        || settings.temperature < 0.0
        || settings.temperature > 2.0
        || settings.seed > i64::MAX as u64
        || settings.query_prefix.len() > MAX_REQUEST_BYTES / 2
        || settings.document_prefix.len() > MAX_REQUEST_BYTES / 2
    {
        return Err(Error::new(
            "model_config_invalid",
            "model limits, sampling values or prefixes are invalid",
        ));
    }
    if purpose == ModelPurpose::Embedding
        && (settings.dimensions == 0 || settings.dimensions > MAX_DIMENSIONS)
    {
        return Err(Error::new(
            "model_config_invalid",
            "embedding dimensions must be explicitly configured within bounds",
        ));
    }
    if purpose == ModelPurpose::Generator
        && (settings.max_output_tokens == 0
            || settings.max_output_tokens >= settings.context_window)
    {
        return Err(Error::new(
            "model_config_invalid",
            "generator output reservation must be positive and below context window",
        ));
    }
    if (settings.provider == "ollama"
        && !matches!(settings.prompt_template.as_str(), "chatml-v1" | "plain-v1"))
        || (settings.provider == "litellm"
            && purpose == ModelPurpose::Generator
            && settings.prompt_template != "openai-chat-v1")
    {
        return Err(Error::new(
            "model_config_invalid",
            "unsupported pinned prompt template",
        ));
    }
    if let Some(digest) = settings.provider_reported_tag_digest.as_deref() {
        validate_hash(&normalized_digest(digest), "provider_reported_tag_digest")?;
    }
    Ok(())
}

fn local_endpoint(text: &str) -> Result<Url> {
    let error = || {
        Error::new(
            "provider_endpoint_rejected",
            "endpoint requires literal 127.0.0.1 or [::1], explicit nonzero port, no credentials/query/fragment",
        )
    };
    let (scheme, rest) = text.split_once("://").ok_or_else(error)?;
    if !matches!(scheme, "http" | "https")
        || text.contains('\\')
        || text.chars().any(char::is_whitespace)
    {
        return Err(error());
    }
    let authority = rest.split('/').next().ok_or_else(error)?;
    let port = authority
        .strip_prefix("127.0.0.1:")
        .or_else(|| authority.strip_prefix("[::1]:"))
        .ok_or_else(error)?;
    if port.is_empty()
        || !port.bytes().all(|byte| byte.is_ascii_digit())
        || port.parse::<u16>().ok().is_none_or(|n| n == 0)
    {
        return Err(error());
    }
    let mut url = Url::parse(text).map_err(|_| error())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(error());
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

fn reject_remote(value: &Value) -> Result<()> {
    if ["remote_host", "remote_model"].iter().any(|key| {
        value
            .get(*key)
            .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
    }) {
        return Err(Error::new(
            "provider_endpoint_rejected",
            "remote-backed Ollama models are forbidden",
        ));
    }
    Ok(())
}
fn independent_artifact(settings: &ModelConfig, end: Instant) -> Result<Option<String>> {
    match (
        &settings.independent_artifact_path,
        &settings.independent_artifact_sha256,
    ) {
        (None, None) if !settings.require_independent_artifact_sha256 => Ok(None),
        (Some(path), Some(expected)) => {
            validate_hash(expected, "independent_artifact_sha256")?;
            let mut file = File::open(path).map_err(|_| {
                Error::new(
                    "artifact_verification_required",
                    "cannot open configured independent artifact",
                )
            })?;
            if !file.metadata()?.is_file() {
                return Err(Error::new(
                    "artifact_verification_required",
                    "independent artifact must be a regular file",
                ));
            }
            let mut hasher = Sha256::new();
            let mut buffer = [0; 64 * 1024];
            loop {
                crate::mcp::checkpoint()?;
                remaining(end)?;
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
            let actual = format!("{:x}", hasher.finalize());
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(Error::new(
                    "artifact_identity_mismatch",
                    "independent artifact bytes do not match the configured SHA-256",
                ));
            }
            Ok(Some(actual))
        }
        _ => Err(Error::new(
            "artifact_verification_required",
            "independent verification requires both a local artifact path and its expected SHA-256",
        )),
    }
}
fn read_bounded_file(path: &Path, limit: usize, code: &str) -> Result<Vec<u8>> {
    let file =
        File::open(path).map_err(|_| Error::new(code, "cannot open pinned local tokenizer"))?;
    if !file.metadata()?.is_file() {
        return Err(Error::new(code, "tokenizer must be a regular file"));
    }
    let mut bytes = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(Error::new(code, "tokenizer exceeds byte limit"));
    }
    Ok(bytes)
}
fn validate_hash(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::new(
            "model_config_invalid",
            format!("{field} must contain 64 SHA-256 hexadecimal characters"),
        ));
    }
    Ok(())
}
fn normalized_digest(digest: &str) -> String {
    digest
        .strip_prefix("sha256:")
        .unwrap_or(digest)
        .to_ascii_lowercase()
}
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn ordered(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort_unstable();
                let mut result = serde_json::Map::new();
                for key in keys {
                    result.insert(key.clone(), ordered(&map[key]));
                }
                Value::Object(result)
            }
            Value::Array(values) => Value::Array(values.iter().map(ordered).collect()),
            other => other.clone(),
        }
    }
    Ok(serde_json::to_vec(&ordered(value))?)
}
fn template_contract(template: &str) -> Value {
    match template {
        "chatml-v1" => {
            json!({"prefix":"<|im_start|>system\n","separator":"<|im_end|>\n<|im_start|>user\n","suffix":"<|im_end|>\n<|im_start|>assistant\n"})
        }
        _ => json!({"prefix":"SYSTEM:\n","separator":"\n\nUSER:\n","suffix":"\n\nASSISTANT:\n"}),
    }
}
fn deadline(timeout: u64) -> Instant {
    Instant::now() + Duration::from_millis(timeout)
}
fn remaining(end: Instant) -> Result<Duration> {
    end.checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| {
            Error::new(
                "provider_timeout",
                "model provider operation exceeded its total deadline",
            )
        })
}
fn http_error(error: reqwest::Error) -> Error {
    Error::new(
        if error.is_timeout() {
            "provider_timeout"
        } else {
            "provider_unavailable"
        },
        "model provider request failed; no response content is logged",
    )
}
fn invalid(message: &str) -> Error {
    Error::new("provider_invalid_response", message)
}
fn require_count(response: &Value, field: &str, expected: usize, equal: bool) -> Result<()> {
    let actual = response[field]
        .as_u64()
        .ok_or_else(|| invalid("missing or invalid runtime token counter"))?;
    if (equal && actual != expected as u64) || (!equal && actual > expected as u64) {
        return Err(Error::new(
            "tokenizer_mismatch",
            "runtime token count differs from pinned count or exceeds configured limit; truncation is not accepted",
        ));
    }
    Ok(())
}

fn norm(vector: &[f32]) -> Result<f64> {
    if vector.is_empty() || vector.len() > MAX_DIMENSIONS || vector.iter().any(|v| !v.is_finite()) {
        return Err(Error::new(
            "vector_invalid",
            "vector dimension or finite-value check failed",
        ));
    }
    let square = vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    if !square.is_finite() || square <= 0.0 {
        return Err(Error::new(
            "vector_invalid",
            "vector must have a finite nonzero norm",
        ));
    }
    Ok(square.sqrt())
}
fn normalize(vector: &mut [f32]) -> Result<()> {
    let length = norm(vector)?;
    for value in vector.iter_mut() {
        *value = (f64::from(*value) / length) as f32;
    }
    validate_vector(vector)
}
/// Persisted/query vectors must be normalized already; corrupt BLOBs are never repaired silently.
pub fn validate_vector(vector: &[f32]) -> Result<()> {
    if (norm(vector)? - 1.0).abs() > NORM_TOLERANCE {
        return Err(Error::new(
            "vector_invalid",
            "stored vector is not L2 normalized",
        ));
    }
    Ok(())
}
pub fn encode_vector(vector: &[f32]) -> Result<Vec<u8>> {
    validate_vector(vector)?;
    let length = vector
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::new("vector_invalid", "vector byte length overflow"))?;
    let mut bytes = Vec::with_capacity(length);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}
pub fn decode_vector(blob: &[u8], dimensions: usize) -> Result<Vec<f32>> {
    let length = dimensions
        .checked_mul(4)
        .ok_or_else(|| Error::new("vector_invalid", "vector byte length overflow"))?;
    if dimensions == 0 || dimensions > MAX_DIMENSIONS || blob.len() != length {
        return Err(Error::new(
            "vector_invalid",
            "vector BLOB length differs from exact checked dimension",
        ));
    }
    let vector: Vec<f32> = blob
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    validate_vector(&vector)?;
    Ok(vector)
}

impl LocalModel {
    /// Whether a pinned tokenizer can count this representation, not upstream chat framing.
    pub fn has_exact_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Byte budgets remain bytes. Remote chat framing is controlled upstream, not attested here.
    pub fn validate_prompt_budget(&self, prompt: &str) -> Result<()> {
        crate::mcp::checkpoint()?;
        if prompt.len() > self.settings.max_input_bytes {
            return Err(Error::new(
                "prompt_too_large",
                "serialized prompt exceeds explicit input byte budget",
            ));
        }
        if self.has_exact_tokenizer() {
            let tokens = self.count_tokens(prompt)?;
            if tokens > self.settings.max_input_tokens
                || (self.settings.provider == "ollama"
                    && tokens
                        .checked_add(self.settings.max_output_tokens)
                        .is_none_or(|n| n > self.settings.context_window))
            {
                return Err(Error::new(
                    "token_limit_exceeded",
                    "pinned prompt representation exceeds configured token budget",
                ));
            }
        }
        Ok(())
    }

    /// Observable provider claims, separate from immutable configured alias contracts.
    pub fn observations(&self) -> Value {
        self.observations
            .lock()
            .map(|value| value.clone())
            .unwrap_or_else(|_| json!({"observation_error":"state_unavailable"}))
    }

    fn prepare_litellm(settings: &ModelConfig, purpose: ModelPurpose) -> Result<Self> {
        let end = deadline(settings.timeout_ms);
        let base = litellm_endpoint(&settings.base_url)?;
        if settings.require_independent_artifact_sha256
            || settings.independent_artifact_path.is_some()
            || settings.independent_artifact_sha256.is_some()
            || settings.provider_reported_tag_digest.is_some()
        {
            return Err(Error::new(
                "artifact_verification_required",
                "remote alias identity cannot satisfy local artifact or Ollama digest verification policies",
            ));
        }
        let key_env = settings
            .api_key_env
            .as_deref()
            .filter(|name| {
                !name.is_empty()
                    && name.len() <= 256
                    && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            })
            .ok_or_else(|| {
                Error::new(
                    "provider_auth_unavailable",
                    "LiteLLM requires an explicit API-key environment variable reference",
                )
            })?;
        let key = std::env::var(key_env).map_err(|_| {
            Error::new(
                "provider_auth_unavailable",
                "configured API-key environment variable is unavailable",
            )
        })?;
        if key.is_empty() || key.len() > 16 * 1024 {
            return Err(Error::new(
                "provider_auth_unavailable",
                "configured API key is empty or oversized",
            ));
        }
        let mut authorization = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| {
                Error::new(
                    "provider_auth_unavailable",
                    "configured API key cannot be represented safely as an authorization header",
                )
            })?;
        authorization.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, authorization);
        let (tokenizer, tokenizer_hash) =
            match (&settings.tokenizer_path, &settings.tokenizer_sha256) {
                (None, None) => (None, None),
                (Some(path), Some(expected)) => {
                    validate_hash(expected, "tokenizer_sha256")?;
                    let bytes =
                        read_bounded_file(path, MAX_TOKENIZER_BYTES, "tokenizer_unavailable")?;
                    let hash = sha256(&bytes);
                    if !hash.eq_ignore_ascii_case(expected) {
                        return Err(Error::new(
                            "tokenizer_mismatch",
                            "tokenizer bytes do not match the pinned SHA-256",
                        ));
                    }
                    let mut tokenizer = Tokenizer::from_bytes(&bytes).map_err(|_| {
                        Error::new("tokenizer_invalid", "cannot parse pinned tokenizer")
                    })?;
                    tokenizer.with_truncation(None).map_err(|_| {
                        Error::new("tokenizer_invalid", "cannot disable tokenizer truncation")
                    })?;
                    tokenizer.with_padding(None);
                    (Some(tokenizer), Some(hash))
                }
                _ => {
                    return Err(Error::new(
                        "tokenizer_unavailable",
                        "optional tokenizer requires both path and SHA-256",
                    ));
                }
            };
        let timeout = remaining(end)?;
        let client = Client::builder()
            .default_headers(headers)
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .referer(false)
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .connect_timeout(timeout.min(Duration::from_secs(5)))
            .timeout(timeout)
            .build()
            .map_err(|_| {
                Error::new(
                    "provider_unavailable",
                    "cannot construct bounded authenticated provider client",
                )
            })?;
        let mut model = Self {
            contract: ModelContract {
                id: String::new(),
                details: Value::Null,
            },
            settings: settings.clone(),
            purpose,
            client,
            base,
            tokenizer,
            identity: Value::Null,
            observations: std::sync::Mutex::new(
                json!({"response_model_identity":null,"usage":null,"calls":0}),
            ),
        };
        model.identity = model.read_identity(end)?;
        let details = json!({
            "contract_version":2,"provider":"litellm","endpoint":model.base.as_str(),
            "purpose":match purpose { ModelPurpose::Embedding=>"embedding",ModelPurpose::Generator=>"generator" },
            "model":settings.model,"model_alias":settings.model,"catalog_identity":model.identity,
            "runtime":null,"provider_reported_tag_digest":null,"independent_artifact_sha256":null,
            "artifact_verification_status":"provider_identity_only","observable_response_model_identity":"reported_separately_per_operation",
            "identity_limitation":"Remote model aliases and response model identifiers are provider claims, not pinned backend artifact/runtime identities. Catalog revalidation cannot detect opaque routing or silent upstream changes.",
            "license":settings.license,"tokenizer":{"sha256":tokenizer_hash,"add_special_tokens":true,"padding":false,"truncation":false,"upstream_chat_framing_counted":false},
            "input_budget":{"unit":"bytes","limit":settings.max_input_bytes,"optional_pinned_representation_token_limit":if model.has_exact_tokenizer(){Some(settings.max_input_tokens)}else{None::<usize>}},
            "usage_policy":"provider_reported_usage_recorded_separately; never equated to bytes or locally attested chat counts",
            "transforms":{"query":{"version":"prepend-utf8-v1","prefix":settings.query_prefix},"document":{"version":"prepend-utf8-v1","prefix":settings.document_prefix}},
            "dimensions":settings.dimensions,"dtype":"f32-le","normalization":"l2-f64-accumulator-f32-output-v1",
            "chunker":crate::ingest::CHUNKER_VERSION,"parser":crate::ingest::PARSER_VERSION,
            "prompt_template":if purpose==ModelPurpose::Generator{"openai-chat-v1"}else{"not_used"},
            "prompt_representation":"UTF-8 serialized two-message JSON(system,user); upstream chat template unknown",
            "temperature":settings.temperature,"seed":settings.seed,"max_output_tokens":settings.max_output_tokens,
            "configured_context_window":settings.context_window,"context_enforcement":"provider_side; no exact upstream token guarantee without upstream tokenizer/template attestation",
            "timeout_ms":settings.timeout_ms,"max_response_bytes":settings.max_response_bytes,
            "network_policy":"explicit-https-remote-or-literal-loopback-http-no-proxy-no-redirect-no-retry",
            "privacy":"operator-authorized remote inference; inputs leave this host; upstream logging/retention controlled by provider"
        });
        model.contract = ModelContract {
            id: sha256(&canonical_json(&details)?),
            details,
        };
        remaining(end)?;
        Ok(model)
    }

    fn embed_litellm(&self, texts: &[&str], prefix: &str) -> Result<Vec<Vec<f32>>> {
        let end = deadline(self.settings.timeout_ms);
        let mut inputs = Vec::with_capacity(texts.len().min(2048));
        let mut bytes = 0usize;
        for text in texts {
            crate::mcp::checkpoint()?;
            let length = prefix.len().checked_add(text.len()).ok_or_else(|| {
                Error::new(
                    "provider_request_too_large",
                    "embedding input byte length overflow",
                )
            })?;
            bytes = bytes.checked_add(length).ok_or_else(|| {
                Error::new(
                    "provider_request_too_large",
                    "embedding batch byte length overflow",
                )
            })?;
            if length > self.settings.max_input_bytes
                || bytes > MAX_REQUEST_BYTES
                || inputs.len() >= 2048
            {
                return Err(Error::new(
                    "provider_request_too_large",
                    "embedding input or batch exceeds explicit byte/count limit",
                ));
            }
            let input = format!("{prefix}{text}");
            if self.has_exact_tokenizer()
                && self.count_tokens(&input)? > self.settings.max_input_tokens
            {
                return Err(Error::new(
                    "token_limit_exceeded",
                    "embedding representation exceeds optional pinned tokenizer limit",
                ));
            }
            inputs.push(input);
        }
        self.revalidate_until(end)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // dimensions is deliberately omitted: arbitrary embedding models do not support reduction.
        // The actual returned dimension must equal the explicitly configured vector-space contract.
        let response = self.request(
            Method::POST,
            "embeddings",
            Some(&json!({
                "model":self.settings.model,"input":inputs,"encoding_format":"float"
            })),
            end,
        )?;
        let rows = response["data"]
            .as_array()
            .ok_or_else(|| invalid("missing embedding data array"))?;
        if rows.len() != texts.len() {
            return Err(invalid("embedding response count differs from request"));
        }
        let mut vectors: Vec<Option<Vec<f32>>> = (0..texts.len()).map(|_| None).collect();
        for row in rows {
            let index = row["index"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n < texts.len())
                .ok_or_else(|| invalid("embedding index is missing or outside request"))?;
            if vectors[index].is_some() {
                return Err(invalid("duplicate embedding response index"));
            }
            let values = row["embedding"]
                .as_array()
                .ok_or_else(|| invalid("embedding data is not an array"))?;
            if values.len() != self.settings.dimensions {
                return Err(Error::new(
                    "vector_invalid",
                    "embedding dimension differs from contract",
                ));
            }
            let mut vector = values
                .iter()
                .map(|value| {
                    value.as_f64().map(|n| n as f32).ok_or_else(|| {
                        Error::new("vector_invalid", "embedding contains a nonnumeric value")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            normalize(&mut vector)?;
            vectors[index] = Some(vector);
        }
        self.record_observation(&response, false)?;
        self.revalidate_until(end)?;
        vectors
            .into_iter()
            .map(|vector| vector.ok_or_else(|| invalid("missing indexed embedding")))
            .collect()
    }

    fn generate_litellm(&self, prompt: &str, schema: &Value) -> Result<String> {
        self.validate_prompt_budget(prompt)?;
        let messages: Value = serde_json::from_str(prompt).map_err(|_| {
            Error::new(
                "invalid_prompt",
                "native chat prompt must be serialized system/user message JSON",
            )
        })?;
        let rows = messages
            .as_array()
            .filter(|rows| rows.len() == 2)
            .ok_or_else(|| {
                Error::new(
                    "invalid_prompt",
                    "native chat prompt must contain exactly system and user messages",
                )
            })?;
        for (row, role) in rows.iter().zip(["system", "user"]) {
            if row.as_object().is_none_or(|map| map.len() != 2)
                || row["role"] != role
                || !row["content"].is_string()
            {
                return Err(Error::new(
                    "invalid_prompt",
                    "native chat roles/content are invalid; tools and extra fields are forbidden",
                ));
            }
        }
        let end = deadline(self.settings.timeout_ms);
        self.revalidate_until(end)?;
        let response = self.request(Method::POST, "chat/completions", Some(&json!({
            "model":self.settings.model,"messages":messages,"stream":false,"n":1,
            "temperature":self.settings.temperature,"seed":self.settings.seed,
            "max_tokens":self.settings.max_output_tokens,
            "response_format":{"type":"json_schema","json_schema":{"name":"scribe_answer","schema":schema,"strict":true}}
        })), end)?;
        let choices = response["choices"]
            .as_array()
            .filter(|choices| choices.len() == 1)
            .ok_or_else(|| invalid("generation requires exactly one choice"))?;
        let choice = &choices[0];
        if choice["finish_reason"] != "stop" || choice["index"].as_u64() != Some(0) {
            return Err(invalid("generation stopped abnormally or was truncated"));
        }
        let message = &choice["message"];
        if message["role"] != "assistant"
            || message
                .get("tool_calls")
                .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|v| !v.is_empty()))
            || message.get("function_call").is_some_and(|v| !v.is_null())
            || message
                .get("refusal")
                .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
        {
            return Err(invalid(
                "generation returned an unexpected role, refusal or tool call",
            ));
        }
        let text = message["content"]
            .as_str()
            .filter(|text| !text.is_empty())
            .ok_or_else(|| invalid("generation returned no structured content"))?;
        serde_json::from_str::<Value>(text)
            .map_err(|_| invalid("generated structured response is not JSON"))?;
        self.record_observation(&response, true)?;
        self.revalidate_until(end)?;
        Ok(text.to_owned())
    }

    fn record_observation(&self, response: &Value, generation: bool) -> Result<()> {
        let model = response["model"]
            .as_str()
            .filter(|value| !value.is_empty() && value.len() <= 1024)
            .ok_or_else(|| invalid("missing bounded observable response model identity"))?;
        let usage = &response["usage"];
        let prompt = usage["prompt_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("missing provider-reported prompt token usage"))?;
        let total = usage["total_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("missing provider-reported total token usage"))?;
        if total < prompt {
            return Err(invalid("provider token usage counters are inconsistent"));
        }
        if generation {
            let output = usage["completion_tokens"]
                .as_u64()
                .ok_or_else(|| invalid("missing provider-reported output token usage"))?;
            if output > self.settings.max_output_tokens as u64
                || prompt.checked_add(output).is_none_or(|n| n != total)
            {
                return Err(invalid(
                    "provider output usage exceeds token cap or is inconsistent",
                ));
            }
        }
        let mut observations = self.observations.lock().map_err(|_| {
            Error::new(
                "provider_unavailable",
                "provider observation state is unavailable",
            )
        })?;
        if observations["response_model_identity"]
            .as_str()
            .is_some_and(|old| old != model)
        {
            return Err(Error::new(
                "model_identity_changed",
                "observable backend response model changed within this operation",
            ));
        }
        let calls = observations["calls"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid("provider observation count overflow"))?;
        let cumulative_prompt = observations["aggregate_usage"]["prompt_tokens"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(prompt)
            .ok_or_else(|| invalid("aggregate provider usage overflow"))?;
        let cumulative_total = observations["aggregate_usage"]["total_tokens"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(total)
            .ok_or_else(|| invalid("aggregate provider usage overflow"))?;
        *observations = json!({"response_model_identity":model,"usage":usage,"usage_scope":"last_response",
            "aggregate_usage":{"prompt_tokens":cumulative_prompt,"total_tokens":cumulative_total},
            "usage_attribution":"provider_reported_not_independently_verified","calls":calls,
            "input_budget_unit":"bytes","upstream_chat_framing_counted":false});
        Ok(())
    }
}

fn litellm_endpoint(text: &str) -> Result<Url> {
    let error = || {
        Error::new(
            "provider_endpoint_rejected",
            "LiteLLM requires explicit HTTPS endpoint (HTTP only for literal loopback), without credentials/query/fragment",
        )
    };
    if text.contains('\\') || text.chars().any(char::is_whitespace) {
        return Err(error());
    }
    let mut url = Url::parse(text).map_err(|_| error())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host_str().is_none()
    {
        return Err(error());
    }
    if url.scheme() == "http" {
        return local_endpoint(text);
    }
    if url.scheme() != "https" {
        return Err(error());
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}
