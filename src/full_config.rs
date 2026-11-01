use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub license: String,
    pub api_key_env: Option<String>,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_sha256: Option<String>,
    pub provider_reported_tag_digest: Option<String>,
    pub independent_artifact_path: Option<PathBuf>,
    pub independent_artifact_sha256: Option<String>,
    pub require_independent_artifact_sha256: bool,
    pub dimensions: usize,
    pub query_prefix: String,
    pub document_prefix: String,
    pub max_input_tokens: usize,
    pub max_input_bytes: usize,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub prompt_template: String,
    pub temperature: f64,
    pub seed: u64,
    pub timeout_ms: u64,
    pub max_response_bytes: usize,
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".into(),
            base_url: String::new(),
            model: String::new(),
            license: String::new(),
            api_key_env: None,
            tokenizer_path: None,
            tokenizer_sha256: None,
            provider_reported_tag_digest: None,
            independent_artifact_path: None,
            independent_artifact_sha256: None,
            require_independent_artifact_sha256: false,
            dimensions: 0,
            query_prefix: String::new(),
            document_prefix: String::new(),
            max_input_tokens: 8192,
            max_input_bytes: 32768,
            context_window: 8192,
            max_output_tokens: 1024,
            prompt_template: "chatml-v1".into(),
            temperature: 0.0,
            seed: 0,
            timeout_ms: 30000,
            max_response_bytes: 4 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    pub fusion: String,
    pub lexical_weight: f64,
    pub candidates: usize,
    pub rrf_k: usize,
}
impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            fusion: "rrf".into(),
            lexical_weight: 0.5,
            candidates: 100,
            rrf_k: 60,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullConfig {
    pub embedding: Option<ModelConfig>,
    pub generator: Option<ModelConfig>,
    pub retrieval: RetrievalConfig,
    pub index_timeout_ms: u64,
}
impl Default for FullConfig {
    fn default() -> Self {
        Self {
            embedding: None,
            generator: None,
            retrieval: RetrievalConfig::default(),
            index_timeout_ms: 300000,
        }
    }
}
