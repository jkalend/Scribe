use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub id: String,
    pub title: String,
    pub doc_type: String,
    pub status: String,
    pub tags: Vec<String>,
    pub aliases: Vec<String>,
    pub depends_on: Vec<String>,
    pub supersedes: Vec<String>,
    pub conflicts: Vec<String>,
    pub updated: Option<String>,
    pub extra: serde_json::Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub target: String,
    pub line: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub key: String,
    pub parent_key: Option<String>,
    pub heading: String,
    pub line_start: usize,
    pub line_end: usize,
    pub body: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub root_key: String,
    pub path: String,
    pub key: String,
    pub sha256: String,
    pub mtime: f64,
    pub size: u64,
    pub meta: Metadata,
    pub chunks: Vec<Chunk>,
    pub links: Vec<Link>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Filters {
    pub root: Option<String>,
    pub role: Option<String>,
    pub status: Option<String>,
    pub project: Option<String>,
    #[serde(rename = "type")]
    pub doc_type: Option<String>,
    pub tag: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
}
pub fn default_limit() -> usize {
    10
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default = "default_limit")]
    pub limit: usize,
}
impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: Filters::default(),
            limit: default_limit(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GetRequest {
    pub key: String,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRequest {
    pub query: String,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub budget_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerRequest {
    pub question: String,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub budget_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Leg {
    pub name: String,
    pub rank: usize,
    pub raw_score: f64,
    pub score_kind: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub root_key: String,
    pub document_key: String,
    pub chunk_key: String,
    pub path: String,
    pub heading: String,
    pub line_start: usize,
    pub line_end: usize,
    pub source_sha256: String,
    pub status: String,
    pub role: String,
    pub authority: String,
    pub project: String,
    pub doc_type: String,
    pub tags: Vec<String>,
    pub updated: Option<String>,
    pub supersedes: Vec<String>,
    pub conflicts: Vec<String>,
    pub snippet: String,
    pub body: String,
    pub legs: Vec<Leg>,
    #[serde(default)]
    pub fused_rank: Option<usize>,
    #[serde(default)]
    pub fused_score: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub schema_version: u32,
    pub profile: String,
    pub index_generation: i64,
    pub retrieval_legs: Vec<String>,
    pub embedding_contract_id: Option<String>,
    pub generator_contract_id: Option<String>,
    #[serde(default)]
    pub model_contracts: serde_json::Value,
    #[serde(default)]
    pub fusion_method: Option<String>,
    pub data: T,
}
impl<T> Envelope<T> {
    pub fn light(index_generation: i64, data: T) -> Self {
        Self {
            schema_version: 1,
            profile: "light".into(),
            index_generation,
            retrieval_legs: vec!["lexical".into()],
            embedding_contract_id: None,
            generator_contract_id: None,
            model_contracts: serde_json::Value::Null,
            fusion_method: None,
            data,
        }
    }
}
