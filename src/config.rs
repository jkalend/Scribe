use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_file_bytes: usize,
    pub max_frontmatter_bytes: usize,
    pub max_chunk_bytes: usize,
    pub max_query_bytes: usize,
    pub max_results: usize,
    pub max_context_bytes: usize,
    pub max_response_bytes: usize,
    pub timeout_ms: u64,
    pub max_files: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_bytes: 4 * 1024 * 1024,
            max_frontmatter_bytes: 65536,
            max_chunk_bytes: 8192,
            max_query_bytes: 4096,
            max_results: 100,
            max_context_bytes: 65536,
            max_response_bytes: 1024 * 1024,
            timeout_ms: 10000,
            max_files: 10000,
        }
    }
}
fn role() -> String {
    "notes".into()
}
fn trust() -> String {
    "user-configured".into()
}
fn ignore_dirs() -> Vec<String> {
    [
        ".git",
        ".kb",
        "target",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "templates",
    ]
    .map(String::from)
    .to_vec()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Root {
    pub path: PathBuf,
    #[serde(default = "role")]
    pub role: String,
    #[serde(default = "trust")]
    pub trust: String,
    #[serde(default = "ignore_dirs")]
    pub ignore_dirs: Vec<String>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    #[default]
    Light,
    Full,
}
impl Profile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Full => "full",
        }
    }
    pub fn is_full(self) -> bool {
        self == Self::Full
    }
}
#[derive(Clone, Debug)]
pub struct Config {
    pub database: PathBuf,
    pub roots: BTreeMap<String, Root>,
    pub limits: Limits,
    pub profile: Profile,
    pub full: Option<crate::full_config::FullConfig>,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    profile: Option<String>,
    #[serde(default)]
    index: IndexConfig,
    #[serde(default)]
    roots: BTreeMap<String, Root>,
    #[serde(default)]
    limits: Limits,
    full: Option<crate::full_config::FullConfig>,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexConfig {
    database: Option<PathBuf>,
}

pub fn validate_profile(value: &str) -> Result<()> {
    match value {
        "light" => Ok(()),
        "full" if cfg!(feature = "full") => Ok(()),
        "full" => Err(Error::new(
            "profile_unavailable",
            "this light binary has no full profile; no lexical fallback is performed",
        )),
        _ => Err(Error::new(
            "invalid_profile",
            "profile must be light or full",
        )),
    }
}
fn profile_config(file: Option<&Path>, profile: Option<&str>) -> Result<FileConfig> {
    let selected = file
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("KB_CONFIG").map(PathBuf::from));
    let mut raw = match selected {
        Some(path) => {
            let bytes = std::fs::read(&path)?;
            if bytes.len() > 262144 {
                return Err(Error::new(
                    "invalid_config",
                    "configuration exceeds 256 KiB",
                ));
            }
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| Error::new("invalid_config", "configuration must be UTF-8"))?;
            toml::from_str::<FileConfig>(text)
                .map_err(|e| Error::new("invalid_config", e.to_string()))?
        }
        None => FileConfig::default(),
    };
    let selected_profile = profile
        .map(str::to_owned)
        .or_else(|| std::env::var("KB_PROFILE").ok())
        .or_else(|| raw.profile.take())
        .unwrap_or_else(|| "light".into());
    validate_profile(&selected_profile)?;
    raw.profile = Some(selected_profile);
    Ok(raw)
}

impl Config {
    pub fn load(
        file: Option<&Path>,
        database: Option<PathBuf>,
        roots: &[String],
        profile: Option<String>,
    ) -> Result<Self> {
        let raw = profile_config(file, profile.as_deref())?;
        let selected_profile = if raw.profile.as_deref() == Some("full") {
            Profile::Full
        } else {
            Profile::Light
        };
        let mut configured_roots = raw.roots;
        for input in roots {
            let (key, value) = input
                .split_once('=')
                .ok_or_else(|| Error::new("invalid_config", "--root requires key=absolute-path"))?;
            configured_roots.insert(
                key.into(),
                Root {
                    path: value.into(),
                    role: role(),
                    trust: trust(),
                    ignore_dirs: ignore_dirs(),
                },
            );
        }
        let database = database.or(raw.index.database).ok_or_else(|| {
            Error::new(
                "configuration_required",
                "configure [index].database or --database; no checkout default is used",
            )
        })?;
        let mut config = Self {
            database,
            roots: configured_roots,
            limits: raw.limits,
            profile: selected_profile,
            full: raw.full,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn embedding(&self) -> Result<&crate::full_config::ModelConfig> {
        self.full
            .as_ref()
            .and_then(|full| full.embedding.as_ref())
            .ok_or_else(|| {
                Error::new(
                    "embedding_unavailable",
                    "full retrieval requires explicitly configured [full.embedding]",
                )
            })
    }
    pub fn generator(&self) -> Result<&crate::full_config::ModelConfig> {
        self.full
            .as_ref()
            .and_then(|full| full.generator.as_ref())
            .ok_or_else(|| {
                Error::new(
                    "generator_unavailable",
                    "answer requires explicitly configured [full.generator]",
                )
            })
    }
    pub fn envelope<T>(&self, generation: i64, data: T) -> crate::model::Envelope<T> {
        let mut result = crate::model::Envelope::light(generation, data);
        result.profile = self.profile.as_str().into();
        result
    }
    pub fn validate(&mut self) -> Result<()> {
        if !self.database.is_absolute() || self.database.starts_with("\\\\") {
            return Err(Error::new(
                "invalid_config",
                "database must be an absolute local path, not a network share",
            ));
        }
        if self.roots.is_empty() {
            return Err(Error::new(
                "configuration_required",
                "at least one explicit source root is required",
            ));
        }
        let l = &self.limits;
        if l.max_file_bytes == 0
            || l.max_file_bytes > 64 * 1024 * 1024
            || l.max_frontmatter_bytes == 0
            || l.max_frontmatter_bytes > 1024 * 1024
            || l.max_chunk_bytes < 256
            || l.max_chunk_bytes > l.max_file_bytes
            || l.max_query_bytes == 0
            || l.max_query_bytes > 65536
            || l.max_results == 0
            || l.max_results > 1000
            || l.max_context_bytes == 0
            || l.max_context_bytes > 1024 * 1024
            || l.max_response_bytes < l.max_context_bytes
            || l.max_response_bytes > 16 * 1024 * 1024
            || l.timeout_ms == 0
            || l.timeout_ms > 60000
            || l.max_files == 0
            || l.max_files > 100000
        {
            return Err(Error::new(
                "invalid_config",
                "limits are zero, inconsistent, or exceed hard safety maxima",
            ));
        }
        for (key, root) in &mut self.roots {
            if key.is_empty()
                || key.len() > 64
                || !key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err(Error::new(
                    "invalid_config",
                    "root keys allow only ASCII letters, digits, underscore and hyphen",
                ));
            }
            if !root.path.is_absolute() {
                return Err(Error::new("invalid_config", "source root must be absolute"));
            }
            root.path = root.path.canonicalize()?;
            if !root.path.is_dir() {
                return Err(Error::new(
                    "invalid_config",
                    "source root is not a directory",
                ));
            }
            if root.role.len() > 128
                || root.trust.len() > 128
                || root
                    .ignore_dirs
                    .iter()
                    .any(|v| v.is_empty() || v.contains(['/', '\\']))
            {
                return Err(Error::new("invalid_config", "invalid root policy"));
            }
        }
        Ok(())
    }
    pub fn resolve_source(&self, root_key: &str, relative: &str) -> Result<PathBuf> {
        let root = self
            .roots
            .get(root_key)
            .ok_or_else(|| Error::new("inaccessible_path", "unknown source root"))?;
        let path = Path::new(relative);
        if relative.is_empty()
            || relative.contains(['\\', ':'])
            || path.is_absolute()
            || path
                .components()
                .any(|v| !matches!(v, Component::Normal(_)))
        {
            return Err(Error::new(
                "inaccessible_path",
                "source path must be normalized and relative",
            ));
        }
        if path.components().any(|v| {
            root.ignore_dirs
                .iter()
                .any(|ignored| v.as_os_str() == ignored.as_str())
        }) {
            return Err(Error::new(
                "inaccessible_path",
                "source path is excluded by root policy",
            ));
        }
        let canonical = root
            .path
            .join(path)
            .canonicalize()
            .map_err(|_| Error::new("stale_source", "indexed source is unavailable"))?;
        if !canonical.starts_with(&root.path) || !canonical.is_file() {
            return Err(Error::new(
                "inaccessible_path",
                "source escapes configured root or is not a file",
            ));
        }
        Ok(canonical)
    }
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(&self.roots).expect("serializable root policies"));
        hasher.update(serde_json::to_vec(&self.limits).expect("serializable limits"));
        hasher.update(crate::ingest::PARSER_VERSION);
        hasher.update(crate::ingest::CHUNKER_VERSION);
        format!("{:x}", hasher.finalize())
    }
}
