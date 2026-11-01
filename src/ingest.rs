use crate::{
    config::{Config, Limits},
    error::{Error, Result},
    model::{Chunk, Document, Link, Metadata},
};
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use saphyr::{LoadableYamlNode, Yaml};
use saphyr_parser::{Event as YamlEvent, Parser as YamlParser};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    path::{Component, Path},
    time::{Instant, UNIX_EPOCH},
};
use walkdir::WalkDir;

pub const PARSER_VERSION: &str = "commonmark-0.13.4-tables-yaml-0.2.2-v1";
pub const CHUNKER_VERSION: &str = "heading-block-whole-lines-v1";
const MAX_YAML_DEPTH: usize = 16;

pub fn digest(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

/// Normalize only line endings, never whitespace, Unicode, or a UTF-8 BOM.
pub fn normalized_source(raw: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(raw).map_err(|e| {
        Error::new(
            "invalid_utf8",
            format!("source is not UTF-8 at byte {}", e.valid_up_to()),
        )
    })?;
    let mut normalized = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(index) = remaining.find('\r') {
        normalized.push_str(&remaining[..index]);
        normalized.push('\n');
        remaining = &remaining[index + 1..];
        if remaining.starts_with('\n') {
            remaining = &remaining[1..];
        }
    }
    normalized.push_str(remaining);
    Ok(normalized)
}

fn source_error(root: &str, path: &str, error: Error) -> Error {
    Error::new(error.code, format!("{root}:{path}: {}", error.message))
}

fn io_error(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::new("source_io", format!("{}: {error}", path.display()))
}

// Windows junctions/reparse points must not bypass the no-follow directory policy.
fn reparse(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

/// Return a complete candidate or an error; no partial candidate is published.
pub fn scan(config: &Config) -> Result<Vec<Document>> {
    scan_cached(config, &[])
}

/// Reuse parsed source only when the caller has verified the configuration and
/// parser/chunker fingerprint. Every current file is still reopened and hashed.
pub fn scan_cached(config: &Config, previous: &[Document]) -> Result<Vec<Document>> {
    let cached: HashMap<(&str, &str), &Document> = previous
        .iter()
        .map(|doc| ((doc.root_key.as_str(), doc.path.as_str()), doc))
        .collect();
    let started = Instant::now();
    let mut documents = Vec::new();
    for (root_key, root) in &config.roots {
        let root_meta = fs::symlink_metadata(&root.path).map_err(|e| io_error(&root.path, e))?;
        if !root_meta.is_dir() || reparse(&root_meta) {
            return Err(Error::new(
                "inaccessible_path",
                format!("{root_key}: configured root is not a real directory"),
            ));
        }
        let mut walker = WalkDir::new(&root.path)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter();
        while let Some(entry) = walker.next() {
            if started.elapsed().as_millis() >= u128::from(config.limits.timeout_ms) {
                return Err(Error::new(
                    "timeout",
                    "source scan exceeded configured timeout",
                ));
            }
            let entry = entry.map_err(|e| Error::new("source_io", format!("{root_key}: {e}")))?;
            let metadata =
                fs::symlink_metadata(entry.path()).map_err(|e| io_error(entry.path(), e))?;
            if reparse(&metadata) || metadata.file_type().is_symlink() {
                if entry.file_type().is_dir() {
                    walker.skip_current_dir();
                }
                continue;
            }
            if entry.depth() > 0
                && metadata.is_dir()
                && root
                    .ignore_dirs
                    .iter()
                    .any(|name| entry.file_name() == name.as_str())
            {
                walker.skip_current_dir();
                continue;
            }
            if !metadata.is_file()
                || !entry
                    .path()
                    .extension()
                    .is_some_and(|v| v.eq_ignore_ascii_case("md"))
            {
                continue;
            }
            if documents.len() >= config.limits.max_files {
                return Err(Error::new(
                    "file_limit",
                    format!(
                        "{root_key}: Markdown source count exceeds {}",
                        config.limits.max_files
                    ),
                ));
            }
            let relative = entry
                .path()
                .strip_prefix(&root.path)
                .map_err(|_| Error::new("inaccessible_path", "source escaped configured root"))?;
            let relative = relative.to_str().ok_or_else(|| {
                Error::new(
                    "invalid_path",
                    format!("{root_key}: source filename is not UTF-8"),
                )
            })?;
            let relative = if std::path::MAIN_SEPARATOR == '\\' {
                relative.replace('\\', "/")
            } else {
                relative.to_owned()
            };
            let path = config
                .resolve_source(root_key, &relative)
                .map_err(|e| source_error(root_key, &relative, e))?;
            let file = File::open(&path).map_err(|e| io_error(&path, e))?;
            let opened = file.metadata().map_err(|e| io_error(&path, e))?;
            if !opened.is_file() || opened.len() > config.limits.max_file_bytes as u64 {
                return Err(source_error(
                    root_key,
                    &relative,
                    Error::new(
                        "file_limit",
                        "source is not a regular file or exceeds max_file_bytes",
                    ),
                ));
            }
            let mut raw = Vec::with_capacity(opened.len() as usize);
            file.take(config.limits.max_file_bytes.saturating_add(1) as u64)
                .read_to_end(&mut raw)
                .map_err(|e| io_error(&path, e))?;
            let mtime = opened
                .modified()
                .map_err(|e| io_error(&path, e))?
                .duration_since(UNIX_EPOCH)
                .map(|v| v.as_secs_f64())
                .unwrap_or_else(|e| -e.duration().as_secs_f64());
            if raw.len() > config.limits.max_file_bytes {
                return Err(source_error(
                    root_key,
                    &relative,
                    Error::new(
                        "file_limit",
                        "source grew beyond max_file_bytes during read",
                    ),
                ));
            }
            let hash = digest(&raw);
            let document = if let Some(previous) = cached
                .get(&(root_key.as_str(), relative.as_str()))
                .filter(|doc| doc.sha256 == hash)
            {
                let mut document = (**previous).clone();
                document.mtime = mtime;
                document.size = raw.len() as u64;
                document
            } else {
                parse_inner(root_key, &relative, &raw, mtime, &config.limits, Some(hash))
                    .map_err(|e| source_error(root_key, &relative, e))?
            };
            if started.elapsed().as_millis() >= u128::from(config.limits.timeout_ms) {
                return Err(Error::new(
                    "timeout",
                    format!("{root_key}:{relative}: source scan exceeded configured timeout"),
                ));
            }
            documents.push(document);
        }
    }
    Ok(documents)
}

pub fn parse(
    root_key: &str,
    relative: &str,
    raw: &[u8],
    mtime: f64,
    limits: &Limits,
) -> Result<Document> {
    parse_inner(root_key, relative, raw, mtime, limits, None)
        .map_err(|e| source_error(root_key, relative, e))
}

fn parse_inner(
    root_key: &str,
    relative: &str,
    raw: &[u8],
    mtime: f64,
    limits: &Limits,
    source_sha: Option<String>,
) -> Result<Document> {
    if relative.is_empty()
        || relative.contains(['\\', ':'])
        || Path::new(relative).is_absolute()
        || relative
            .split('/')
            .any(|v| v.is_empty() || v == "." || v == "..")
        || Path::new(relative)
            .components()
            .any(|v| !matches!(v, Component::Normal(_)))
    {
        return Err(Error::new(
            "invalid_path",
            "source path must be normalized and relative",
        ));
    }
    if raw.len() > limits.max_file_bytes {
        return Err(Error::new("file_limit", "source exceeds max_file_bytes"));
    }
    if limits.max_chunk_bytes == 0 {
        return Err(Error::new(
            "chunk_limit",
            "max_chunk_bytes must be positive",
        ));
    }
    let source = normalized_source(raw)?;
    let lines = Lines::new(&source);
    let (mut meta, body_start) = frontmatter(&source, limits)?;
    let source_sha = source_sha.unwrap_or_else(|| digest(raw));
    let document_key = format!("d:{}", identity(&[root_key, relative]));
    let mut links = Vec::new();
    let mut headings: Vec<Heading> = Vec::new();
    let mut current_heading: Option<Heading> = None;
    let mut depth = 0usize;
    let mut block_boundaries = vec![body_start, source.len()];
    let parser = Parser::new_ext(&source[body_start..], Options::ENABLE_TABLES).into_offset_iter();
    for (event, range) in parser {
        let start = body_start + range.start;
        let end = body_start + range.end;
        match &event {
            Event::Start(tag) => {
                if depth == 0 {
                    block_boundaries.push(lines.start_at(start));
                }
                depth += 1;
                match tag {
                    Tag::Heading { level, .. } => {
                        current_heading = Some(Heading {
                            start: lines.start_at(start),
                            level: *level as u8,
                            title: String::new(),
                        })
                    }
                    Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => links.push(Link {
                        target: dest_url.to_string(),
                        line: lines.number(start),
                    }),
                    _ => {}
                }
            }
            Event::End(tag) => {
                if matches!(tag, TagEnd::Heading(_)) {
                    if let Some(heading) = current_heading.take() {
                        headings.push(heading);
                    }
                }
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    block_boundaries.push(lines.after(end));
                }
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some(heading) = &mut current_heading {
                    heading.title.push_str(text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(heading) = &mut current_heading {
                    heading.title.push(' ');
                }
            }
            _ => {
                if depth == 0 {
                    block_boundaries.push(lines.start_at(start));
                    block_boundaries.push(lines.after(end));
                }
            }
        }
    }
    if meta.title.is_empty() {
        meta.title = headings
            .first()
            .map(|h| h.title.clone())
            .unwrap_or_else(|| {
                Path::new(relative)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(relative)
                    .to_owned()
            });
    }
    block_boundaries.sort_unstable();
    block_boundaries.dedup();
    let mut sections = vec![Section {
        start: body_start,
        end: source.len(),
        heading: String::new(),
        parent: None,
    }];
    let mut ancestors: Vec<(u8, usize)> = Vec::new();
    for heading in headings {
        // Nested inline/container offsets may share a source line: never overlap excerpts.
        if heading.start < body_start
            || heading.start < sections.last().map_or(body_start, |s| s.start)
        {
            continue;
        }
        while ancestors
            .last()
            .is_some_and(|(level, _)| *level >= heading.level)
        {
            ancestors.pop();
        }
        let parent = ancestors.last().map(|(_, index)| *index);
        let title = heading.title.trim();
        let path = match parent {
            Some(index) => format!("{} / {title}", sections[index].heading),
            None => title.to_owned(),
        };
        if let Some(previous) = sections.last_mut() {
            previous.end = heading.start;
        }
        let index = sections.len();
        sections.push(Section {
            start: heading.start,
            end: source.len(),
            heading: path,
            parent,
        });
        ancestors.push((heading.level, index));
    }
    let mut chunks = Vec::new();
    let mut section_keys: Vec<Option<String>> = vec![None; sections.len()];
    for (index, section) in sections.iter().enumerate() {
        if source[section.start..section.end].trim().is_empty() {
            continue;
        }
        let mut start = section.start;
        while start < section.end {
            let ceiling = start
                .saturating_add(limits.max_chunk_bytes)
                .min(section.end);
            let end = if ceiling == section.end {
                section.end
            } else {
                let safe = block_boundaries.partition_point(|boundary| *boundary <= ceiling);
                let block_end = safe
                    .checked_sub(1)
                    .map(|n| block_boundaries[n])
                    .filter(|v| *v > start);
                match block_end {
                    Some(end) => end,
                    None => lines.last_boundary(start, ceiling).ok_or_else(|| {
                        Error::new(
                            "chunk_limit",
                            format!(
                                "line {} exceeds max_chunk_bytes; refusing a partial-line citation",
                                lines.number(start)
                            ),
                        )
                    })?,
                }
            };
            let line_start = lines.number(start);
            let line_end = lines.number(end.saturating_sub(1));
            let key = format!(
                "c:{}",
                identity(&[
                    &document_key,
                    PARSER_VERSION,
                    CHUNKER_VERSION,
                    &section.heading,
                    &line_start.to_string(),
                    &line_end.to_string(),
                    &source_sha
                ])
            );
            let parent_key = section_keys[index]
                .clone()
                .or_else(|| section.parent.and_then(|p| section_keys[p].clone()));
            if section_keys[index].is_none() {
                section_keys[index] = Some(key.clone());
            }
            chunks.push(Chunk {
                key,
                parent_key,
                heading: section.heading.clone(),
                line_start,
                line_end,
                body: source[start..end].to_owned(),
            });
            start = end;
        }
    }
    Ok(Document {
        root_key: root_key.into(),
        path: relative.into(),
        key: document_key,
        sha256: source_sha,
        mtime,
        size: raw.len() as u64,
        meta,
        chunks,
        links,
    })
}

fn identity(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

struct Heading {
    start: usize,
    level: u8,
    title: String,
}
struct Section {
    start: usize,
    end: usize,
    heading: String,
    parent: Option<usize>,
}

struct Lines {
    starts: Vec<usize>,
    len: usize,
}
impl Lines {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(
            source
                .bytes()
                .enumerate()
                .filter_map(|(i, b)| (b == b'\n' && i + 1 < source.len()).then_some(i + 1)),
        );
        Self {
            starts,
            len: source.len(),
        }
    }
    fn number(&self, offset: usize) -> usize {
        self.starts.partition_point(|start| *start <= offset)
    }
    fn start_at(&self, offset: usize) -> usize {
        self.starts[self.number(offset).saturating_sub(1)]
    }
    // End offsets are exclusive; a block already ending at a new line stays there.
    fn after(&self, end: usize) -> usize {
        if end == 0 {
            return 0;
        }
        let line = self.number(end - 1);
        self.starts.get(line).copied().unwrap_or(self.len)
    }
    fn last_boundary(&self, start: usize, ceiling: usize) -> Option<usize> {
        let index = self.starts.partition_point(|offset| *offset <= ceiling);
        index
            .checked_sub(1)
            .map(|i| self.starts[i])
            .filter(|end| *end > start)
    }
}

fn frontmatter(source: &str, limits: &Limits) -> Result<(Metadata, usize)> {
    let prefix = if source.starts_with('\u{feff}') {
        '\u{feff}'.len_utf8()
    } else {
        0
    };
    let mut lines = source[prefix..].split_inclusive('\n');
    let Some(first) = lines.next() else {
        return Ok((Metadata::default(), 0));
    };
    if first.trim_end_matches('\n') != "---" {
        return Ok((Metadata::default(), 0));
    }
    let yaml_start = prefix + first.len();
    let mut offset = yaml_start;
    for line in lines {
        let marker = line.trim_end_matches('\n');
        if marker == "---" || marker == "..." {
            if offset - yaml_start > limits.max_frontmatter_bytes {
                return Err(Error::new(
                    "frontmatter_limit",
                    "frontmatter exceeds max_frontmatter_bytes",
                ));
            }
            return Ok((metadata(&source[yaml_start..offset])?, offset + line.len()));
        }
        offset += line.len();
        if offset - yaml_start > limits.max_frontmatter_bytes {
            return Err(Error::new(
                "frontmatter_limit",
                "frontmatter exceeds max_frontmatter_bytes",
            ));
        }
    }
    Err(Error::new(
        "invalid_metadata",
        "frontmatter opening delimiter has no closing delimiter",
    ))
}

fn invalid_metadata(message: impl Into<String>) -> Error {
    Error::new("invalid_metadata", message)
}

fn metadata(source: &str) -> Result<Metadata> {
    // Preflight events before the DOM loader: aliases can expand exponentially and
    // a deeply nested tree must never be allocated or recursively dropped.
    let mut depth = 0usize;
    for event in YamlParser::new_from_str(source) {
        let (event, _) = event.map_err(|e| invalid_metadata(e.to_string()))?;
        match event {
            YamlEvent::Alias(_) => return Err(invalid_metadata("YAML aliases are not supported")),
            YamlEvent::SequenceStart(anchor, tag) | YamlEvent::MappingStart(anchor, tag) => {
                if anchor != 0 || tag.is_some() {
                    return Err(invalid_metadata(
                        "YAML anchors and explicit tags are not supported",
                    ));
                }
                depth += 1;
                if depth > MAX_YAML_DEPTH {
                    return Err(Error::new(
                        "frontmatter_limit",
                        "YAML nesting exceeds 16 levels",
                    ));
                }
            }
            YamlEvent::SequenceEnd | YamlEvent::MappingEnd => depth = depth.saturating_sub(1),
            YamlEvent::Scalar(_, _, anchor, tag) if anchor != 0 || tag.is_some() => {
                return Err(invalid_metadata(
                    "YAML anchors and explicit tags are not supported",
                ));
            }
            _ => {}
        }
    }
    let documents = Yaml::load_from_str(source).map_err(|e| invalid_metadata(e.to_string()))?;
    if documents.len() != 1 {
        return Err(invalid_metadata(
            "frontmatter must contain exactly one YAML mapping",
        ));
    }
    let mapping = documents[0]
        .as_mapping()
        .ok_or_else(|| invalid_metadata("frontmatter must be a top-level mapping"))?;
    let mut meta = Metadata::default();
    let mut extra = Map::new();
    for (key, value) in mapping {
        let name = key
            .as_str()
            .ok_or_else(|| invalid_metadata("metadata keys must be strings"))?;
        match name {
            "id" => {
                meta.id = string(value, name)?;
                validate_id(&meta.id)?;
            }
            "title" => meta.title = string(value, name)?,
            "type" => meta.doc_type = string(value, name)?,
            "status" => meta.status = string(value, name)?,
            "tags" => meta.tags = strings(value, name)?,
            "aliases" => meta.aliases = strings(value, name)?,
            "depends_on" => {
                meta.depends_on = strings(value, name)?;
                for id in &meta.depends_on {
                    validate_id(id)?;
                }
            }
            "supersedes" => {
                meta.supersedes = strings(value, name)?;
                for id in &meta.supersedes {
                    validate_id(id)?;
                }
            }
            "conflicts" => {
                meta.conflicts = strings(value, name)?;
                for id in &meta.conflicts {
                    validate_id(id)?;
                }
            }
            "updated" => {
                let updated = string(value, name)?;
                if !valid_date(&updated) {
                    return Err(invalid_metadata("updated must be a valid YYYY-MM-DD date"));
                }
                meta.updated = Some(updated);
            }
            _ => {
                extra.insert(name.into(), extra_value(value)?);
            }
        }
    }
    meta.extra = Value::Object(extra);
    Ok(meta)
}

fn string(value: &Yaml<'_>, name: &str) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid_metadata(format!("{name} must be a string")))
}
fn strings(value: &Yaml<'_>, name: &str) -> Result<Vec<String>> {
    value
        .as_sequence()
        .ok_or_else(|| invalid_metadata(format!("{name} must be a sequence of strings")))?
        .iter()
        .map(|value| string(value, name))
        .collect()
}
fn validate_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 256
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid_metadata(
            "IDs must be 1..256 ASCII letters, digits, hyphens, underscores, dots or colons",
        ));
    }
    Ok(())
}
fn extra_scalar(value: &Yaml<'_>) -> Result<Value> {
    if let Some(v) = value.as_str() {
        return Ok(Value::String(v.into()));
    }
    if let Some(v) = value.as_bool() {
        return Ok(Value::Bool(v));
    }
    if let Some(v) = value.as_integer() {
        return Ok(Value::from(v));
    }
    if let Some(v) = value.as_floating_point() {
        return serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| invalid_metadata("metadata numbers must be finite"));
    }
    if value.is_null() {
        return Ok(Value::Null);
    }
    Err(invalid_metadata(
        "unknown metadata fields support only scalar values or flat scalar lists",
    ))
}
fn extra_value(value: &Yaml<'_>) -> Result<Value> {
    if let Some(values) = value.as_sequence() {
        return values
            .iter()
            .map(extra_scalar)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array);
    }
    extra_scalar(value)
}
fn valid_date(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() != 10
        || b[4] != b'-'
        || b[7] != b'-'
        || !b
            .iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
    {
        return false;
    }
    let year = value[..4].parse::<u32>().unwrap_or(0);
    let month = value[5..7].parse::<u32>().unwrap_or(0);
    let day = value[8..].parse::<u32>().unwrap_or(0);
    let max = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    };
    year != 0 && day != 0 && day <= max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Root;
    use std::collections::BTreeMap;

    fn parsed(source: &str) -> Document {
        parse(
            "notes",
            "folder/test.md",
            source.as_bytes(),
            0.0,
            &Limits::default(),
        )
        .unwrap()
    }
    fn assert_exact(source: &str, document: &Document) {
        let normalized = normalized_source(source.as_bytes()).unwrap();
        let lines = Lines::new(&normalized);
        for chunk in &document.chunks {
            let start = lines.starts[chunk.line_start - 1];
            let end = lines
                .starts
                .get(chunk.line_end)
                .copied()
                .unwrap_or(normalized.len());
            assert_eq!(chunk.body, normalized[start..end]);
        }
    }

    #[test]
    fn crlf_frontmatter_and_unicode_preserve_whole_file_lines() {
        let source = "---\r\nid: P-1\r\ntags: [rust, rust]\r\ncustom: [true, 3, café]\r\n---\r\n# Héading\r\nα text\r\n## Child\r\n[link](other.md)\r\n";
        let doc = parsed(source);
        assert_eq!(doc.sha256, digest(source.as_bytes()));
        assert_eq!(doc.meta.tags, ["rust", "rust"]);
        assert_eq!(doc.meta.extra["custom"][2], "café");
        assert_eq!(doc.chunks[0].line_start, 6);
        assert_eq!(doc.chunks[0].line_end, 7);
        assert_eq!(doc.chunks[1].heading, "Héading / Child");
        assert_eq!(doc.chunks[1].parent_key.as_ref(), Some(&doc.chunks[0].key));
        assert_eq!(doc.links[0].line, 9);
        assert_exact(source, &doc);
    }

    #[test]
    fn code_examples_are_not_headings_or_links_but_real_commonmark_is() {
        let source = "# Real\n```md\n# Fake\n[bad](missing.md)\n```\n\n    # Indented\n    [bad](also-missing.md)\n\n`[bad](inline.md)`\n\n[reference][r] and <https://example.com>\n\n[r]: target.md\n\nSetext *heading*\n---------------\ntext\n";
        let doc = parsed(source);
        assert_eq!(
            doc.links
                .iter()
                .map(|v| v.target.as_str())
                .collect::<Vec<_>>(),
            ["target.md", "https://example.com"]
        );
        assert_eq!(doc.links[0].line, 12);
        assert_eq!(doc.chunks.len(), 2);
        assert_eq!(doc.chunks[1].heading, "Real / Setext heading");
        assert_exact(source, &doc);
    }

    #[test]
    fn tables_skipped_heading_levels_and_unterminated_last_line() {
        let source = "# Parent\n### Deep\n| Link | Text |\n| --- | --- |\n| [target](table.md) | `# not heading` |\n## Sibling\nlast line";
        let doc = parsed(source);
        assert_eq!(doc.links[0].target, "table.md");
        assert_eq!(doc.links[0].line, 5);
        assert_eq!(doc.chunks[1].heading, "Parent / Deep");
        assert_eq!(doc.chunks[2].heading, "Parent / Sibling");
        assert_eq!(doc.chunks[1].parent_key, doc.chunks[2].parent_key);
        assert_eq!(doc.chunks[2].line_end, 7);
        assert!(doc.chunks[2].body.ends_with("last line"));
        assert_exact(source, &doc);
    }

    #[test]
    fn empty_and_bare_carriage_return_sources_are_precise() {
        assert!(parsed("").chunks.is_empty());
        let doc = parsed("# Heading\rbody\rlast");
        assert_eq!(doc.chunks[0].body, "# Heading\nbody\nlast");
        assert_eq!(doc.chunks[0].line_end, 3);
        assert_exact("# Heading\rbody\rlast", &doc);
    }

    #[test]
    fn safe_block_splits_and_oversized_multiline_fallback_are_exact() {
        let source =
            "# Heading\n\nfirst paragraph\n\nsecond paragraph\n\n```txt\nααα\nβββ\nγγγ\n```\n";
        let mut limits = Limits::default();
        limits.max_chunk_bytes = 24;
        let doc = parse("r", "a.md", source.as_bytes(), 0.0, &limits).unwrap();
        assert!(doc.chunks.len() > 2);
        assert!(
            doc.chunks
                .iter()
                .all(|c| c.body.len() <= limits.max_chunk_bytes)
        );
        assert!(
            doc.chunks[1..]
                .iter()
                .all(|c| c.parent_key.as_ref() == Some(&doc.chunks[0].key))
        );
        assert_eq!(
            doc.chunks
                .iter()
                .map(|c| c.body.as_str())
                .collect::<String>(),
            source
        );
        assert_exact(source, &doc);
    }

    #[test]
    fn oversized_single_unicode_line_is_rejected_not_truncated() {
        let mut limits = Limits::default();
        limits.max_chunk_bytes = 10;
        let error = parse("r", "a.md", "界界界界\n".as_bytes(), 0.0, &limits).unwrap_err();
        assert_eq!(error.code, "chunk_limit");
        assert!(error.message.contains("r:a.md"));
    }

    #[test]
    fn metadata_requires_mapping_and_strict_known_field_types() {
        for yaml in [
            "- x",
            "id: 123",
            "tags: one",
            "tags: [ok, 1]",
            "updated: 2025-02-29",
            "unknown: {nested: map}",
            "id: one\nid: two",
            "id: '../bad'",
            "x: &a [one]\ny: *a",
            "id: !!str 1",
        ] {
            let source = format!("---\n{yaml}\n---\nbody\n");
            assert!(
                parse("r", "a.md", source.as_bytes(), 0.0, &Limits::default()).is_err(),
                "accepted {yaml}"
            );
        }
        assert_eq!(
            parsed("---\nupdated: 2024-02-29\naliases: [a, a]\n---\ntext")
                .meta
                .aliases,
            ["a", "a"]
        );
    }

    #[test]
    fn frontmatter_only_at_start_and_bounded_before_loading() {
        let doc = parsed("intro\n\n---\nid: not-metadata\n---\n");
        assert!(doc.meta.id.is_empty());
        assert!(parsed("\u{feff}---\nid: bom\n---\n# Title\n").meta.id == "bom");
        let mut limits = Limits::default();
        limits.max_frontmatter_bytes = 4;
        assert_eq!(
            parse("r", "a.md", b"---\nid: long\n---\n", 0.0, &limits)
                .unwrap_err()
                .code,
            "frontmatter_limit"
        );
        let nested = format!("---\nx: {}0{}\n---\n", "[".repeat(20), "]".repeat(20));
        assert_eq!(
            parse("r", "a.md", nested.as_bytes(), 0.0, &Limits::default())
                .unwrap_err()
                .code,
            "frontmatter_limit"
        );
        assert_eq!(
            parse("r", "a.md", b"---\nid: no-close", 0.0, &Limits::default())
                .unwrap_err()
                .code,
            "invalid_metadata"
        );
    }

    #[test]
    fn invalid_utf8_paths_and_file_limit_are_diagnostics() {
        let limits = Limits::default();
        assert_eq!(
            parse("r", "a.md", &[0xff], 0.0, &limits).unwrap_err().code,
            "invalid_utf8"
        );
        for path in [
            "../a.md",
            "a/../b.md",
            "/a.md",
            "C:/a.md",
            "a\\b.md",
            "a//b.md",
            "./a.md",
        ] {
            assert_eq!(
                parse("r", path, b"text", 0.0, &limits).unwrap_err().code,
                "invalid_path"
            );
        }
        let mut small = limits;
        small.max_file_bytes = 2;
        assert_eq!(
            parse("r", "a.md", b"text", 0.0, &small).unwrap_err().code,
            "file_limit"
        );
    }

    #[test]
    fn stable_keys_distinguish_duplicate_ids_roots_and_source_versions() {
        let first = parsed("---\nid: SAME\n---\n# Title\none\n");
        let other = parse(
            "notes",
            "other.md",
            b"---\nid: SAME\n---\n# Title\none\n",
            0.0,
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(first.meta.id, other.meta.id);
        assert_ne!(first.key, other.key);
        let changed = parsed("---\nid: SAME\n---\n# Title\ntwo\n");
        assert_eq!(first.key, changed.key);
        assert_ne!(first.chunks[0].key, changed.chunks[0].key);
        assert_eq!(
            first.chunks[0].key,
            parsed("---\nid: SAME\n---\n# Title\none\n").chunks[0].key
        );
    }

    #[test]
    fn scan_is_root_only_sorted_ignored_and_atomic_on_bad_source() {
        let temp = tempfile::tempdir().unwrap();
        let root_path = temp.path().join("notes");
        fs::create_dir(&root_path).unwrap();
        fs::create_dir(root_path.join("ignored")).unwrap();
        fs::write(root_path.join("ignored/invalid.md"), [0xff]).unwrap();
        fs::write(root_path.join("z.MD"), "# Z\n").unwrap();
        fs::write(root_path.join("a.md"), "# A\n").unwrap();
        fs::write(temp.path().join("outside.md"), "# Outside\n").unwrap();
        let config = Config {
            database: temp.path().join("index.sqlite"),
            roots: BTreeMap::from([(
                "r".into(),
                Root {
                    path: root_path.canonicalize().unwrap(),
                    role: "notes".into(),
                    trust: "configured".into(),
                    ignore_dirs: vec!["ignored".into()],
                },
            )]),
            limits: Limits::default(),
            profile: Default::default(),
            full: None,
        };
        let docs = scan(&config).unwrap();
        assert_eq!(
            docs.iter().map(|v| v.path.as_str()).collect::<Vec<_>>(),
            ["a.md", "z.MD"]
        );
        let cached = scan_cached(&config, &docs).unwrap();
        assert_eq!(cached[0].sha256, docs[0].sha256);
        assert_eq!(cached[0].chunks[0].key, docs[0].chunks[0].key);
        fs::write(root_path.join("a.md"), "# Edited\n").unwrap();
        let changed = scan_cached(&config, &docs).unwrap();
        assert_ne!(changed[0].sha256, docs[0].sha256);
        assert_ne!(changed[0].chunks[0].key, docs[0].chunks[0].key);
        fs::write(root_path.join("bad.md"), [0xff]).unwrap();
        assert_eq!(scan(&config).unwrap_err().code, "invalid_utf8");
    }

    #[cfg(unix)]
    #[test]
    fn scan_never_follows_file_or_directory_symlinks() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        let outside = temp.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret.md"), "outside").unwrap();
        symlink(&outside, root.join("escape")).unwrap();
        symlink(outside.join("secret.md"), root.join("linked.md")).unwrap();
        fs::write(root.join("real.md"), "inside").unwrap();
        let config = Config {
            database: temp.path().join("index.sqlite"),
            roots: BTreeMap::from([(
                "r".into(),
                Root {
                    path: root.canonicalize().unwrap(),
                    role: "notes".into(),
                    trust: "configured".into(),
                    ignore_dirs: vec![],
                },
            )]),
            limits: Limits::default(),
            profile: Default::default(),
            full: None,
        };
        assert_eq!(scan(&config).unwrap().len(), 1);
    }
}
