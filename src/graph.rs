use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    config::Config,
    db,
    error::{Error, Result},
    ingest,
    model::{Document, Envelope},
};

#[derive(Debug, Serialize)]
struct Edge {
    source: String,
    kind: &'static str,
    target: String,
    line: Option<usize>,
    target_keys: Vec<String>,
    resolution: String,
}

#[derive(Default)]
struct IdentityIndex<'a> {
    ids: BTreeMap<&'a str, Vec<&'a Document>>,
    paths: BTreeMap<(&'a str, &'a str), &'a Document>,
    keys: BTreeMap<&'a str, &'a Document>,
}

impl<'a> IdentityIndex<'a> {
    fn new(docs: &'a [Document]) -> Self {
        let mut index = Self::default();
        for doc in docs {
            if !doc.meta.id.is_empty() {
                index.ids.entry(&doc.meta.id).or_default().push(doc);
            }
            index.paths.insert((&doc.root_key, &doc.path), doc);
            index.keys.insert(&doc.key, doc);
        }
        index
    }

    fn resolve_identity(&self, target: &str) -> Vec<&'a Document> {
        if let Some(doc) = self.keys.get(target) {
            return vec![*doc];
        }
        if let Some((root, path)) = target.split_once(':') {
            if let Some(doc) = self.paths.get(&(root, path)) {
                return vec![*doc];
            }
        }
        if let Some(docs) = self.ids.get(target) {
            return docs.clone();
        }
        self.paths
            .iter()
            .filter_map(|((_, path), doc)| (*path == target).then_some(*doc))
            .collect()
    }
}

fn metadata_envelope(config: &Config, generation: i64, data: Value) -> Envelope<Value> {
    let mut result = config.envelope(generation, data);
    result.retrieval_legs.clear();
    result
}

fn snapshot(config: &Config) -> Result<(i64, Vec<Document>)> {
    let conn = db::open_reader(config)?;
    let generation = db::generation(&conn)?;
    let docs = db::load_documents(&conn)?;
    Ok((generation, docs))
}

// Decode before containment checks: encoded dot segments and separators must not bypass them.
fn decode_path(value: &str) -> Result<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] == b'%' {
            let pair = bytes
                .get(cursor + 1..cursor + 3)
                .ok_or_else(|| Error::new("invalid_link", "Incomplete percent escape"))?;
            let hex = std::str::from_utf8(pair)
                .map_err(|_| Error::new("invalid_link", "Invalid percent escape"))?;
            decoded.push(
                u8::from_str_radix(hex, 16)
                    .map_err(|_| Error::new("invalid_link", "Invalid percent escape"))?,
            );
            cursor += 3;
        } else {
            decoded.push(bytes[cursor]);
            cursor += 1;
        }
    }
    let path = String::from_utf8(decoded)
        .map_err(|_| Error::new("invalid_link", "Link path is not UTF-8"))?;
    if path.contains('\0') || path.contains('\\') {
        return Err(Error::new(
            "invalid_link",
            "NUL and backslash are not supported in Markdown link paths",
        ));
    }
    Ok(path)
}

fn relative_link(base: &str, target: &str) -> Result<String> {
    if target.starts_with('/') {
        return Err(Error::new(
            "path_escape",
            "Absolute link paths are outside root-relative Markdown scope",
        ));
    }
    let mut parts: Vec<&str> = base
        .rsplit_once('/')
        .map(|(parent, _)| parent.split('/').collect())
        .unwrap_or_default();
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(Error::new(
                        "path_escape",
                        "Link traverses outside its configured root",
                    ));
                }
            }
            _ if part.contains(':') => {
                return Err(Error::new(
                    "invalid_link",
                    "Colon in a local path is not supported",
                ));
            }
            _ => parts.push(part),
        }
    }
    Ok(parts.join("/"))
}

fn local_target(config: &Config, doc: &Document, target: &str) -> Result<Option<(String, String)>> {
    let target = target.trim();
    if target.is_empty() || target.starts_with('#') || target.starts_with("//") {
        return Ok(None);
    }
    let target = target.split(['#', '?']).next().unwrap_or("");
    if let Some((scheme, remainder)) = target.split_once(':') {
        if config.roots.contains_key(scheme) {
            let path = decode_path(remainder)?;
            // Explicit cross-root links start at the selected root, not the source parent.
            return Ok(Some((scheme.to_owned(), relative_link("", &path)?)));
        }
        if scheme.len() == 1 && scheme.as_bytes()[0].is_ascii_alphabetic() {
            return Err(Error::new(
                "path_escape",
                "Drive-qualified local links are not allowed",
            ));
        }
        if scheme
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        {
            return Ok(None); // RFC-style external URI, not an index dependency.
        }
    }
    let path = decode_path(target)?;
    Ok(Some((
        doc.root_key.clone(),
        relative_link(&doc.path, &path)?,
    )))
}

// Link validation checks existence and containment, not the narrower source-read
// policy: directory navigation is valid Markdown but is never readable through get.
fn resolve_link(config: &Config, root: &str, relative: &str) -> Result<std::path::PathBuf> {
    let policy = config
        .roots
        .get(root)
        .ok_or_else(|| Error::new("inaccessible_path", "unknown link root"))?;
    let canonical = policy
        .path
        .join(relative)
        .canonicalize()
        .map_err(|_| Error::new("missing_link", "local link target is unavailable"))?;
    if !canonical.starts_with(&policy.path) || (!canonical.is_file() && !canonical.is_dir()) {
        return Err(Error::new(
            "inaccessible_path",
            "link target escapes its configured root or has unsupported file type",
        ));
    }
    Ok(canonical)
}

fn build_edges(
    config: &Config,
    docs: &[Document],
    index: &IdentityIndex<'_>,
) -> (Vec<Edge>, Vec<Value>) {
    let mut edges = Vec::new();
    let mut diagnostics = Vec::new();
    for doc in docs {
        for (kind, targets) in [
            ("depends_on", &doc.meta.depends_on),
            ("supersedes", &doc.meta.supersedes),
            ("conflicts", &doc.meta.conflicts),
        ] {
            for target in targets {
                let candidates = index.resolve_identity(target);
                let resolution = match candidates.len() {
                    0 => "unknown",
                    1 => "resolved",
                    _ => "ambiguous",
                };
                if resolution != "resolved" {
                    diagnostics.push(json!({"code": if resolution == "unknown" { "unknown_relation" } else { "ambiguous_relation" },
                        "source": doc.key, "root":doc.root_key, "path":doc.path, "kind":kind, "target":target,
                        "message":format!("{kind} target {target:?} is {resolution}")}));
                }
                edges.push(Edge {
                    source: doc.key.clone(),
                    kind,
                    target: target.clone(),
                    line: None,
                    target_keys: candidates.iter().map(|d| d.key.clone()).collect(),
                    resolution: resolution.into(),
                });
            }
        }
        for link in &doc.links {
            let local = match local_target(config, doc, &link.target) {
                Ok(None) => continue,
                Ok(Some(local)) => local,
                Err(error) => {
                    diagnostics.push(json!({"code":error.code,"message":error.message,"source":doc.key,"root":doc.root_key,"path":doc.path,"target":link.target,"line":link.line}));
                    edges.push(Edge {
                        source: doc.key.clone(),
                        kind: "link",
                        target: link.target.clone(),
                        line: Some(link.line),
                        target_keys: Vec::new(),
                        resolution: "invalid".into(),
                    });
                    continue;
                }
            };
            let (root, path) = local;
            let resolved = resolve_link(config, &root, &path);
            let resolution = match &resolved {
                Ok(_) => "resolved",
                Err(error) => {
                    diagnostics.push(json!({"code":"broken_link","cause":error.code,"message":error.message,"source":doc.key,"root":doc.root_key,"path":doc.path,"target":link.target,"line":link.line}));
                    "broken"
                }
            };
            let target_keys = index
                .paths
                .get(&(root.as_str(), path.as_str()))
                .map(|d| vec![d.key.clone()])
                .unwrap_or_default();
            edges.push(Edge {
                source: doc.key.clone(),
                kind: "link",
                target: link.target.clone(),
                line: Some(link.line),
                target_keys,
                resolution: resolution.into(),
            });
        }
    }
    (edges, diagnostics)
}

// Iterative DFS avoids recursion proportional to the corpus size. Report concrete cycle paths,
// not Kahn leftovers (which incorrectly label nodes downstream from a cycle).
fn dependency_cycles(docs: &[Document], edges: &[Edge]) -> Vec<Vec<String>> {
    let mut adjacency: BTreeMap<&str, Vec<&str>> =
        docs.iter().map(|d| (d.key.as_str(), Vec::new())).collect();
    for edge in edges
        .iter()
        .filter(|e| e.kind == "depends_on" && e.resolution == "resolved")
    {
        adjacency
            .entry(&edge.source)
            .or_default()
            .extend(edge.target_keys.iter().map(String::as_str));
    }
    let mut color: BTreeMap<&str, u8> = BTreeMap::new();
    let mut cycles = BTreeSet::new();
    for &start in adjacency.keys() {
        if color.get(start).copied().unwrap_or(0) != 0 {
            continue;
        }
        color.insert(start, 1);
        let mut stack = vec![(start, 0usize)];
        let mut positions = BTreeMap::from([(start, 0usize)]);
        while let Some(&(node, next)) = stack.last() {
            let neighbors = &adjacency[node];
            if next == neighbors.len() {
                color.insert(node, 2);
                positions.remove(node);
                stack.pop();
                continue;
            }
            stack.last_mut().unwrap().1 += 1;
            let target = neighbors[next];
            match color.get(target).copied().unwrap_or(0) {
                0 => {
                    color.insert(target, 1);
                    positions.insert(target, stack.len());
                    stack.push((target, 0));
                }
                1 => {
                    let begin = positions[target];
                    let mut cycle: Vec<String> = stack[begin..]
                        .iter()
                        .map(|(key, _)| (*key).to_owned())
                        .collect();
                    let smallest = cycle
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, key)| *key)
                        .map(|(i, _)| i)
                        .unwrap();
                    cycle.rotate_left(smallest);
                    cycle.push(cycle[0].clone());
                    cycles.insert(cycle);
                }
                _ => {}
            }
        }
    }
    cycles.into_iter().collect()
}

pub fn graph(config: &Config) -> Result<Envelope<Value>> {
    let (generation, docs) = snapshot(config)?;
    let index = IdentityIndex::new(&docs);
    let (edges, diagnostics) = build_edges(config, &docs, &index);
    let nodes: Vec<Value> = docs.iter().map(|d| json!({"key":d.key,"id":d.meta.id,"root":d.root_key,"path":d.path,"title":d.meta.title,"status":d.meta.status})).collect();
    Ok(metadata_envelope(
        config,
        generation,
        json!({"nodes":nodes,"edges":edges,"diagnostics":diagnostics}),
    ))
}

pub fn backlinks(config: &Config, target: &str) -> Result<Envelope<Value>> {
    let (generation, docs) = snapshot(config)?;
    let index = IdentityIndex::new(&docs);
    let targets = index.resolve_identity(target);
    if targets.is_empty() {
        return Err(Error::new(
            "not_found",
            format!("No indexed source matches {target:?}"),
        ));
    }
    let keys: BTreeSet<&str> = targets.iter().map(|d| d.key.as_str()).collect();
    let (edges, _) = build_edges(config, &docs, &index);
    let inbound: Vec<&Edge> = edges
        .iter()
        .filter(|e| e.target_keys.iter().any(|key| keys.contains(key.as_str())))
        .collect();
    Ok(metadata_envelope(
        config,
        generation,
        json!({"target":target,"ambiguous":targets.len()>1,"target_keys":keys,"backlinks":inbound}),
    ))
}

pub fn check(config: &Config) -> Result<Envelope<Value>> {
    let (generation, docs) = snapshot(config)?;
    let index = IdentityIndex::new(&docs);
    let (edges, mut errors) = build_edges(config, &docs, &index);
    let mut warnings = Vec::new();
    for (id, candidates) in &index.ids {
        if candidates.len() > 1 {
            errors.push(json!({"code":"duplicate_id","id":id,"source_keys":candidates.iter().map(|d| &d.key).collect::<Vec<_>>(),"message":"IDs must be unique; all indexed identities have been preserved"}));
        }
    }
    for cycle in dependency_cycles(&docs, &edges) {
        errors.push(json!({"code":"dependency_cycle","source_keys":cycle,"message":"Dependency graph contains a directed cycle"}));
    }
    match ingest::scan_cached(config, &docs) {
        Err(error) => errors
            .push(json!({"code":"source_scan_failed","cause":error.code,"message":error.message})),
        Ok(current) => {
            let current_by_key: BTreeMap<&str, &Document> =
                current.iter().map(|d| (d.key.as_str(), d)).collect();
            for doc in &docs {
                match current_by_key.get(doc.key.as_str()) {
                    None => errors.push(json!({"code":"removed_source","source":doc.key,"root":doc.root_key,"path":doc.path,"message":"Indexed source is no longer in the configured source set; run index"})),
                    Some(now) if now.sha256 != doc.sha256 => errors.push(json!({"code":"stale_source","source":doc.key,"root":doc.root_key,"path":doc.path,"indexed_sha256":doc.sha256,"current_sha256":now.sha256,"message":"Source differs from its indexed content; run index"})),
                    _ => {},
                }
            }
            for doc in &current {
                if !index.keys.contains_key(doc.key.as_str()) {
                    errors.push(json!({"code":"unindexed_source","source":doc.key,"root":doc.root_key,"path":doc.path,"message":"New source is not indexed; run index"}));
                }
            }
        }
    }
    if docs.is_empty() {
        warnings.push(json!({"code":"empty_index","message":"No documents are indexed"}));
    }
    Ok(metadata_envelope(
        config,
        generation,
        json!({"errors":errors,"warnings":warnings,"documents_checked":docs.len(),"edges_checked":edges.len()}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Metadata;

    fn document(key: &str, id: &str) -> Document {
        Document {
            root_key: "test".into(),
            path: format!("{key}.md"),
            key: key.into(),
            sha256: "hash".into(),
            mtime: 0.0,
            size: 0,
            meta: Metadata {
                id: id.into(),
                ..Metadata::default()
            },
            chunks: vec![],
            links: vec![],
        }
    }

    #[test]
    fn duplicate_ids_keep_all_source_identities() {
        let docs = vec![document("a", "DUP"), document("b", "DUP")];
        let index = IdentityIndex::new(&docs);
        assert_eq!(index.resolve_identity("DUP").len(), 2);
        assert_eq!(index.resolve_identity("test:a.md")[0].key, "a");
        assert_eq!(index.resolve_identity("b")[0].path, "b.md");
        assert!(index.resolve_identity("unknown").is_empty());
    }

    #[test]
    fn relative_paths_normalize_without_escaping() {
        assert_eq!(relative_link("notes/a.md", "../b.md").unwrap(), "b.md");
        assert_eq!(relative_link("a.md", "./b.md").unwrap(), "b.md");
        assert!(relative_link("a.md", "../secret.md").is_err());
        assert!(relative_link("a.md", "/secret.md").is_err());
        assert!(relative_link("a.md", "C:/secret.md").is_err());
    }

    #[test]
    fn encoded_traversal_is_checked_after_decoding() {
        let decoded = decode_path("%2e%2e/%2e%2e/private.md").unwrap();
        assert!(relative_link("notes/a.md", &decoded).is_err());
        assert_eq!(decode_path("hello%20world.md").unwrap(), "hello world.md");
        assert!(decode_path("bad%xx").is_err());
        assert!(decode_path("bad%00.md").is_err());
        assert!(decode_path("..%5csecret").is_err());
    }

    #[test]
    fn cycles_do_not_include_downstream_nodes() {
        let docs = vec![document("a", "A"), document("b", "B"), document("c", "C")];
        let edge = |source: &str, target: &str| Edge {
            source: source.into(),
            kind: "depends_on",
            target: target.into(),
            line: None,
            target_keys: vec![target.into()],
            resolution: "resolved".into(),
        };
        let edges = vec![edge("a", "b"), edge("b", "a"), edge("b", "c")];
        assert_eq!(
            dependency_cycles(&docs, &edges),
            vec![vec!["a".to_string(), "b".to_string(), "a".to_string()]]
        );
        assert_eq!(
            dependency_cycles(&docs, &[edge("c", "c")]),
            vec![vec!["c".to_string(), "c".to_string()]]
        );
    }

    fn test_config(database: std::path::PathBuf, roots: Vec<(&str, std::path::PathBuf)>) -> Config {
        Config {
            database,
            roots: roots
                .into_iter()
                .map(|(key, path)| {
                    (
                        key.into(),
                        crate::config::Root {
                            path: path.canonicalize().unwrap(),
                            role: key.into(),
                            trust: "synthetic".into(),
                            ignore_dirs: vec![],
                        },
                    )
                })
                .collect(),
            limits: Default::default(),
            profile: Default::default(),
            full: None,
        }
    }

    #[test]
    fn parsed_links_and_metadata_backlinks_use_real_sources() {
        let temp = tempfile::tempdir().unwrap();
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let config = test_config(
            temp.path().join("index.sqlite"),
            ["canonical", "research", "audit"]
                .map(|key| (key, fixtures.join(key)))
                .to_vec(),
        );
        db::index(&config, false).unwrap();
        let checked = check(&config).unwrap();
        assert_eq!(checked.data["errors"], json!([]));
        let graph = graph(&config).unwrap();
        assert_eq!(graph.data["nodes"].as_array().unwrap().len(), 16);
        assert!(
            !graph.data["edges"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["target"] == "missing-example.md")
        );
        let inbound = backlinks(&config, "SYS-VAULT").unwrap();
        let links = inbound.data["backlinks"].as_array().unwrap();
        assert!(links.iter().any(|e| e["kind"] == "depends_on"));
        assert!(
            links
                .iter()
                .any(|e| e["kind"] == "link" && e["target"] == "vault.md")
        );
        assert!(backlinks(&config, "SYS-VAUL").is_err());
    }

    #[test]
    fn directory_links_are_valid_without_granting_source_read_access() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("notes");
        std::fs::create_dir_all(root.join("guides")).unwrap();
        std::fs::write(
            root.join("a.md"),
            "# Navigation\n[Guides](guides/)\n[Root](.)\n",
        )
        .unwrap();
        let config = test_config(temp.path().join("index.sqlite"), vec![("notes", root)]);
        db::index(&config, false).unwrap();
        assert_eq!(check(&config).unwrap().data["errors"], json!([]));
        assert!(config.resolve_source("notes", "guides").is_err());
    }

    #[test]
    fn checks_report_duplicates_cycles_unknown_and_broken_targets() {
        let temp = tempfile::tempdir().unwrap();
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/graph-errors");
        let config = test_config(temp.path().join("index.sqlite"), vec![("errors", fixtures)]);
        db::index(&config, false).unwrap();
        let result = check(&config).unwrap();
        let codes: BTreeSet<&str> = result.data["errors"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e["code"].as_str())
            .collect();
        for code in [
            "duplicate_id",
            "dependency_cycle",
            "unknown_relation",
            "ambiguous_relation",
            "path_escape",
            "broken_link",
        ] {
            assert!(codes.contains(code), "Missing {code}: {:?}", result.data);
        }
        let inbound = backlinks(&config, "DUPLICATE").unwrap();
        assert_eq!(inbound.data["ambiguous"], true);
        assert_eq!(inbound.data["target_keys"].as_array().unwrap().len(), 2);
        assert_eq!(
            graph(&config).unwrap().data["nodes"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn checks_compare_the_entire_current_source_set() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("sources");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("a.md"), "---\nid: A\n---\n# Alpha\nInitial.\n").unwrap();
        std::fs::write(root.join("b.md"), "---\nid: B\n---\n# Beta\nInitial.\n").unwrap();
        let config = test_config(
            temp.path().join("index.sqlite"),
            vec![("test", root.clone())],
        );
        db::index(&config, false).unwrap();
        std::fs::write(root.join("a.md"), "---\nid: A\n---\n# Alpha\nChanged.\n").unwrap();
        std::fs::remove_file(root.join("b.md")).unwrap();
        std::fs::write(root.join("c.md"), "---\nid: C\n---\n# Gamma\nNew.\n").unwrap();
        let result = check(&config).unwrap();
        let codes: BTreeSet<&str> = result.data["errors"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e["code"].as_str())
            .collect();
        for code in ["stale_source", "removed_source", "unindexed_source"] {
            assert!(codes.contains(code));
        }
    }
}
