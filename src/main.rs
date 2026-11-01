use clap::{Args, Parser, Subcommand};
use kb::{
    config::Config,
    error::{Error, Result},
    model::{ContextRequest, Filters, GetRequest, SearchRequest},
};
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Markdown retrieval (light by default); opt-in full RAG"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    database: Option<PathBuf>,
    #[arg(long = "root", global = true, value_name = "KEY=ABSOLUTE_PATH")]
    roots: Vec<String>,
    #[arg(long, global = true)]
    profile: Option<String>,
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Index {
        #[arg(long)]
        force: bool,
    },
    Search {
        query: String,
        #[command(flatten)]
        filters: FilterArgs,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Get {
        key: String,
        #[arg(long)]
        line_start: Option<usize>,
        #[arg(long)]
        line_end: Option<usize>,
    },
    Context {
        query: String,
        #[command(flatten)]
        filters: FilterArgs,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long = "budget-bytes", alias = "budget", default_value_t = 16384)]
        budget_bytes: usize,
    },
    Backlinks {
        target: String,
    },
    Graph {
        #[arg(long,default_value="json",value_parser=["json","mermaid"])]
        format: String,
    },
    Check,
    Stats,
    Eval {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long, conflicts_with = "compare")]
        answers: bool,
        #[arg(long, conflicts_with = "answers")]
        compare: bool,
        #[arg(long, requires = "answers")]
        release_manifest: Option<PathBuf>,
    },
    Answer {
        question: String,
        #[command(flatten)]
        filters: FilterArgs,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long = "budget-bytes", default_value_t = 16384)]
        budget_bytes: usize,
    },
    Mcp,
}
#[derive(Args)]
struct FilterArgs {
    #[arg(long = "root-filter")]
    root: Option<String>,
    #[arg(long)]
    role: Option<String>,
    #[arg(long)]
    status: Option<String>,
    #[arg(long)]
    project: Option<String>,
    #[arg(long = "type")]
    doc_type: Option<String>,
    #[arg(long)]
    tag: Option<String>,
    #[arg(long)]
    after: Option<String>,
    #[arg(long)]
    before: Option<String>,
}
impl From<FilterArgs> for Filters {
    fn from(f: FilterArgs) -> Self {
        Self {
            root: f.root,
            role: f.role,
            status: f.status,
            project: f.project,
            doc_type: f.doc_type,
            tag: f.tag,
            after: f.after,
            before: f.before,
        }
    }
}
fn value<T: Serialize>(v: T) -> Result<Value> {
    Ok(serde_json::to_value(v)?)
}
fn execute(cli: Cli) -> Result<bool> {
    if matches!(cli.command, Command::Mcp)
        && cli.config.is_none()
        && std::env::var_os("KB_CONFIG").is_none()
    {
        return Err(Error::new(
            "configuration_required",
            "MCP requires --config or KB_CONFIG at startup",
        ));
    }
    let config = Config::load(cli.config.as_deref(), cli.database, &cli.roots, cli.profile)?;
    let mut failure = false;
    let output = match cli.command {
        Command::Index { force } => value(kb::db::index(&config, force)?)?,
        Command::Search {
            query,
            filters,
            limit,
        } => value(kb::retrieve::search(
            &config,
            &SearchRequest {
                query,
                filters: filters.into(),
                limit,
            },
        )?)?,
        Command::Get {
            key,
            line_start,
            line_end,
        } => value(kb::retrieve::get(
            &config,
            &GetRequest {
                key,
                line_start,
                line_end,
            },
        )?)?,
        Command::Context {
            query,
            filters,
            limit,
            budget_bytes,
        } => value(kb::context::pack(
            &config,
            &ContextRequest {
                query,
                filters: filters.into(),
                limit,
                budget_bytes,
            },
        )?)?,
        Command::Backlinks { target } => value(kb::graph::backlinks(&config, &target)?)?,
        Command::Graph { format } => {
            let graph = value(kb::graph::graph(&config)?)?;
            if format == "mermaid" && !cli.json {
                render_mermaid(&graph);
                return Ok(false);
            }
            graph
        }
        Command::Check => {
            let checked = value(kb::graph::check(&config)?)?;
            failure = checked["data"]["errors"]
                .as_array()
                .is_some_and(|v| !v.is_empty());
            checked
        }
        Command::Stats => value(kb::db::stats(&config)?)?,
        Command::Eval {
            fixture,
            answers,
            compare,
            release_manifest,
        } => {
            if answers {
                #[cfg(feature = "full")]
                {
                    value(kb::answer_eval::run(
                        &config,
                        &fixture,
                        release_manifest.as_deref(),
                    )?)?
                }
                #[cfg(not(feature = "full"))]
                {
                    let _ = release_manifest;
                    return Err(Error::new(
                        "profile_unavailable",
                        "answer evaluation requires a binary built with --features full",
                    ));
                }
            } else if compare {
                #[cfg(feature = "full")]
                {
                    value(kb::eval::compare(&config, &fixture)?)?
                }
                #[cfg(not(feature = "full"))]
                {
                    return Err(Error::new(
                        "profile_unavailable",
                        "retrieval comparison requires --features full",
                    ));
                }
            } else {
                value(kb::eval::run(&config, &fixture)?)?
            }
        }
        Command::Mcp => {
            kb::mcp::run(config)?;
            return Ok(false);
        }
        Command::Answer {
            question,
            filters,
            limit,
            budget_bytes,
        } => {
            if !config.profile.is_full() {
                return Err(Error::new(
                    "profile_requires_full",
                    "answer is available only with explicitly selected --profile full",
                ));
            }
            #[cfg(feature = "full")]
            {
                value(kb::answer::run(
                    &config,
                    &kb::model::AnswerRequest {
                        question,
                        filters: filters.into(),
                        limit,
                        budget_bytes,
                    },
                )?)?
            }
            #[cfg(not(feature = "full"))]
            {
                let _ = (question, filters, limit, budget_bytes);
                return Err(Error::new(
                    "profile_unavailable",
                    "answer requires --features full",
                ));
            }
        }
    };
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        render(&output);
    }
    Ok(failure)
}
fn render(output: &Value) {
    let data = &output["data"];
    if let Some(hits) = data.as_array() {
        for hit in hits {
            println!(
                "{}:{}:{}-{} — {}\n{}\n",
                hit["root_key"].as_str().unwrap_or(""),
                hit["path"].as_str().unwrap_or(""),
                hit["line_start"],
                hit["line_end"],
                hit["heading"].as_str().unwrap_or(""),
                hit["snippet"].as_str().unwrap_or("")
            );
        }
        if hits.is_empty() {
            println!("No results.");
        }
    } else if let Some(items) = data["items"].as_array() {
        for item in items {
            println!(
                "[{}] {}:{}:{}-{} — {}\n{}",
                item["evidence_id"].as_str().unwrap_or(""),
                item["root_key"].as_str().unwrap_or(""),
                item["path"].as_str().unwrap_or(""),
                item["line_start"],
                item["line_end"],
                item["heading"].as_str().unwrap_or(""),
                item["body"].as_str().unwrap_or("")
            );
        }
        println!(
            "{} / {} bytes; truncation: {}",
            data["used_bytes"], data["budget_bytes"], data["truncation"]
        );
    } else if let Some(body) = data["body"].as_str() {
        println!(
            "{}:{}:{}-{}\n{}",
            data["root_key"].as_str().unwrap_or(""),
            data["path"].as_str().unwrap_or(""),
            data["line_start"],
            data["line_end"],
            body
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(output).expect("serializable output")
        );
    }
}
fn render_mermaid(graph: &Value) {
    println!("flowchart LR");
    if let Some(nodes) = graph["data"]["nodes"].as_array() {
        for (index, node) in nodes.iter().enumerate() {
            let label = node["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .or(node["title"].as_str())
                .unwrap_or("document");
            let safe = label
                .chars()
                .filter(|c| c.is_alphanumeric() || " -_".contains(*c))
                .collect::<String>();
            println!("  N{index}[\"{safe}\"]");
        }
        if let Some(edges) = graph["data"]["edges"].as_array() {
            for edge in edges.iter().filter(|e| e["kind"] == "depends_on") {
                let from = nodes.iter().position(|n| n["key"] == edge["source"]);
                if let (Some(a), Some(targets)) = (from, edge["target_keys"].as_array()) {
                    for target in targets {
                        if let Some(b) = nodes.iter().position(|n| &n["key"] == target) {
                            println!("  N{b} --> N{a}");
                        }
                    }
                }
            }
        }
    }
}
fn main() {
    match execute(Cli::parse()) {
        Ok(false) => {}
        Ok(true) => std::process::exit(1),
        Err(error) => {
            eprintln!("{}", serde_json::json!({"schema_version":1,"error":error}));
            std::process::exit(2);
        }
    }
}
