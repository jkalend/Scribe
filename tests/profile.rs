use std::process::Command;

#[test]
fn cli_environment_and_config_precedence_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("notes");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("one.md"), "# Local\noffline evidence\n").unwrap();
    let path = temp.path().join("config.toml");
    let config = serde_json::json!({"profile":"full","index":{"database":temp.path().join("index.sqlite")},"roots":{"notes":{"path":root}}});
    let as_toml = toml::to_string(&config).unwrap();
    std::fs::write(&path, as_toml).unwrap();
    let run = |extra: &[&str], profile: Option<&str>| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
        cmd.env_remove("KB_CONFIG")
            .env_remove("KB_PROFILE")
            .arg("--config")
            .arg(&path);
        if let Some(p) = profile {
            cmd.env("KB_PROFILE", p);
        }
        cmd.args(extra).output().unwrap()
    };
    let from_config = run(&["index"], None);
    assert!(!from_config.status.success());
    assert!(
        String::from_utf8_lossy(&from_config.stderr).contains(if cfg!(feature = "full") {
            "embedding_unavailable"
        } else {
            "profile_unavailable"
        })
    );
    let from_env = run(&["index", "--json"], Some("light"));
    assert!(
        from_env.status.success(),
        "{}",
        String::from_utf8_lossy(&from_env.stderr)
    );
    let env_data: serde_json::Value = serde_json::from_slice(&from_env.stdout).unwrap();
    assert_eq!(env_data["profile"], "light");
    let from_cli = run(
        &["search", "offline", "--profile", "light", "--json"],
        Some("full"),
    );
    assert!(from_cli.status.success());
    let data: serde_json::Value = serde_json::from_slice(&from_cli.stdout).unwrap();
    assert_eq!(data["data"][0]["path"], "one.md");
    let full = run(&["search", "offline", "--profile", "full"], Some("light"));
    assert!(!full.status.success());
    assert!(
        String::from_utf8_lossy(&full.stderr).contains(if cfg!(feature = "full") {
            "embedding_unavailable"
        } else {
            "profile_unavailable"
        })
    );
    let answer = run(&["answer", "what?", "--profile", "light"], None);
    assert!(!answer.status.success());
    assert!(String::from_utf8_lossy(&answer.stderr).contains("profile_requires_full"));
    for environment in [None, Some("full")] {
        let answer = run(&["answer", "what?"], environment);
        assert!(!answer.status.success());
        assert!(
            String::from_utf8_lossy(&answer.stderr).contains(if cfg!(feature = "full") {
                "generator_unavailable"
            } else {
                "profile_unavailable"
            })
        );
    }
    let answer = run(&["answer", "what?"], Some("light"));
    assert!(String::from_utf8_lossy(&answer.stderr).contains("profile_requires_full"));
}

#[test]
fn no_configuration_never_indexes_the_current_directory() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("private.md"), "# Never implicit\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_kb"))
        .current_dir(temp.path())
        .env_remove("KB_CONFIG")
        .env_remove("KB_PROFILE")
        .arg("index")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("configuration_required"));
    assert!(!temp.path().join(".kb").exists());
}
