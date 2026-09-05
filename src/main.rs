use modelfit::recommend::{BudgetKind, Verdict};
use modelfit::{catalog, get, recommend};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("modelfit {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let is_get = args.get(1).map(String::as_str) == Some("get");
    let is_show = args.get(1).map(String::as_str) == Some("show");
    let json = args.iter().any(|a| a == "--json");
    let arg_value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let catalog_path = arg_value("--catalog");

    let cat = match catalog::load(catalog_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let info = hwprobe::detect();
    let rec = recommend::recommend(&info, &cat);

    if is_get {
        // `modelfit get [id] [--dir PATH]` — id defaults to the pick.
        let id = args
            .get(2)
            .filter(|a| !a.starts_with("--"))
            .map(String::as_str);
        let picked_id = rec.pick.map(|i| rec.assessments[i].id.as_str());
        let opts = get::GetOptions {
            id,
            dir: arg_value("--dir"),
        };
        if let Err(e) = get::run(&cat, picked_id, &opts) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }

    if is_show {
        // `modelfit show <id>` — one catalogue entry, assessed on this machine.
        let Some(id) = args.get(2).filter(|a| !a.starts_with("--")) else {
            eprintln!("usage: modelfit show <id>");
            eprintln!(
                "ids: {}",
                cat.models
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            std::process::exit(1);
        };
        let Some(a) = rec.assessments.iter().find(|a| &a.id == id) else {
            eprintln!("error: '{id}' is not in the catalogue");
            std::process::exit(1);
        };
        println!("{} (q{})", a.display_name, a.quality);
        println!("  {}", a.description);
        let verdict = match &a.verdict {
            Verdict::Resident { predicted_tok_s } => {
                format!("fits on this machine, ~{predicted_tok_s:.0} tok/s")
            }
            Verdict::Spilled { predicted_tok_s } => {
                format!("fits with MoE spill, ~{predicted_tok_s:.0} tok/s")
            }
            Verdict::TooBig => match &a.tier_hint {
                Some(t) => format!("too big for this machine (needs: {t})"),
                None => "too big for this machine".to_string(),
            },
            Verdict::UnknownSize => "size unknown in catalogue".to_string(),
        };
        println!("  {verdict}");
        if let Some(repo) = &a.repo {
            println!("  get: modelfit get {id}   ({repo} — {} quant)", a.quant);
        }
        if let Some(launch) = &a.launch {
            println!(
                "  run: llama-server -m <path-to-model.gguf> {} {}",
                launch.args, launch.samplers
            );
            if !launch.tested {
                println!("       (flags authored from model defaults, not yet field-tested)");
            }
        }
        return;
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&rec).unwrap());
        return;
    }

    // human output
    let kind = match rec.budget.kind {
        BudgetKind::DiscreteGpu => "discrete GPU",
        BudgetKind::UnifiedMemory => "unified memory",
        BudgetKind::CpuOnly => "CPU only",
    };
    println!(
        "machine: {} | usable for models: {:.1} GiB{}",
        kind,
        rec.budget.resident_bytes as f64 / (1u64 << 30) as f64,
        if rec.budget.spill_extra_bytes > 0 {
            format!(
                " (+{:.1} GiB host for MoE spill)",
                rec.budget.spill_extra_bytes as f64 / (1u64 << 30) as f64
            )
        } else {
            String::new()
        }
    );
    if rec.budget.driver_missing_nvidia {
        println!(
            "note: NVIDIA GPU detected without a working driver — install it to unlock GPU inference"
        );
    }
    println!();

    for a in &rec.assessments {
        let verdict = match &a.verdict {
            Verdict::Resident { predicted_tok_s } => {
                format!("fits          ~{predicted_tok_s:.0} tok/s")
            }
            Verdict::Spilled { predicted_tok_s } => {
                format!("fits (spill)  ~{predicted_tok_s:.0} tok/s")
            }
            Verdict::TooBig => match &a.tier_hint {
                Some(t) => format!("too big       (needs: {t})"),
                None => "too big".to_string(),
            },
            Verdict::UnknownSize => "size unknown (fill size_bytes)".to_string(),
        };
        println!("  {:<24} q{:<4} {}", a.display_name, a.quality, verdict);
    }
    println!();

    match rec.pick {
        Some(i) => {
            let a = &rec.assessments[i];
            println!(
                "recommended: {} ({})",
                a.display_name,
                rec.reason.unwrap_or("")
            );
            if let Some(repo) = &a.repo {
                println!("  get: modelfit get   ({repo} — {} quant)", a.quant);
            }
            if let Some(launch) = &a.launch {
                println!(
                    "  run: llama-server -m <path-to-model.gguf> {} {}",
                    launch.args, launch.samplers
                );
                if !launch.tested {
                    println!("       (flags authored from model defaults, not yet field-tested)");
                }
            }
        }
        None => println!("recommended: nothing in the catalogue fits this machine"),
    }
}
