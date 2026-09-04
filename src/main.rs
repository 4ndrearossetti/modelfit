use modelfit::recommend::{BudgetKind, Verdict};
use modelfit::{catalog, recommend};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let catalog_path = args
        .iter()
        .position(|a| a == "--catalog")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);

    let cat = match catalog::load(catalog_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let info = hwprobe::detect();
    let rec = recommend::recommend(&info, &cat);

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
                println!("  get: {} — {} quant", repo, a.quant);
            }
        }
        None => println!("recommended: nothing in the catalogue fits this machine"),
    }
}
