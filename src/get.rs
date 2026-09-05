//! `modelfit get`: download the recommended (or named) catalogue model
//! from Hugging Face. Filenames are discovered from the HF API at
//! download time — never guessed or stored. After a successful download
//! the launch command is printed with the real path.

use crate::catalog::Catalog;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct GetOptions<'a> {
    /// Catalogue id to fetch; None = the recommendation's pick.
    pub id: Option<&'a str>,
    /// Target directory; None = ~/models.
    pub dir: Option<&'a str>,
}

pub fn run(cat: &Catalog, picked_id: Option<&str>, opts: &GetOptions) -> Result<(), String> {
    let id = opts
        .id
        .or(picked_id)
        .ok_or("nothing to get: no pick and no id given")?;
    let entry = cat
        .models
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| format!("'{id}' is not in the catalogue"))?;
    let repo = entry
        .repo
        .as_deref()
        .ok_or_else(|| format!("catalogue entry '{id}' has no repo"))?;

    let dir = target_dir(opts.dir)?;
    fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    println!("resolving {repo} ({} quant) ...", entry.quant);
    let files = list_repo_ggufs(repo)?;
    let wanted = select_files(&files, &entry.quant);
    if wanted.is_empty() {
        return Err(format!(
            "no .gguf files matching quant '{}' found in {repo} — check the catalogue entry",
            entry.quant
        ));
    }

    let total: u64 = wanted.iter().map(|f| f.size).sum();
    if let Some(expected) = entry.size_bytes {
        let drift = (total as f64 - expected as f64).abs() / expected as f64;
        if drift > 0.05 {
            println!(
                "warning: repo files total {:.2} GB but catalogue says {:.2} GB — proceeding with repo truth",
                total as f64 / 1e9,
                expected as f64 / 1e9
            );
        }
    }
    println!(
        "{} file(s), {:.2} GB total -> {}",
        wanted.len(),
        total as f64 / 1e9,
        dir.display()
    );

    let mut first_local: Option<PathBuf> = None;
    for f in &wanted {
        let local = dir.join(
            Path::new(&f.path)
                .file_name()
                .ok_or_else(|| format!("bad path in repo listing: {}", f.path))?,
        );
        if first_local.is_none() {
            first_local = Some(local.clone());
        }
        download_file(repo, f, &local)?;
    }

    println!("done.");
    if let (Some(path), Some(launch)) = (first_local, &entry.launch) {
        println!(
            "run: llama-server -m {} {} {}",
            path.display(),
            launch.args,
            launch.samplers
        );
        if !launch.tested {
            println!("     (flags authored from model defaults, not yet field-tested)");
        }
    }
    Ok(())
}

pub struct RepoFile {
    pub path: String,
    pub size: u64,
}

/// GGUF files in the repo, via the HF tree API. (Repos with >1000 files
/// paginate; GGUF repos don't get there.)
fn list_repo_ggufs(repo: &str) -> Result<Vec<RepoFile>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let resp = request(&url).call().map_err(|e| format!("HF API: {e}"))?;
    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("HF API response: {e}"))?;
    let entries = json.as_array().ok_or("unexpected HF API response shape")?;
    Ok(entries
        .iter()
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?;
            if !path.ends_with(".gguf") {
                return None;
            }
            Some(RepoFile {
                path: path.to_string(),
                size: e.get("size")?.as_u64()?,
            })
        })
        .collect())
}

/// Files matching the catalogue quant. Handles the UD- prefix trap:
/// "Q4_K_M" must not match "UD-Q4_K_M" files, and vice versa is
/// impossible by substring. Matches single files (-QUANT.gguf) and
/// shards (-QUANT-00001-of-0000N.gguf).
pub fn select_files(files: &[RepoFile], quant: &str) -> Vec<RepoFile> {
    files
        .iter()
        .filter(|f| {
            let name = f.path.rsplit('/').next().unwrap_or(&f.path);
            let matches =
                name.contains(&format!("-{quant}.")) || name.contains(&format!("-{quant}-"));
            let ud_trap = !quant.starts_with("UD-") && name.contains(&format!("-UD-{quant}"));
            matches && !ud_trap
        })
        .map(|f| RepoFile {
            path: f.path.clone(),
            size: f.size,
        })
        .collect()
}

/// Download one file with resume: skip if complete, Range-append if
/// partial, fresh otherwise.
fn download_file(repo: &str, f: &RepoFile, local: &Path) -> Result<(), String> {
    let existing = fs::metadata(local).map(|m| m.len()).unwrap_or(0);
    if existing == f.size {
        println!("  {} — already complete, skipping", display_name(local));
        return Ok(());
    }
    if existing > f.size {
        return Err(format!(
            "{} is larger than expected ({existing} > {}); remove it and retry",
            local.display(),
            f.size
        ));
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{}", f.path);
    let mut req = request(&url);
    let mut file = if existing > 0 {
        println!(
            "  {} — resuming at {:.1}%",
            display_name(local),
            existing as f64 / f.size as f64 * 100.0
        );
        req = req.set("Range", &format!("bytes={existing}-"));
        fs::OpenOptions::new()
            .append(true)
            .open(local)
            .map_err(|e| format!("opening {}: {e}", local.display()))?
    } else {
        fs::File::create(local).map_err(|e| format!("creating {}: {e}", local.display()))?
    };

    let resp = req
        .call()
        .map_err(|e| format!("downloading {}: {e}", f.path))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = existing;
    let mut last_pct = u64::MAX;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("reading {}: {e}", f.path))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("writing {}: {e}", local.display()))?;
        done += n as u64;
        let pct = done * 100 / f.size;
        if pct != last_pct {
            print!(
                "\r  {} — {pct}% ({:.2}/{:.2} GB)",
                display_name(local),
                done as f64 / 1e9,
                f.size as f64 / 1e9
            );
            std::io::stdout().flush().ok();
            last_pct = pct;
        }
    }
    println!();
    if done != f.size {
        return Err(format!(
            "{}: got {done} of {} bytes — connection dropped? rerun to resume",
            f.path, f.size
        ));
    }
    Ok(())
}

/// HF_TOKEN honoured when set (gated repos).
fn request(url: &str) -> ureq::Request {
    let req = ureq::get(url);
    match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => req.set("Authorization", &format!("Bearer {t}")),
        _ => req,
    }
}

fn target_dir(over: Option<&str>) -> Result<PathBuf, String> {
    if let Some(d) = over {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "can't find home directory; use --dir")?;
    Ok(PathBuf::from(home).join("models"))
}

fn display_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rf(path: &str, size: u64) -> RepoFile {
        RepoFile {
            path: path.into(),
            size,
        }
    }

    #[test]
    fn plain_quant_does_not_match_ud_variant() {
        let files = vec![
            rf("gemma-4-E4B-it-Q4_K_M.gguf", 10),
            rf("gemma-4-E4B-it-UD-Q4_K_M.gguf", 11),
        ];
        let picked = select_files(&files, "Q4_K_M");
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].path, "gemma-4-E4B-it-Q4_K_M.gguf");
    }

    #[test]
    fn ud_quant_matches_only_ud() {
        let files = vec![rf("m-Q4_K_M.gguf", 10), rf("m-UD-Q4_K_M.gguf", 11)];
        let picked = select_files(&files, "UD-Q4_K_M");
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].path, "m-UD-Q4_K_M.gguf");
    }

    #[test]
    fn shards_and_subfolders_match() {
        let files = vec![
            rf("UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00002.gguf", 10),
            rf("UD-Q4_K_XL/Big-UD-Q4_K_XL-00002-of-00002.gguf", 10),
            rf("Q8_0/Big-Q8_0.gguf", 99),
            rf("mmproj-BF16.gguf", 5),
        ];
        let picked = select_files(&files, "UD-Q4_K_XL");
        assert_eq!(picked.len(), 2);
    }
}
