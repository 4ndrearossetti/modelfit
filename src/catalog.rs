//! Catalogue schema + loading. The catalogue is data, the logic is code:
//! editorial judgement (which models, what quality ordering) lives in
//! models.json; physics lives in recommend.rs.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Catalog {
    pub schema_version: u32,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub repo: Option<String>,
    pub quant: String,
    /// Total GGUF size in bytes (sum of shards). None = not yet filled in;
    /// the entry is listed but can't be recommended.
    pub size_bytes: Option<u64>,
    /// Editorial ordering, higher = smarter. Authored, not computed.
    pub quality: u32,
    /// Share of weights read per decoded token: 1.0 dense, active/total MoE.
    pub decode_fraction: f64,
    #[serde(default)]
    pub tier_hint: Option<String>,
}

/// Embedded snapshot — works offline. --catalog PATH overrides.
const EMBEDDED: &str = include_str!("../models.json");

pub fn load(path: Option<&str>) -> Result<Catalog, String> {
    let text = match path {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("reading catalogue {p}: {e}"))?,
        None => EMBEDDED.to_string(),
    };
    let cat: Catalog =
        serde_json::from_str(&text).map_err(|e| format!("parsing catalogue: {e}"))?;
    if cat.schema_version != 1 {
        return Err(format!(
            "unsupported catalogue schema_version {}",
            cat.schema_version
        ));
    }
    Ok(cat)
}
