//! The opinions: budget from measured hardware, then quality x physics.
//!
//! Margins and constants are field-tested values from Nous Research's
//! hermes-agent local_runtime (MIT), adapted:
//!   - discrete VRAM reserve: max(2 GiB, 9%) for the desktop's co-residents
//!   - unified-memory headroom: 20% of RAM stays with the OS
//!   - decode is memory-bound: tok/s ~ bandwidth / bytes-read-per-token;
//!     partially offloaded models read the VRAM-resident fraction at GPU
//!     speed and stream the rest from host RAM (serial two-tier model)
//!   - below 20 tok/s a model stops feeling pleasant for interactive use
//!
//! GPU bandwidth: measured per machine when hwprobe reports it
//! (bandwidth_gb_s on the primary GPU), class constant as fallback.
//! Host bandwidth: class value from ram_kind when known.
//!
//! Departure from hermes: a spilled MoE that clears the pleasant floor is
//! a first-class candidate, not a last resort — partial offload of MoE
//! models is normal daily-driver territory on 8-16GB cards.

use crate::catalog::{Catalog, ModelEntry};
use hwprobe::{GpuState, HardwareInfo};
use serde::Serialize;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// KV cache + runtime overhead allowance at a modest context, scaled to
/// model size (a 5GB model does not carry a 20GB model's KV). A real
/// estimator reads the GGUF header; v1 prices the rough cut.
fn overhead_for(size: u64) -> u64 {
    (768 * MIB).max((size as f64 * 0.12) as u64)
}

/// Fallbacks when hwprobe can't report a measured value.
const DISCRETE_BANDWIDTH_GB_S: f64 = 1000.0;
const UMA_BANDWIDTH_GB_S: f64 = 210.0;
const HOST_BANDWIDTH_GB_S: f64 = 80.0;
const PLEASANT_FLOOR_TOK_S: f64 = 20.0;

/// Host RAM bandwidth by memory kind (effective class values, GB/s).
fn host_bandwidth_gb_s(ram_kind: Option<&str>) -> f64 {
    match ram_kind {
        Some(k) if k.starts_with("LPDDR5") => 90.0,
        Some(k) if k.starts_with("DDR5") => 65.0,
        Some(k) if k.starts_with("LPDDR4") => 50.0,
        Some(k) if k.starts_with("DDR4") => 45.0,
        _ => HOST_BANDWIDTH_GB_S,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BudgetKind {
    DiscreteGpu,
    UnifiedMemory,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize)]
pub struct Budget {
    pub kind: BudgetKind,
    /// Bytes the model (weights + overhead) may occupy resident.
    pub resident_bytes: u64,
    /// Extra host bytes available for spilled MoE weights (discrete only).
    pub spill_extra_bytes: u64,
    /// Bandwidth of the resident tier (GPU or unified pool), GB/s.
    pub resident_bandwidth_gb_s: f64,
    /// Bandwidth of host RAM (spill tier / CPU-only), GB/s.
    pub host_bandwidth_gb_s: f64,
    /// True when resident_bandwidth came from hwprobe measurement rather
    /// than a class constant.
    pub bandwidth_measured: bool,
    pub driver_missing_nvidia: bool,
}

pub fn budget_from(info: &HardwareInfo) -> Budget {
    let ram_bytes = info.ram_mb * MIB;
    let host_bw = host_bandwidth_gb_s(info.ram_kind.as_deref());
    let driver_missing_nvidia = info.gpus.iter().any(|g| g.state == GpuState::DriverMissing);

    if info.unified_memory {
        let resident = match info.metal_max_working_set_mb {
            Some(mb) => mb * MIB,
            None => (ram_bytes as f64 * 0.80) as u64,
        };
        let measured = info.gpus.iter().find_map(|g| g.bandwidth_gb_s);
        return Budget {
            kind: BudgetKind::UnifiedMemory,
            resident_bytes: resident,
            spill_extra_bytes: 0,
            resident_bandwidth_gb_s: measured.unwrap_or(UMA_BANDWIDTH_GB_S),
            host_bandwidth_gb_s: host_bw,
            bandwidth_measured: measured.is_some(),
            driver_missing_nvidia,
        };
    }

    let best_gpu = info
        .gpus
        .iter()
        .filter(|g| !g.shared && g.state == GpuState::Ok)
        .filter(|g| g.vram_mb.is_some())
        .max_by_key(|g| g.vram_mb.unwrap_or(0));

    if let Some(gpu) = best_gpu {
        let vram = gpu.vram_mb.unwrap_or(0) * MIB;
        let reserve = (2 * GIB).max((vram as f64 * 0.09) as u64);
        let resident = vram.saturating_sub(reserve);
        let spill = (ram_bytes as f64 * 0.80) as u64;
        return Budget {
            kind: BudgetKind::DiscreteGpu,
            resident_bytes: resident,
            spill_extra_bytes: spill,
            resident_bandwidth_gb_s: gpu.bandwidth_gb_s.unwrap_or(DISCRETE_BANDWIDTH_GB_S),
            host_bandwidth_gb_s: host_bw,
            bandwidth_measured: gpu.bandwidth_gb_s.is_some(),
            driver_missing_nvidia,
        };
    }

    Budget {
        kind: BudgetKind::CpuOnly,
        resident_bytes: (ram_bytes as f64 * 0.80) as u64,
        spill_extra_bytes: 0,
        resident_bandwidth_gb_s: host_bw,
        host_bandwidth_gb_s: host_bw,
        bandwidth_measured: false,
        driver_missing_nvidia,
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "verdict")]
pub enum Verdict {
    /// Fits entirely in the resident budget.
    Resident {
        predicted_tok_s: f64,
    },
    /// Weights exceed resident but fit across resident+host; MoE only.
    Spilled {
        predicted_tok_s: f64,
    },
    TooBig,
    /// size_bytes missing in the catalogue.
    UnknownSize,
}

impl Verdict {
    /// Fits on this machine, resident or spilled.
    fn fits_tok_s(&self) -> Option<(f64, bool)> {
        match *self {
            Verdict::Resident { predicted_tok_s } => Some((predicted_tok_s, false)),
            Verdict::Spilled { predicted_tok_s } => Some((predicted_tok_s, true)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Assessment {
    pub id: String,
    pub display_name: String,
    pub quality: u32,
    pub description: String,
    pub repo: Option<String>,
    pub quant: String,
    pub tier_hint: Option<String>,
    #[serde(flatten)]
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub budget: Budget,
    pub assessments: Vec<Assessment>,
    /// Index into assessments, with the reason branch that fired.
    pub pick: Option<usize>,
    pub reason: Option<&'static str>,
}

fn assess(entry: &ModelEntry, budget: &Budget) -> Verdict {
    let Some(size) = entry.size_bytes else {
        return Verdict::UnknownSize;
    };
    let need = size + overhead_for(size);
    let bytes_per_token = (size as f64 * entry.decode_fraction).max(1.0);

    if need <= budget.resident_bytes {
        return Verdict::Resident {
            predicted_tok_s: budget.resident_bandwidth_gb_s * 1e9 / bytes_per_token,
        };
    }

    // Spill path: discrete GPU + MoE (dense models stream everything over
    // the bus every token — miserable; MoE reads only the active slice).
    // Serial two-tier read: the VRAM-resident fraction of the per-token
    // bytes moves at GPU bandwidth, the host-resident remainder at host
    // bandwidth.
    if budget.kind == BudgetKind::DiscreteGpu
        && entry.decode_fraction < 0.5
        && need <= budget.resident_bytes + budget.spill_extra_bytes
    {
        let resident_fraction = budget.resident_bytes as f64 / need as f64;
        let secs_per_token = bytes_per_token
            * (resident_fraction / (budget.resident_bandwidth_gb_s * 1e9)
                + (1.0 - resident_fraction) / (budget.host_bandwidth_gb_s * 1e9));
        return Verdict::Spilled {
            predicted_tok_s: 1.0 / secs_per_token,
        };
    }

    Verdict::TooBig
}

/// The pick: highest quality among everything that fits (resident or
/// spilled) above the pleasant floor; else the fastest thing that fits.
pub fn recommend(info: &HardwareInfo, catalog: &Catalog) -> Recommendation {
    let budget = budget_from(info);
    let assessments: Vec<Assessment> = catalog
        .models
        .iter()
        .map(|m| Assessment {
            id: m.id.clone(),
            display_name: m.display_name.clone(),
            quality: m.quality,
            description: m.description.clone(),
            repo: m.repo.clone(),
            quant: m.quant.clone(),
            tier_hint: m.tier_hint.clone(),
            verdict: assess(m, &budget),
        })
        .collect();

    let mut pick = None;
    let mut reason = None;

    // 1. best quality that fits and clears the floor
    let mut best_q = 0u32;
    for (i, a) in assessments.iter().enumerate() {
        if let Some((tok_s, spilled)) = a.verdict.fits_tok_s()
            && tok_s >= PLEASANT_FLOOR_TOK_S
            && a.quality > best_q
        {
            best_q = a.quality;
            pick = Some(i);
            reason = Some(if spilled {
                "best-quality-spilled"
            } else {
                "best-quality-resident"
            });
        }
    }
    // was a higher-quality entry runnable but speed-gated?
    if pick.is_some()
        && assessments.iter().any(|a| {
            matches!(a.verdict.fits_tok_s(), Some((t, _)) if t < PLEASANT_FLOOR_TOK_S)
                && a.quality > best_q
        })
    {
        reason = Some("speed-gated-quality");
    }

    // 2. fastest thing that fits at all
    if pick.is_none() {
        let mut best_speed = 0.0f64;
        for (i, a) in assessments.iter().enumerate() {
            if let Some((tok_s, _)) = a.verdict.fits_tok_s()
                && tok_s > best_speed
            {
                best_speed = tok_s;
                pick = Some(i);
                reason = Some("fastest-fits");
            }
        }
    }

    Recommendation {
        budget,
        assessments,
        pick,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;
    use hwprobe::{GpuInfo, GpuVendor, HardwareInfo};

    fn machine(ram_mb: u64, vram_mb: Option<u64>, unified: bool) -> HardwareInfo {
        HardwareInfo {
            ram_mb,
            ram_kind: None,
            gpus: vram_mb
                .map(|v| {
                    vec![GpuInfo {
                        vendor: GpuVendor::Nvidia,
                        model: "test GPU".into(),
                        vram_mb: Some(v),
                        shared: false,
                        state: GpuState::Ok,
                        primary: true,
                        bandwidth_gb_s: None,
                    }]
                })
                .unwrap_or_default(),
            unified_memory: unified,
            metal_max_working_set_mb: unified.then_some((ram_mb as f64 * 0.75) as u64),
        }
    }

    fn picked_id(rec: &Recommendation) -> &str {
        &rec.assessments[rec.pick.expect("expected a pick")].id
    }

    #[test]
    fn big_discrete_card_gets_the_dense_flagship() {
        let cat = catalog::load(None).unwrap();
        let rec = recommend(&machine(65536, Some(24576), false), &cat);
        assert_eq!(picked_id(&rec), "qwen3.8-27b");
        assert_eq!(rec.reason, Some("best-quality-resident"));
    }

    #[test]
    fn eight_gb_card_spills_the_moe() {
        // kurtGodel-shaped: 32GB RAM, 8GB VRAM
        let cat = catalog::load(None).unwrap();
        let rec = recommend(&machine(31775, Some(8192), false), &cat);
        assert_eq!(picked_id(&rec), "qwen3.6-35b-a3b");
        assert_eq!(rec.reason, Some("best-quality-spilled"));
    }

    #[test]
    fn mac_32gb_speed_gates_the_dense_flagship() {
        // 27B fits resident on 32GB unified but decodes below the floor;
        // the MoE wins and the reason says why the big model lost.
        let cat = catalog::load(None).unwrap();
        let rec = recommend(&machine(32768, None, true), &cat);
        assert_eq!(picked_id(&rec), "qwen3.6-35b-a3b");
        assert_eq!(rec.reason, Some("speed-gated-quality"));
    }

    #[test]
    fn cpu_only_laptop_gets_the_tiny_model() {
        // 8GB RAM, no GPU: E4B fits but is too slow on host bandwidth;
        // E2B clears the floor.
        let cat = catalog::load(None).unwrap();
        let rec = recommend(&machine(8192, None, false), &cat);
        assert_eq!(rec.budget.kind, BudgetKind::CpuOnly);
        assert_eq!(picked_id(&rec), "gemma-4-e2b");
        assert_eq!(rec.reason, Some("speed-gated-quality"));
    }

    #[test]
    fn nothing_fits_returns_none() {
        let cat = catalog::load(None).unwrap();
        let rec = recommend(&machine(2048, None, false), &cat);
        assert!(rec.pick.is_none());
    }

    #[test]
    fn measured_bandwidth_is_used_over_class_constant() {
        let cat = catalog::load(None).unwrap();
        let mut info = machine(31775, Some(8192), false);
        info.gpus[0].bandwidth_gb_s = Some(326.0);
        let rec = recommend(&info, &cat);
        assert!(rec.budget.bandwidth_measured);
        assert!((rec.budget.resident_bandwidth_gb_s - 326.0).abs() < 1e-9);
    }
}
