//! Library API for embedding modelfit: run
//! `recommend::recommend(&hwprobe::detect(), &catalog)`.
//! The CLI in main.rs is a thin wrapper over this.
pub mod catalog;
pub mod recommend;
