//! Optional sidecar of `text-generation-session-execution-v2`: which earlier round
//! a round's declared prefix comes from.
//!
//! The session format has one prefix length per round and takes it to be shared
//! with the previous round. A rewind that returns to an older branch, or a side
//! lineage inside a session, is not expressible that way. A file named
//! `<trace>.prefix_rounds.csv` next to the trace, with the header
//! `session_id,round_idx,prefix_round`, names the source round for the rows that
//! differ. Without the file, every round follows the previous one.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Row {
    session_id: String,
    round_idx: usize,
    prefix_round: u32,
}

pub(super) fn load(trace_path: &str) -> Result<HashMap<(String, usize), u32>> {
    let path = format!("{trace_path}.prefix_rounds.csv");
    if !Path::new(&path).exists() {
        return Ok(HashMap::new());
    }
    let mut reader =
        csv::Reader::from_path(&path).with_context(|| format!("failed to open {path}"))?;
    let mut rounds = HashMap::new();
    for row in reader.deserialize() {
        let row: Row = row.with_context(|| format!("failed to parse {path}"))?;
        rounds.insert((row.session_id, row.round_idx), row.prefix_round);
    }
    Ok(rounds)
}
