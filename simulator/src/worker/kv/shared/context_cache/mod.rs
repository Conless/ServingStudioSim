//! Content-level retained KV with a host tier, for hybrid-model prefix caching.
//!
//! [`PrefixCache`](super::prefix_cache::PrefixCache) keeps one count-only entry
//! per session. This cache instead keeps each session's token tree: a node is a
//! run of tokens shared by every context below it, and it holds, per tier (HBM
//! and host memory), its KV and the recurrent state at its end. KV and states are
//! separate residencies, which is what makes the following expressible:
//!
//! - a rewind that resumes at an old state, with the newer tail still cached;
//! - a state evicted while its KV stays, and the reverse;
//! - a hit served by loading host data before the iteration starts;
//! - branches of one session sharing their common prefix.
//!
//! A hit is the deepest state whose whole KV prefix is resident in some tier, at
//! or below the declared prefix. A session stays on one partition for life.
//!
//! ## Ownership
//!
//! Active requests still own their KV through the hard tier: `begin` locks the
//! request's path and its HBM bytes leave [`ContextCache::used_units`], so
//! nothing is counted twice, and `finish` returns them. Two running streams of
//! one session that share a prefix are both charged for it, which overstates the
//! hard tier for multi-stream traces.
//!
//! ## Tiers and timing
//!
//! Tiers are exclusive except for a loaded host copy, which stays until host room
//! is needed. Loads and demotions move over a per-partition link at a fixed rate;
//! an iteration waits for its requests' loads (see
//! [`crate::worker::kv::IterWorkerKv::iteration_load_end`]), while a demotion is
//! asynchronous and frees its HBM bytes at once.
//!
//! ## Not modeled
//!
//! The scheduler CPU the policies themselves cost; sharing across sessions.

mod lru;
mod price_graph;
mod survival;

use std::collections::{BinaryHeap, HashMap, VecDeque};

use serde::Deserialize;

use crate::common::{ContextRef, RequestId, SessionInput};
use crate::worker::kv::shared::prefix_cache::{
    PrefixCacheInsertResult, PrefixCacheMutation, PrefixCacheMutationKind, PrefixCacheTakeResult,
};

use price_graph::PriceGraph;
use survival::Survival;

pub(super) const HBM: usize = 0;
pub(super) const HOST: usize = 1;
const ON_HBM: u8 = 1 << HBM;
const ON_HOST: u8 = 1 << HOST;
const NO_NODE: u32 = u32::MAX;
/// Fixed cost of starting one host load.
const LOAD_SETUP_S: f64 = 30e-6;

/// How the retained data is chosen.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ContextCachePolicy {
    /// Per-component LRU over leaf KV, then over states (SGLang's radix cache).
    #[default]
    Lru,
    /// Online value-per-byte pricing per context; see [`price_graph`].
    PriceGraph,
}

/// Selector-level knobs of the content-level cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextCacheConfig {
    pub policy: ContextCachePolicy,
    /// Host memory per partition.
    pub host_capacity_bytes: u64,
    /// Host link rate per partition.
    pub link_bytes_per_s: f64,
    /// `alpha p + beta p^2 / 2` is the seconds to recompute a prefix of `p` tokens.
    pub recompute_alpha_s: f64,
    pub recompute_beta_s: f64,
    /// Keep a host copy of everything at insertion (HiCache `write_through`);
    /// otherwise demote only on eviction (`write_back`).
    pub write_through: bool,
    /// Share of each tier that only states may use (SGLang's static split);
    /// `None` is one pool.
    pub state_share: Option<f64>,
    /// Engines resume on whole KV pages: positions are floored to this.
    pub page_tokens: u32,
    /// Bytes per token when the modeled engine's KV differs from the arch layout's.
    pub bytes_per_token: Option<u64>,
}

/// Byte sizes the cache charges, resolved from the arch's layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextCacheSizes {
    pub(crate) bytes_per_token: u64,
    pub(crate) bytes_per_state: u64,
    pub(crate) bytes_per_unit: u64,
    pub(crate) capacity_units: u64,
    /// Checkpoint a state at every multiple of this many tokens.
    pub(crate) checkpoint_interval: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
struct Demand {
    /// `class_of(is_tail, is_mark)`.
    class: usize,
    last_finish: f64,
    spell: Option<u32>,
    running: u32,
}

const fn class_of(tail: bool, mark: bool) -> usize {
    (if tail { 0 } else { 2 }) + (if mark { 1 } else { 0 })
}

#[derive(Clone, Debug)]
struct Node {
    session: u32,
    parent: u32,
    start: u32,
    tokens: u32,
    /// Tiers holding `[start, start + tokens)`.
    kv: u8,
    /// Tiers holding the state at `start + tokens`.
    state: u8,
    locks: u32,
    last_used: u64,
    /// Children holding KV, per tier: a parent is an LRU leaf only without them.
    kv_children: [u32; 2],
    demand: Option<Demand>,
    /// Index in the session's start-sorted node list, set by `sorted_nodes`.
    slot: u32,
    host_ready_s: f64,
}

impl Node {
    fn end(&self) -> u32 {
        self.start + self.tokens
    }
}

#[derive(Clone, Debug, Default)]
struct Session {
    nodes: Vec<u32>,
    nodes_sorted: bool,
    /// The context each finished round left, by round index.
    round_tips: HashMap<u32, u32>,
    /// The last finished context; the linear-chain default for the next round.
    tip: Option<u32>,
    finished_at_s: f64,
    pricing: price_graph::SessionPricing,
}

struct ActiveRequest {
    session: u32,
    round: u32,
    /// The node ending at the declared prefix; `NO_NODE` for an empty one.
    prefix_node: u32,
    prefix_tokens: u32,
    hit_tokens: u32,
    chunk_ends: Vec<u32>,
    load_end_s: f64,
}

/// What a request could reuse right now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Plan {
    hit_tokens: u32,
    /// KV tokens below the hit that only the host holds.
    host_kv_tokens: u32,
    host_state: bool,
}

pub(crate) struct ContextCache {
    config: ContextCacheConfig,
    sizes: ContextCacheSizes,
    nodes: Vec<Node>,
    sessions: HashMap<u32, Session>,
    active: HashMap<RequestId, ActiveRequest>,
    /// Bytes by tier, KV then state. HBM bytes of locked nodes are hard-tier.
    kv_bytes: [u64; 2],
    state_bytes: [u64; 2],
    used_tokens: u64,
    h2d_free_s: f64,
    d2h_free_s: f64,
    use_counter: u64,
    /// Nodes that may hold a host copy of HBM data, reclaimed first.
    duplicates: VecDeque<u32>,
    lru: [lru::LruHeaps; 2],
    survival: Survival,
    pricing: PriceGraph,
    /// Evictions since the last drain, as receipts for the KV owner.
    receipts: Vec<PrefixCacheMutation>,
    counters: ContextCacheCounters,
}

/// Run totals, logged when the worker drops.
#[derive(Clone, Copy, Debug, Default)]
struct ContextCacheCounters {
    hits_on_hbm: u64,
    hits_on_host: u64,
    misses: u64,
    hit_tokens: u64,
    declared_tokens: u64,
    loaded_bytes: u64,
    demoted_bytes: u64,
    /// Occupancy sampled at each finish: KV and state, per tier.
    samples: u64,
    kv_byte_samples: [f64; 2],
    state_byte_samples: [f64; 2],
}

impl ContextCache {
    pub(crate) fn new(config: ContextCacheConfig, mut sizes: ContextCacheSizes) -> Self {
        if let Some(bytes_per_token) = config.bytes_per_token {
            sizes.bytes_per_token = bytes_per_token;
        }
        Self {
            config,
            sizes,
            nodes: Vec::new(),
            sessions: HashMap::new(),
            active: HashMap::new(),
            kv_bytes: [0; 2],
            state_bytes: [0; 2],
            used_tokens: 0,
            h2d_free_s: 0.0,
            d2h_free_s: 0.0,
            use_counter: 0,
            duplicates: VecDeque::new(),
            lru: Default::default(),
            survival: Survival::new(),
            pricing: PriceGraph::new(),
            receipts: Vec::new(),
            counters: ContextCacheCounters::default(),
        }
    }

    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    pub(crate) fn used_units(&self) -> u64 {
        (self.kv_bytes[HBM] + self.state_bytes[HBM]).div_ceil(self.sizes.bytes_per_unit)
    }

    /// Whether this partition owns the session: it stays here for life.
    pub(crate) fn owns_session(&self, session: u32) -> bool {
        self.sessions.contains_key(&session)
    }

    /// The hit a request declaring `input` would get right now.
    pub(crate) fn peek(&self, input: SessionInput) -> u32 {
        match input {
            SessionInput::Session {
                session_id,
                declared_prefix_tokens,
                context,
                ..
            } => {
                self.plan(session_id, context, declared_prefix_tokens)
                    .hit_tokens
            }
            _ => 0,
        }
    }

    /// Lock the request's path, account its arrival, and start its load.
    pub(crate) fn begin(
        &mut self,
        request: RequestId,
        input: SessionInput,
        now_s: f64,
    ) -> PrefixCacheTakeResult {
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            context,
            ..
        } = input
        else {
            panic!("a context cache only serves session requests");
        };
        let cache_used_before = self.used_tokens;
        if !self.sessions.contains_key(&session_id) {
            self.sessions.insert(session_id, Session::default());
            self.pricing.register_session(session_id);
        }
        let plan = self.plan(session_id, context, declared_prefix_tokens);
        let prefix_tip = self.context_tip(session_id, context);
        let chain_end = prefix_tip.map_or(0, |tip| self.nodes[tip as usize].end());
        let prefix_tokens = (declared_prefix_tokens / self.config.page_tokens
            * self.config.page_tokens)
            .min(chain_end)
            .max(plan.hit_tokens);
        let prefix_node = if prefix_tokens > 0 {
            self.boundary_on_chain(prefix_tip.expect("a prefix needs a chain"), prefix_tokens)
        } else {
            NO_NODE
        };

        // Lock first so loaded data lands as hard-tier bytes.
        let mut cursor = prefix_node;
        let mut load_ready_s = now_s;
        while cursor != NO_NODE {
            self.lock(cursor);
            self.touch(cursor);
            cursor = self.nodes[cursor as usize].parent;
        }
        let mut load_bytes = 0;
        if plan.hit_tokens > 0 {
            let mut cursor = self.boundary_on_chain(prefix_node, plan.hit_tokens);
            let top_state = cursor;
            while cursor != NO_NODE {
                let node = &self.nodes[cursor as usize];
                load_ready_s = load_ready_s.max(node.host_ready_s);
                if node.kv & ON_HBM == 0 {
                    load_bytes += u64::from(node.tokens) * self.sizes.bytes_per_token;
                    self.set_kv(cursor, HBM, true);
                    self.duplicates.push_back(cursor);
                }
                cursor = self.nodes[cursor as usize].parent;
            }
            if self.nodes[top_state as usize].state & ON_HBM == 0 {
                load_bytes += self.sizes.bytes_per_state;
                self.set_state(top_state, HBM, true);
                self.duplicates.push_back(top_state);
            }
        }
        let load_end_s = if load_bytes > 0 {
            let start = load_ready_s.max(self.h2d_free_s);
            self.h2d_free_s = start + load_bytes as f64 / self.config.link_bytes_per_s;
            self.h2d_free_s + LOAD_SETUP_S
        } else {
            now_s
        };
        self.counters.loaded_bytes += load_bytes;
        self.counters.declared_tokens += u64::from(declared_prefix_tokens);
        self.counters.hit_tokens += u64::from(plan.hit_tokens);
        if plan.hit_tokens == 0 {
            self.counters.misses += 1;
        } else if load_bytes > 0 {
            self.counters.hits_on_host += 1;
        } else {
            self.counters.hits_on_hbm += 1;
        }

        if self.config.policy == ContextCachePolicy::PriceGraph {
            self.note_arrival(prefix_node, now_s);
        }
        self.active.insert(
            request,
            ActiveRequest {
                session: session_id,
                round: context.round,
                prefix_node,
                prefix_tokens,
                hit_tokens: plan.hit_tokens,
                chunk_ends: Vec::new(),
                load_end_s,
            },
        );
        PrefixCacheTakeResult {
            hit_tokens: plan.hit_tokens,
            full_hit_tokens: declared_prefix_tokens.min(chain_end),
            entry_tokens: cache_used_before - self.used_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            return_metadata: None,
        }
    }

    /// The time this request's host load finishes, if it still has one.
    pub(crate) fn load_end_s(&self, request: RequestId) -> Option<f64> {
        self.active.get(&request).map(|active| active.load_end_s)
    }

    /// A prefill chunk ended at `position`: engines checkpoint a state there.
    pub(crate) fn note_chunk_end(&mut self, request: RequestId, position: u32) {
        if let Some(active) = self.active.get_mut(&request) {
            active.chunk_ends.push(position);
        }
    }

    /// Return the request's context, add the new tail and its states, and let
    /// the policy decide what stays. Eviction to the limit is `shrink_to`'s job.
    pub(crate) fn finish(
        &mut self,
        request: RequestId,
        prompt_tokens: u32,
        now_s: f64,
    ) -> PrefixCacheInsertResult {
        let Some(active) = self.active.remove(&request) else {
            return PrefixCacheInsertResult {
                mutations: Vec::new(),
                retained_tokens: 0,
            };
        };
        let cache_used_before = self.used_tokens;
        let session_id = active.session;
        // The hard tier already released this request: its path counts here again.
        let mut cursor = active.prefix_node;
        while cursor != NO_NODE {
            self.unlock(cursor);
            cursor = self.nodes[cursor as usize].parent;
        }
        let page = self.config.page_tokens;
        let end = prompt_tokens / page * page;
        let tip = if end > active.prefix_tokens {
            self.new_node(
                session_id,
                active.prefix_node,
                active.prefix_tokens,
                end - active.prefix_tokens,
            )
        } else {
            active.prefix_node
        };
        if tip != NO_NODE {
            // Whatever the request computed on the old path is resident again.
            let mut cursor = tip;
            while cursor != NO_NODE {
                if self.nodes[cursor as usize].kv == ON_HOST {
                    self.duplicates.push_back(cursor);
                }
                self.set_kv(cursor, HBM, true);
                self.touch(cursor);
                cursor = self.nodes[cursor as usize].parent;
            }
            for position in &active.chunk_ends {
                self.place_state(tip, position / page * page, active.hit_tokens);
            }
            self.place_state(tip, active.prefix_tokens, active.hit_tokens);
            if let Some(interval) = self.sizes.checkpoint_interval {
                let mut position = (active.hit_tokens / interval + 1) * interval;
                while position < end {
                    self.place_state(tip, position, active.hit_tokens);
                    position += interval;
                }
            }
            self.place_state(tip, end, active.hit_tokens);
            let session = self.sessions.get_mut(&session_id).expect("session exists");
            session.tip = Some(tip);
            session.finished_at_s = now_s;
            session.round_tips.insert(active.round, tip);
            if self.config.write_through {
                self.copy_chain_to_host(tip, now_s);
            }
            if self.config.policy == ContextCachePolicy::PriceGraph {
                self.price_finish(session_id, tip, active.prefix_node, now_s);
            }
        }
        self.counters.samples += 1;
        for tier in [HBM, HOST] {
            self.counters.kv_byte_samples[tier] += self.kv_bytes[tier] as f64;
            self.counters.state_byte_samples[tier] += self.state_bytes[tier] as f64;
        }
        let mut mutations = std::mem::take(&mut self.receipts);
        let cache_used_after = self.used_tokens;
        let retained_tokens = cache_used_after.saturating_sub(cache_used_before);
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: cache_used_before.saturating_sub(cache_used_after),
            entry_tokens_after: retained_tokens,
            cache_used_before,
            cache_used_after,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens,
        }
    }

    /// Evict or demote until the HBM data fits `limit_units`.
    pub(crate) fn shrink_to(&mut self, limit_units: u64, now_s: f64) -> Vec<PrefixCacheMutation> {
        let limit_bytes = limit_units.saturating_mul(self.sizes.bytes_per_unit);
        match self.config.policy {
            ContextCachePolicy::Lru => self.lru_shrink(limit_bytes, now_s),
            ContextCachePolicy::PriceGraph => self.price_shrink(limit_bytes, now_s),
        }
        std::mem::take(&mut self.receipts)
    }

    fn log_counters(&self) {
        let counters = self.counters;
        let mean = |bytes: f64| bytes / counters.samples.max(1) as f64 / 1e9;
        tracing::info!(
            hits_on_hbm = counters.hits_on_hbm,
            hits_on_host = counters.hits_on_host,
            misses = counters.misses,
            hit_tokens = counters.hit_tokens,
            declared_tokens = counters.declared_tokens,
            loaded_bytes = counters.loaded_bytes,
            demoted_bytes = counters.demoted_bytes,
            mean_hbm_kv_gb = mean(counters.kv_byte_samples[HBM]),
            mean_hbm_state_gb = mean(counters.state_byte_samples[HBM]),
            mean_host_kv_gb = mean(counters.kv_byte_samples[HOST]),
            mean_host_state_gb = mean(counters.state_byte_samples[HOST]),
            "context cache"
        );
    }

    // ---------------------------------------------------------------- planning

    fn context_tip(&self, session_id: u32, context: ContextRef) -> Option<u32> {
        let session = self.sessions.get(&session_id)?;
        match context.prefix_round {
            Some(round) => session.round_tips.get(&round).copied().or(session.tip),
            None => session.tip,
        }
    }

    fn plan(&self, session_id: u32, context: ContextRef, declared: u32) -> Plan {
        let Some(tip) = self.context_tip(session_id, context) else {
            return Plan::default();
        };
        let mut chain = Vec::new();
        let mut cursor = tip;
        while cursor != NO_NODE {
            chain.push(cursor);
            cursor = self.nodes[cursor as usize].parent;
        }
        let mut best = Plan::default();
        let mut host_kv_tokens = 0;
        for &id in chain.iter().rev() {
            let node = &self.nodes[id as usize];
            if node.end() > declared || node.kv == 0 {
                break;
            }
            if node.kv & ON_HBM == 0 {
                host_kv_tokens += node.tokens;
            }
            if node.state != 0 {
                best = Plan {
                    hit_tokens: node.end(),
                    host_kv_tokens,
                    host_state: node.state & ON_HBM == 0,
                };
            }
        }
        best
    }

    // ---------------------------------------------------------------- tree

    fn new_node(&mut self, session_id: u32, parent: u32, start: u32, tokens: u32) -> u32 {
        debug_assert!(tokens > 0);
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            session: session_id,
            parent,
            start,
            tokens,
            kv: 0,
            state: 0,
            locks: 0,
            last_used: 0,
            kv_children: [0; 2],
            demand: None,
            slot: 0,
            host_ready_s: 0.0,
        });
        let session = self.sessions.get_mut(&session_id).expect("session exists");
        session.nodes.push(id);
        session.nodes_sorted = false;
        id
    }

    /// The node on `tip`'s chain ending exactly at `position`, splitting one if
    /// needed. Node ids of the upper part stay, so tips and demand stay valid.
    fn boundary_on_chain(&mut self, tip: u32, position: u32) -> u32 {
        let mut cursor = tip;
        loop {
            let node = &self.nodes[cursor as usize];
            if node.end() == position {
                return cursor;
            }
            if node.start < position && position < node.end() {
                return self.split_below(cursor, position);
            }
            if node.parent == NO_NODE {
                panic!("position {position} is not on the chain of node {tip}");
            }
            cursor = node.parent;
        }
    }

    /// Split `id` at `position`: a new lower node takes `[start, position)`.
    fn split_below(&mut self, id: u32, position: u32) -> u32 {
        let node = self.nodes[id as usize].clone();
        let lower_tokens = position - node.start;
        let lower = self.new_node(node.session, node.parent, node.start, lower_tokens);
        {
            let lower_node = &mut self.nodes[lower as usize];
            lower_node.kv = node.kv;
            lower_node.locks = node.locks;
            lower_node.last_used = node.last_used;
            lower_node.host_ready_s = node.host_ready_s;
            lower_node.kv_children = [
                u32::from(node.kv & ON_HBM != 0),
                u32::from(node.kv & ON_HOST != 0),
            ];
        }
        let upper = &mut self.nodes[id as usize];
        upper.parent = lower;
        upper.start = position;
        upper.tokens -= lower_tokens;
        // The old children of the parent keep their count: one child became
        // another. Every other child of `id` hangs below the upper part.
        self.lru_note_split(lower);
        lower
    }

    /// Place a state at `position` on `tip`'s chain if the request computed it.
    fn place_state(&mut self, tip: u32, position: u32, hit_tokens: u32) {
        if position == 0 || position <= hit_tokens {
            return;
        }
        let end = self.nodes[tip as usize].end();
        if position > end {
            return;
        }
        let node = self.boundary_on_chain(tip, position);
        self.set_state(node, HBM, true);
    }

    fn sorted_nodes(&mut self, session_id: u32) -> Vec<u32> {
        let session = self.sessions.get_mut(&session_id).expect("session exists");
        if !session.nodes_sorted {
            let nodes = &self.nodes;
            session
                .nodes
                .sort_unstable_by_key(|&id| (nodes[id as usize].start, id));
            session.nodes_sorted = true;
            for (slot, &id) in session.nodes.iter().enumerate() {
                self.nodes[id as usize].slot = slot as u32;
            }
        }
        session.nodes.clone()
    }

    // ---------------------------------------------------------------- residency

    /// Whether the HBM bytes of `id` are the cache's rather than the hard tier's.
    fn counts_on_hbm(&self, id: u32) -> bool {
        self.nodes[id as usize].locks == 0
    }

    fn set_kv(&mut self, id: u32, tier: usize, present: bool) {
        let bit = 1u8 << tier;
        let node = &self.nodes[id as usize];
        if (node.kv & bit != 0) == present {
            return;
        }
        let tokens = u64::from(node.tokens);
        let parent = node.parent;
        let bytes = tokens * self.sizes.bytes_per_token;
        let counted = tier == HOST || self.counts_on_hbm(id);
        let node = &mut self.nodes[id as usize];
        if present {
            node.kv |= bit;
        } else {
            node.kv &= !bit;
        }
        if counted {
            if present {
                self.kv_bytes[tier] += bytes;
            } else {
                self.kv_bytes[tier] -= bytes;
            }
            if tier == HBM {
                if present {
                    self.used_tokens += tokens;
                } else {
                    self.used_tokens -= tokens;
                }
            }
        }
        if parent != NO_NODE {
            let count = &mut self.nodes[parent as usize].kv_children[tier];
            *count = if present {
                *count + 1
            } else {
                count.saturating_sub(1)
            };
        }
        if !present && parent != NO_NODE {
            self.lru_push(parent);
        }
        if present {
            self.lru_push(id);
        }
    }

    fn set_state(&mut self, id: u32, tier: usize, present: bool) {
        let bit = 1u8 << tier;
        let node = &self.nodes[id as usize];
        if (node.state & bit != 0) == present {
            return;
        }
        let counted = tier == HOST || self.counts_on_hbm(id);
        let node = &mut self.nodes[id as usize];
        if present {
            node.state |= bit;
        } else {
            node.state &= !bit;
        }
        if counted {
            if present {
                self.state_bytes[tier] += self.sizes.bytes_per_state;
            } else {
                self.state_bytes[tier] -= self.sizes.bytes_per_state;
            }
        }
        if present {
            self.lru_push(id);
        }
    }

    fn lock(&mut self, id: u32) {
        let node = &mut self.nodes[id as usize];
        node.locks += 1;
        if node.locks == 1 {
            let (kv, state, tokens) = (node.kv, node.state, u64::from(node.tokens));
            if kv & ON_HBM != 0 {
                self.kv_bytes[HBM] -= tokens * self.sizes.bytes_per_token;
                self.used_tokens -= tokens;
            }
            if state & ON_HBM != 0 {
                self.state_bytes[HBM] -= self.sizes.bytes_per_state;
            }
        }
    }

    fn unlock(&mut self, id: u32) {
        let node = &mut self.nodes[id as usize];
        node.locks -= 1;
        if node.locks == 0 {
            let (kv, state, tokens) = (node.kv, node.state, u64::from(node.tokens));
            if kv & ON_HBM != 0 {
                self.kv_bytes[HBM] += tokens * self.sizes.bytes_per_token;
                self.used_tokens += tokens;
            }
            if state & ON_HBM != 0 {
                self.state_bytes[HBM] += self.sizes.bytes_per_state;
            }
        }
    }

    fn touch(&mut self, id: u32) {
        self.use_counter += 1;
        self.nodes[id as usize].last_used = self.use_counter;
        self.lru_push(id);
    }

    fn host_fits(&self, bytes: u64) -> bool {
        self.kv_bytes[HOST] + self.state_bytes[HOST] + bytes <= self.config.host_capacity_bytes
    }

    /// One HBM eviction's receipt: `tokens_before` used tokens became the current.
    fn record_eviction(&mut self, session_id: u32, used_before: u64) {
        self.receipts.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::CapacityEviction,
            session_id,
            entry_tokens_before: used_before - self.used_tokens,
            entry_tokens_after: 0,
            cache_used_before: used_before,
            cache_used_after: self.used_tokens,
        });
    }

    /// Host copies of data HBM also holds are worth nothing: drop those first.
    fn reclaim_duplicates(&mut self, needed: u64) {
        while !self.host_fits(needed) {
            let Some(id) = self.duplicates.pop_front() else {
                return;
            };
            let node = &self.nodes[id as usize];
            let (kv, state) = (node.kv, node.state);
            if kv == ON_HBM | ON_HOST {
                self.set_kv(id, HOST, false);
            }
            if state == ON_HBM | ON_HOST {
                self.set_state(id, HOST, false);
            }
        }
    }

    /// Account an asynchronous D2H of `bytes`; returns when it lands.
    fn queue_demotion(&mut self, bytes: u64, now_s: f64) -> f64 {
        self.counters.demoted_bytes += bytes;
        let start = self.d2h_free_s.max(now_s);
        self.d2h_free_s = start + bytes as f64 / self.config.link_bytes_per_s;
        self.d2h_free_s
    }

    /// Bytes to move node `id`'s HBM-only data to the host, if the policy wants it.
    fn demotable_bytes(&self, id: u32) -> u64 {
        let node = &self.nodes[id as usize];
        let mut bytes = 0;
        if node.kv == ON_HBM {
            bytes += u64::from(node.tokens) * self.sizes.bytes_per_token;
        }
        if node.state == ON_HBM {
            bytes += self.sizes.bytes_per_state;
        }
        bytes
    }

    /// Give node `id` a host copy of its HBM data. The caller made room.
    fn copy_to_host(&mut self, id: u32, now_s: f64) {
        let bytes = self.demotable_bytes(id);
        if bytes == 0 {
            return;
        }
        let ready = self.queue_demotion(bytes, now_s);
        self.nodes[id as usize].host_ready_s = ready;
        let node = &self.nodes[id as usize];
        let (kv, state) = (node.kv, node.state);
        if kv == ON_HBM {
            self.set_kv(id, HOST, true);
        }
        if state == ON_HBM {
            self.set_state(id, HOST, true);
        }
    }

    fn copy_chain_to_host(&mut self, tip: u32, now_s: f64) {
        let mut cursor = tip;
        while cursor != NO_NODE {
            let bytes = self.demotable_bytes(cursor);
            if bytes > 0 && self.lru_host_room(cursor) {
                self.copy_to_host(cursor, now_s);
            }
            cursor = self.nodes[cursor as usize].parent;
        }
    }

    #[cfg(test)]
    fn bytes_consistent(&self) -> bool {
        let mut kv = [0u64; 2];
        let mut state = [0u64; 2];
        for node in &self.nodes {
            for tier in [HBM, HOST] {
                let counted = tier == HOST || node.locks == 0;
                if counted && node.kv & (1 << tier) != 0 {
                    kv[tier] += u64::from(node.tokens) * self.sizes.bytes_per_token;
                }
                if counted && node.state & (1 << tier) != 0 {
                    state[tier] += self.sizes.bytes_per_state;
                }
            }
        }
        kv == self.kv_bytes && state == self.state_bytes
    }
}

impl Drop for ContextCache {
    fn drop(&mut self) {
        self.log_counters();
    }
}

type MinHeap<T> = BinaryHeap<std::cmp::Reverse<T>>;

#[cfg(test)]
mod tests;
