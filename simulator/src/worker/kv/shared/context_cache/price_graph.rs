//! PriceGraph: online cache decisions by value per byte.
//!
//! A *context* is what one finished call leaves: a chain of KV and the states on
//! it. Its value is `sum_q w_q * saving(q)` over the positions `q` a later call may
//! land on: `w_q` is the return density from [`Survival`], and `saving(q)` is the
//! recompute time of the prefix up to the deepest usable state at or below `q`,
//! minus the time to load whatever of it the host holds.
//!
//! - At a call's finish, a DP over the context's states picks the subset and the
//!   tier maximising `value - lambda_tier * bytes`; what is not picked leaves.
//! - Each session has one item per tier, keyed by its net value density, in a
//!   heap. Eviction pops the minimum: an HBM item moves to the host if its data
//!   there beats the host price, else it is dropped.
//! - `lambda_tier` is an EWMA of the densities popped from that tier.
//! - The horizon `tau_tier` over which a return counts is that tier's turnover
//!   time, by Little's law.
//!
//! No decision reads the future, the kind of a response, or where a call will land.

use std::cmp::Reverse;
use std::collections::VecDeque;

use super::survival::Ordered;
use super::{
    class_of, ContextCache, Demand, MinHeap, HBM, HOST, LOAD_SETUP_S, NO_NODE, ON_HBM, ON_HOST,
};

/// Fraction of a tier's bytes one EWMA time constant of `lambda` spans.
const PRICE_EWMA_BYTES: f64 = 0.05;
/// Cap on a return density, against a horizon near zero.
const MAX_DENSITY: f64 = 1e3;

#[derive(Clone, Copy, Debug)]
pub(super) struct SessionPricing {
    key: [f64; 2],
    version: [u64; 2],
    rekey_generation: u64,
}

impl Default for SessionPricing {
    fn default() -> Self {
        Self {
            key: [f64::NAN; 2],
            version: [0; 2],
            rekey_generation: 0,
        }
    }
}

pub(super) struct PriceGraph {
    /// Session ids in arrival order: a rebuild's order must be deterministic.
    sessions: Vec<u32>,
    lambda: [f64; 2],
    tau: [f64; 2],
    first_finish_s: Option<f64>,
    /// `(time, cumulative bytes popped)` over the last tier capacity.
    turnover: [VecDeque<(f64, f64)>; 2],
    popped_bytes: [f64; 2],
    heap: [MinHeap<(Ordered, u64, u32)>; 2],
    rekeys: MinHeap<(Ordered, u64, u32)>,
    now_s: f64,
}

impl PriceGraph {
    pub(super) fn new() -> Self {
        Self {
            sessions: Vec::new(),
            lambda: [0.0; 2],
            tau: [0.0; 2],
            first_finish_s: None,
            turnover: Default::default(),
            popped_bytes: [0.0; 2],
            heap: Default::default(),
            rekeys: Default::default(),
            now_s: 0.0,
        }
    }

    pub(super) fn register_session(&mut self, session: u32) {
        self.sessions.push(session);
    }
}

/// What removing a session's data would do, for [`ContextCache::evaluate`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Removal {
    None,
    DropHbm,
    DemoteHbm,
    DropHost,
    /// Everything off the chain marked in the scratch vector.
    DropOffChain,
}

#[derive(Clone, Copy, Debug, Default)]
struct Key {
    bytes: [u64; 2],
    bytes_added_to_host: u64,
    key: [f64; 2],
    host_alternative: f64,
    host_alternative_net: f64,
}

struct Chain {
    nodes: Vec<u32>,
    state_positions: Vec<i64>,
    state_nodes: Vec<u32>,
    savings: [Vec<f64>; 2],
    landings: [Vec<(i64, f64)>; 2],
}

/// Pick the states of one context in one tier: the best `value - lambda * bytes`,
/// where a landing is served by the deepest kept state at or below it. Returns
/// the score (0 = keep nothing) and the chosen indices, ascending.
fn choose_states(
    positions: &[i64],
    savings: &[f64],
    landings: &[(i64, f64)],
    lambda: f64,
    bytes_per_token: f64,
    bytes_per_state: f64,
) -> (f64, Vec<usize>) {
    let count = positions.len();
    let mut cumulative = Vec::with_capacity(landings.len() + 1);
    cumulative.push(0.0f64);
    for &(_, weight) in landings {
        cumulative.push(cumulative[cumulative.len() - 1] + weight);
    }
    let total = cumulative[cumulative.len() - 1];
    let below = |position: i64| cumulative[landings.partition_point(|x| x.0 < position)];
    let mut best = (0.0f64, usize::MAX);
    let mut gain = vec![0.0f64; count];
    let mut previous = vec![usize::MAX; count];
    for j in 0..count {
        let tail = savings[j] * (total - below(positions[j]));
        let mut through = 0.0f64;
        let mut through_index = usize::MAX;
        for i in 0..j {
            let value = gain[i] + savings[i] * (below(positions[j]) - below(positions[i]));
            if value > through {
                through = value;
                through_index = i;
            }
        }
        gain[j] = through - lambda * bytes_per_state;
        previous[j] = through_index;
        let score = gain[j] + tail - lambda * bytes_per_token * positions[j] as f64;
        if score > best.0 {
            best = (score, j);
        }
    }
    let mut chosen = Vec::new();
    let mut j = best.1;
    while j != usize::MAX {
        chosen.push(j);
        j = previous[j];
    }
    chosen.reverse();
    (best.0, chosen)
}

impl ContextCache {
    // ------------------------------------------------------------ demand

    /// A request landed on `node`'s end: the wait of the point there is over.
    pub(super) fn note_arrival(&mut self, node: u32, now_s: f64) {
        self.pricing.now_s = now_s;
        if node != NO_NODE {
            let demand = self.nodes[node as usize].demand.take();
            let updated = match demand {
                Some(mut demand) => {
                    if let Some(spell) = demand.spell.take() {
                        self.survival.close(spell, now_s, true);
                    }
                    demand.class &= !1;
                    demand.running += 1;
                    demand
                }
                None => Demand {
                    class: class_of(false, false),
                    last_finish: now_s,
                    spell: None,
                    running: 1,
                },
            };
            self.nodes[node as usize].demand = Some(updated);
        }
        if self.survival.wants_rebuild(now_s) {
            self.survival.rebuild(now_s);
            self.rekey_all();
        }
    }

    /// The call finished at `tip`: `tip` becomes a tail point and every idle
    /// point below it a mark, with its own, usually much lower, return rate.
    fn note_finish(&mut self, prefix_node: u32, tip: u32, now_s: f64) {
        if prefix_node != NO_NODE {
            if let Some(demand) = self.nodes[prefix_node as usize].demand.as_mut() {
                demand.running = demand.running.saturating_sub(1);
                demand.last_finish = now_s;
            }
        }
        let class = match self.nodes[tip as usize].demand {
            Some(demand) if demand.running == 0 && tip == prefix_node => demand.class & !1,
            _ => class_of(true, false),
        };
        if self.nodes[tip as usize]
            .demand
            .is_none_or(|d| d.running == 0)
        {
            if let Some(spell) = self.nodes[tip as usize].demand.and_then(|d| d.spell) {
                self.survival.close(spell, now_s, false);
            }
            let spell = self.survival.open(class, now_s, now_s);
            self.nodes[tip as usize].demand = Some(Demand {
                class,
                last_finish: now_s,
                spell: Some(spell),
                running: 0,
            });
        }
        let mut cursor = self.nodes[tip as usize].parent;
        while cursor != NO_NODE {
            if let Some(demand) = self.nodes[cursor as usize].demand {
                if demand.running == 0 && demand.class & 1 == 0 {
                    if let Some(spell) = demand.spell {
                        self.survival.close(spell, now_s, false);
                    }
                    let class = demand.class | 1;
                    let spell = self.survival.open(class, demand.last_finish, now_s);
                    self.nodes[cursor as usize].demand = Some(Demand {
                        class,
                        spell: Some(spell),
                        ..demand
                    });
                }
            }
            cursor = self.nodes[cursor as usize].parent;
        }
    }

    fn density(&self, demand: &Demand, tier: usize) -> f64 {
        if demand.running > 0 {
            return 0.0;
        }
        let density = self.survival.return_density(
            demand.class,
            self.pricing.now_s - demand.last_finish,
            self.pricing.tau[tier],
        );
        if density.is_finite() {
            density.min(MAX_DENSITY)
        } else {
            MAX_DENSITY
        }
    }

    fn recompute_s(&self, position: i64) -> f64 {
        let position = position as f64;
        self.config.recompute_alpha_s * position
            + self.config.recompute_beta_s * position * position * 0.5
    }

    fn kv_load_s(&self, tokens: i64) -> f64 {
        tokens as f64 * self.sizes.bytes_per_token as f64 / self.config.link_bytes_per_s
    }

    fn state_load_s(&self) -> f64 {
        self.sizes.bytes_per_state as f64 / self.config.link_bytes_per_s
    }

    // ------------------------------------------------------------ valuation

    /// The session's value under `removal`, with the HBM and the host horizon.
    /// Only unlocked data can be removed.
    fn evaluate(&mut self, session_id: u32, removal: Removal, on_chain: &[bool]) -> [f64; 2] {
        self.sorted_nodes(session_id);
        let session = self.sessions.get_mut(&session_id).expect("session exists");
        let ids = std::mem::take(&mut session.nodes);
        // (covered through the node end, deepest state position, its saving,
        // host-only tokens below)
        let mut scan: Vec<(bool, i64, f64, i64)> = Vec::with_capacity(ids.len());
        let mut value = [0.0; 2];
        for (slot, &id) in ids.iter().enumerate() {
            let node = &self.nodes[id as usize];
            let (above_covered, base_position, base_saving, host_below) = if node.parent == NO_NODE
            {
                (true, 0, 0.0, 0)
            } else {
                scan[self.nodes[node.parent as usize].slot as usize]
            };
            let removable = node.locks == 0
                && match removal {
                    Removal::None => false,
                    Removal::DropOffChain => !on_chain[slot],
                    _ => true,
                };
            let (kv, state) = if !removable {
                (node.kv, node.state)
            } else {
                match removal {
                    Removal::DropHbm => (node.kv & !ON_HBM, node.state & !ON_HBM),
                    Removal::DemoteHbm => (
                        if node.kv & ON_HBM != 0 {
                            ON_HOST
                        } else {
                            node.kv
                        },
                        if node.state & ON_HBM != 0 {
                            ON_HOST
                        } else {
                            node.state
                        },
                    ),
                    Removal::DropHost => (node.kv & !ON_HOST, node.state & !ON_HOST),
                    _ => (0, 0),
                }
            };
            let covered = above_covered && kv != 0;
            let tokens = i64::from(node.tokens);
            let host_through = host_below + if kv & ON_HBM == 0 { tokens } else { 0 };
            let end = i64::from(node.end());
            let (position, saving) = if covered && state != 0 {
                let load = self.kv_load_s(host_through)
                    + if state & ON_HBM == 0 {
                        self.state_load_s()
                    } else {
                        0.0
                    };
                let saving =
                    self.recompute_s(end) - if load > 0.0 { load + LOAD_SETUP_S } else { 0.0 };
                (end, saving)
            } else {
                (base_position, base_saving)
            };
            if let Some(demand) = node.demand {
                if saving != 0.0 {
                    for (tier, slot_value) in value.iter_mut().enumerate() {
                        *slot_value += self.density(&demand, tier) * saving;
                    }
                }
            }
            scan.push((covered, position, saving, host_through));
        }
        self.sessions.get_mut(&session_id).unwrap().nodes = ids;
        value
    }

    fn removable_bytes(&self, session_id: u32) -> ([u64; 2], u64) {
        let (mut bytes, mut added) = ([0u64; 2], 0u64);
        for &id in &self.sessions[&session_id].nodes {
            let node = &self.nodes[id as usize];
            if node.locks != 0 {
                continue;
            }
            for tier in [HBM, HOST] {
                if node.kv & (1 << tier) != 0 {
                    bytes[tier] += u64::from(node.tokens) * self.sizes.bytes_per_token;
                }
                if node.state & (1 << tier) != 0 {
                    bytes[tier] += self.sizes.bytes_per_state;
                }
            }
            added += self.demotable_bytes(id);
        }
        (bytes, added)
    }

    fn compute_key(&mut self, session_id: u32) -> Key {
        let (bytes, bytes_added_to_host) = self.removable_bytes(session_id);
        let mut key = Key {
            bytes,
            bytes_added_to_host,
            ..Default::default()
        };
        if bytes[HBM] == 0 && bytes[HOST] == 0 {
            return key;
        }
        let current = self.evaluate(session_id, Removal::None, &[]);
        if bytes[HBM] > 0 {
            let dropped = self.evaluate(session_id, Removal::DropHbm, &[]);
            let demoted = self.evaluate(session_id, Removal::DemoteHbm, &[]);
            let loss = current[HBM] - dropped[HBM];
            key.host_alternative = demoted[HOST] - dropped[HOST];
            key.host_alternative_net =
                key.host_alternative - self.pricing.lambda[HOST] * bytes_added_to_host as f64;
            // A demotable item loses only what the host copy would not keep.
            let keeps = if key.host_alternative_net > 0.0 {
                demoted[HBM] - dropped[HBM]
            } else {
                0.0
            };
            key.key[HBM] = (loss - keeps) / bytes[HBM] as f64;
        }
        if bytes[HOST] > 0 {
            let dropped = self.evaluate(session_id, Removal::DropHost, &[]);
            key.key[HOST] = (current[HOST] - dropped[HOST]) / bytes[HOST] as f64;
        }
        for value in &mut key.key {
            if value.is_nan() {
                *value = 0.0;
            }
        }
        key
    }

    // ------------------------------------------------------------ heaps

    fn set_key(&mut self, session_id: u32, tier: usize, key: Option<f64>) {
        let pricing = &mut self.sessions.get_mut(&session_id).unwrap().pricing;
        pricing.version[tier] += 1;
        let version = pricing.version[tier];
        match key {
            Some(value) => {
                pricing.key[tier] = value;
                self.pricing.heap[tier].push(Reverse((Ordered(value), version, session_id)));
            }
            None => pricing.key[tier] = f64::NAN,
        }
    }

    /// Give the session a fresh key and schedule the next one, one octave of idle
    /// age later: its value moves with the age bins of the life table.
    fn rekey(&mut self, session_id: u32) -> Key {
        let key = self.compute_key(session_id);
        for tier in [HBM, HOST] {
            self.set_key(
                session_id,
                tier,
                (key.bytes[tier] > 0).then_some(key.key[tier]),
            );
        }
        let now_s = self.pricing.now_s;
        let session = self.sessions.get_mut(&session_id).unwrap();
        let age = now_s - session.finished_at_s;
        let next_age = if age < 1.0 {
            1.0
        } else {
            (1u64 << (u64::BITS - (age as u64).leading_zeros()).min(62)) as f64
        };
        session.pricing.rekey_generation += 1;
        if key.bytes[HBM] > 0 || key.bytes[HOST] > 0 {
            self.pricing.rekeys.push(Reverse((
                Ordered(session.finished_at_s + next_age),
                session.pricing.rekey_generation,
                session_id,
            )));
        }
        key
    }

    fn rekey_all(&mut self) {
        for index in 0..self.pricing.sessions.len() {
            let session_id = self.pricing.sessions[index];
            let key = self.sessions[&session_id].pricing.key;
            if !key[HBM].is_nan() || !key[HOST].is_nan() {
                self.rekey(session_id);
            }
        }
    }

    fn rekey_due(&mut self) {
        let now_s = self.pricing.now_s;
        while let Some(&Reverse((Ordered(due), generation, session_id))) =
            self.pricing.rekeys.peek()
        {
            if due > now_s {
                break;
            }
            self.pricing.rekeys.pop();
            if self.sessions[&session_id].pricing.rekey_generation == generation {
                self.rekey(session_id);
            }
        }
    }

    // ------------------------------------------------------------ prices

    fn tier_capacity(&self, tier: usize) -> f64 {
        if tier == HBM {
            (self.sizes.capacity_units * self.sizes.bytes_per_unit) as f64
        } else {
            self.config.host_capacity_bytes as f64
        }
    }

    fn update_price(&mut self, tier: usize, density: f64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let rate = (bytes as f64 / (PRICE_EWMA_BYTES * self.tier_capacity(tier))).min(1.0);
        let density = if density.is_finite() {
            density.max(0.0)
        } else {
            0.0
        };
        self.pricing.lambda[tier] = (1.0 - rate) * self.pricing.lambda[tier] + rate * density;
    }

    /// Little's law on the bytes popped from `tier` over one capacity's worth.
    fn note_popped(&mut self, tier: usize, bytes: u64) {
        let now_s = self.pricing.now_s;
        let capacity = self.tier_capacity(tier);
        if self.pricing.turnover[tier].is_empty() {
            self.pricing.turnover[tier].push_back((now_s, 0.0));
        }
        self.pricing.popped_bytes[tier] += bytes as f64;
        let popped = self.pricing.popped_bytes[tier];
        let turnover = &mut self.pricing.turnover[tier];
        turnover.push_back((now_s, popped));
        while turnover.len() > 1 && turnover[1].1 <= popped - capacity {
            turnover.pop_front();
        }
        self.update_horizon(tier);
    }

    fn update_horizon(&mut self, tier: usize) {
        let now_s = self.pricing.now_s;
        let elapsed = self
            .pricing
            .first_finish_s
            .map_or(0.0, |first| now_s - first);
        let capacity = self.tier_capacity(tier);
        let popped = self.pricing.popped_bytes[tier];
        self.pricing.tau[tier] = match self.pricing.turnover[tier].front() {
            Some(&(since, popped_then)) if popped >= capacity && popped > popped_then => {
                (capacity * (now_s - since) / (popped - popped_then)).min(elapsed)
            }
            _ => elapsed,
        };
    }

    // ------------------------------------------------------------ eviction

    pub(super) fn price_shrink(&mut self, limit_bytes: u64, now_s: f64) {
        self.pricing.now_s = now_s;
        while self.kv_bytes[HBM] + self.state_bytes[HBM] > limit_bytes {
            if !self.pop_hbm() {
                break;
            }
        }
    }

    /// Evict the HBM item of the lowest net density.
    fn pop_hbm(&mut self) -> bool {
        self.rekey_due();
        loop {
            let Some(Reverse((Ordered(stored), version, session_id))) =
                self.pricing.heap[HBM].pop()
            else {
                return false;
            };
            if self.sessions[&session_id].pricing.version[HBM] != version {
                continue;
            }
            let key = self.compute_key(session_id);
            if key.bytes[HBM] == 0 {
                self.set_key(session_id, HBM, None);
                continue;
            }
            if key.key[HBM] > stored + 1e-12 * stored.abs() {
                self.set_key(session_id, HBM, Some(key.key[HBM]));
                continue;
            }
            self.sessions.get_mut(&session_id).unwrap().pricing.version[HBM] += 1;
            self.sessions.get_mut(&session_id).unwrap().pricing.key[HBM] = f64::NAN;
            let mut density = if key.bytes_added_to_host > 0 {
                key.host_alternative / key.bytes_added_to_host as f64
            } else {
                f64::INFINITY
            };
            let mut keep_going = key.host_alternative_net > 0.0;
            if keep_going {
                // The context's own choice at the host price goes first.
                if let Some(tip) = self.sessions[&session_id].tip {
                    let chain = self.build_chain(session_id, tip);
                    let (score, chosen) = choose_states(
                        &chain.state_positions,
                        &chain.savings[HOST],
                        &chain.landings[HOST],
                        self.pricing.lambda[HOST],
                        self.sizes.bytes_per_token as f64,
                        self.sizes.bytes_per_state as f64,
                    );
                    if chosen.is_empty() || score <= 0.0 {
                        keep_going = false;
                    } else {
                        let top = chain.state_positions[*chosen.last().unwrap()];
                        let bytes = (self.sizes.bytes_per_token as i64 * top
                            + self.sizes.bytes_per_state as i64 * chosen.len() as i64)
                            as f64;
                        density = (score + self.pricing.lambda[HOST] * bytes) / bytes;
                        self.keep_chosen(&chain, &chosen);
                        self.sweep(session_id);
                    }
                }
            }
            if keep_going {
                self.demote_session(session_id, density);
            }
            let used_before = self.used_tokens;
            let freed = self.drop_tier(session_id, HBM);
            self.record_eviction(session_id, used_before);
            self.note_popped(HBM, freed);
            self.update_price(HBM, key.key[HBM], freed);
            self.sweep(session_id);
            self.rekey(session_id);
            return true;
        }
    }

    /// Host room for bytes of `density`; host items below it go, `me` never.
    fn host_room(&mut self, bytes: u64, density: f64, me: u32) -> bool {
        self.reclaim_duplicates(bytes);
        let mut aside = Vec::new();
        let mut ok = true;
        while !self.host_fits(bytes) {
            self.rekey_due();
            let Some(entry) = self.pricing.heap[HOST].pop() else {
                ok = false;
                break;
            };
            let Reverse((Ordered(stored), version, session_id)) = entry;
            if self.sessions[&session_id].pricing.version[HOST] != version {
                continue;
            }
            if session_id == me {
                aside.push(entry);
                continue;
            }
            let key = self.compute_key(session_id);
            if key.bytes[HOST] == 0 {
                self.set_key(session_id, HOST, None);
                continue;
            }
            if key.key[HOST] > stored + 1e-12 * stored.abs() {
                self.set_key(session_id, HOST, Some(key.key[HOST]));
                continue;
            }
            if key.key[HOST] >= density {
                aside.push(entry);
                ok = false;
                break;
            }
            let freed = self.drop_tier(session_id, HOST);
            self.note_popped(HOST, freed);
            self.update_price(HOST, key.key[HOST], freed);
            self.sweep(session_id);
            self.rekey(session_id);
        }
        for entry in aside {
            self.pricing.heap[HOST].push(entry);
        }
        ok && self.host_fits(bytes)
    }

    /// Move the session's unlocked HBM data to the host.
    fn demote_session(&mut self, session_id: u32, density: f64) -> bool {
        let (_, bytes) = self.removable_bytes(session_id);
        if bytes == 0 {
            return true;
        }
        if !self.host_fits(bytes) && !self.host_room(bytes, density, session_id) {
            return false;
        }
        let now_s = self.pricing.now_s;
        for id in self.sessions[&session_id].nodes.clone() {
            if self.nodes[id as usize].locks == 0 {
                self.copy_to_host(id, now_s);
            }
        }
        true
    }

    fn drop_tier(&mut self, session_id: u32, tier: usize) -> u64 {
        let before = self.kv_bytes[tier] + self.state_bytes[tier];
        for id in self.sessions[&session_id].nodes.clone() {
            if self.nodes[id as usize].locks == 0 {
                self.set_kv(id, tier, false);
                self.set_state(id, tier, false);
            }
        }
        before - (self.kv_bytes[tier] + self.state_bytes[tier])
    }

    /// KV no state above it can use, and states missing their KV prefix, leave.
    fn sweep(&mut self, session_id: u32) {
        let ids = self.sorted_nodes(session_id);
        let mut covered = vec![false; ids.len()];
        let mut needed = vec![false; ids.len()];
        for (slot, &id) in ids.iter().enumerate() {
            let node = &self.nodes[id as usize];
            let above =
                node.parent == NO_NODE || covered[self.nodes[node.parent as usize].slot as usize];
            covered[slot] = above && node.kv != 0;
            needed[slot] = covered[slot] && node.state != 0;
        }
        for slot in (0..ids.len()).rev() {
            let node = &self.nodes[ids[slot] as usize];
            if needed[slot] && node.parent != NO_NODE {
                needed[self.nodes[node.parent as usize].slot as usize] = true;
            }
        }
        for (slot, &id) in ids.iter().enumerate() {
            if self.nodes[id as usize].locks != 0 {
                continue;
            }
            for tier in [HBM, HOST] {
                if !needed[slot] {
                    self.set_kv(id, tier, false);
                }
                if !covered[slot] {
                    self.set_state(id, tier, false);
                }
            }
        }
    }

    // ------------------------------------------------------------ finish

    fn build_chain(&mut self, session_id: u32, tip: u32) -> Chain {
        self.sorted_nodes(session_id);
        let mut nodes = Vec::new();
        let mut cursor = tip;
        while cursor != NO_NODE {
            nodes.push(cursor);
            cursor = self.nodes[cursor as usize].parent;
        }
        nodes.reverse();
        let mut chain = Chain {
            nodes,
            state_positions: Vec::new(),
            state_nodes: Vec::new(),
            savings: Default::default(),
            landings: Default::default(),
        };
        let mut covered = true;
        let mut host_tokens = 0i64;
        for index in 0..chain.nodes.len() {
            let id = chain.nodes[index];
            let node = self.nodes[id as usize].clone();
            let end = i64::from(node.end());
            if covered && node.kv != 0 {
                if node.kv & ON_HBM == 0 {
                    host_tokens += i64::from(node.tokens);
                }
                if node.state != 0 {
                    let load = self.kv_load_s(host_tokens)
                        + if node.state & ON_HBM == 0 {
                            self.state_load_s()
                        } else {
                            0.0
                        };
                    let full = self.recompute_s(end);
                    chain.state_positions.push(end);
                    chain.state_nodes.push(id);
                    chain.savings[HBM]
                        .push(full - if load > 0.0 { load + LOAD_SETUP_S } else { 0.0 });
                    chain.savings[HOST]
                        .push(full - (self.kv_load_s(end) + self.state_load_s() + LOAD_SETUP_S));
                }
            } else {
                covered = false;
            }
            if let Some(demand) = node.demand {
                for tier in [HBM, HOST] {
                    chain.landings[tier].push((end, self.density(&demand, tier)));
                }
            }
        }
        chain
    }

    /// Keep KV up to the top chosen state and the chosen states on the chain.
    fn keep_chosen(&mut self, chain: &Chain, chosen: &[usize]) {
        let top = chosen.last().map_or(0, |&j| chain.state_positions[j]);
        let kept: Vec<u32> = chosen.iter().map(|&j| chain.state_nodes[j]).collect();
        for &id in &chain.nodes {
            let node = &self.nodes[id as usize];
            if node.locks != 0 {
                continue;
            }
            let above_top = i64::from(node.start) >= top;
            for tier in [HBM, HOST] {
                if above_top {
                    self.set_kv(id, tier, false);
                }
                if !kept.contains(&id) {
                    self.set_state(id, tier, false);
                }
            }
        }
    }

    pub(super) fn price_finish(&mut self, session_id: u32, tip: u32, prefix_node: u32, now_s: f64) {
        self.pricing.now_s = now_s;
        self.pricing.first_finish_s.get_or_insert(now_s);
        for tier in [HBM, HOST] {
            self.update_horizon(tier);
        }
        self.note_finish(prefix_node, tip, now_s);
        let chain = self.build_chain(session_id, tip);
        let (bytes_per_token, bytes_per_state) = (
            self.sizes.bytes_per_token as f64,
            self.sizes.bytes_per_state as f64,
        );
        let (lambda_hbm, lambda_host) = (self.pricing.lambda[HBM], self.pricing.lambda[HOST]);
        let positions = &chain.state_positions;
        let (hbm_score, hbm_chosen) = choose_states(
            positions,
            &chain.savings[HBM],
            &chain.landings[HBM],
            lambda_hbm,
            bytes_per_token,
            bytes_per_state,
        );
        let (host_score, host_chosen) = choose_states(
            positions,
            &chain.savings[HOST],
            &chain.landings[HOST],
            lambda_host,
            bytes_per_token,
            bytes_per_state,
        );
        // Compare the tiers over one horizon, HBM's; the host's own horizon only
        // says whether the host would take the context at all.
        let host_over_hbm_horizon = if host_score > 0.0 {
            choose_states(
                positions,
                &chain.savings[HOST],
                &chain.landings[HBM],
                lambda_host,
                bytes_per_token,
                bytes_per_state,
            )
            .0
        } else {
            0.0
        };
        let stay_on_hbm =
            hbm_score > 0.0 && (hbm_score >= host_over_hbm_horizon || host_score <= 0.0);
        let (to_host, score, chosen) = if stay_on_hbm {
            (false, hbm_score, hbm_chosen)
        } else if host_score > 0.0 {
            (true, host_score, host_chosen)
        } else {
            (false, 0.0, Vec::new())
        };
        let top = chosen.last().map_or(0, |&j| chain.state_positions[j]);
        let chosen_count = chosen.len();
        self.keep_chosen(&chain, &chosen);
        self.drop_off_chain_if_worthless(session_id, &chain);
        self.sweep(session_id);
        if to_host && chosen_count > 0 {
            let bytes = (self.sizes.bytes_per_token as i64 * top
                + self.sizes.bytes_per_state as i64 * chosen_count as i64)
                as f64;
            let density = (score + lambda_host * bytes) / bytes;
            if self.demote_session(session_id, density) {
                for id in self.sessions[&session_id].nodes.clone() {
                    let node = &self.nodes[id as usize];
                    if node.locks == 0 {
                        if node.kv & ON_HOST != 0 {
                            self.set_kv(id, HBM, false);
                        }
                        if self.nodes[id as usize].state & ON_HOST != 0 {
                            self.set_state(id, HBM, false);
                        }
                    }
                }
            }
        }
        self.rekey(session_id);
    }

    /// Branches a rewind left behind stay only if they are worth their price.
    fn drop_off_chain_if_worthless(&mut self, session_id: u32, chain: &Chain) {
        self.sorted_nodes(session_id);
        let ids = self.sessions[&session_id].nodes.clone();
        if ids.len() <= chain.nodes.len() {
            return;
        }
        let mut on_chain = vec![false; ids.len()];
        for &id in &chain.nodes {
            on_chain[self.nodes[id as usize].slot as usize] = true;
        }
        let (mut hbm_bytes, mut host_bytes) = (0u64, 0u64);
        for (slot, &id) in ids.iter().enumerate() {
            let node = &self.nodes[id as usize];
            if on_chain[slot] || node.locks != 0 {
                continue;
            }
            if node.kv & ON_HBM != 0 {
                hbm_bytes += u64::from(node.tokens) * self.sizes.bytes_per_token;
            }
            if node.kv & ON_HOST != 0 {
                host_bytes += u64::from(node.tokens) * self.sizes.bytes_per_token;
            }
            if node.state & ON_HBM != 0 {
                hbm_bytes += self.sizes.bytes_per_state;
            }
            if node.state & ON_HOST != 0 {
                host_bytes += self.sizes.bytes_per_state;
            }
        }
        if hbm_bytes + host_bytes == 0 {
            return;
        }
        let current = self.evaluate(session_id, Removal::None, &[]);
        let without = self.evaluate(session_id, Removal::DropOffChain, &on_chain);
        let price = self.pricing.lambda[HBM] * hbm_bytes as f64
            + self.pricing.lambda[HOST] * host_bytes as f64;
        if current[HBM] - without[HBM] <= price {
            for (slot, &id) in ids.iter().enumerate() {
                if !on_chain[slot] && self.nodes[id as usize].locks == 0 {
                    for tier in [HBM, HOST] {
                        self.set_kv(id, tier, false);
                        self.set_state(id, tier, false);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(
        positions: &[i64],
        savings: &[f64],
        landings: &[(i64, f64)],
        lambda: f64,
    ) -> f64 {
        let mut best = 0.0f64;
        for mask in 1u32..(1 << positions.len()) {
            let chosen: Vec<usize> = (0..positions.len())
                .filter(|i| mask >> i & 1 == 1)
                .collect();
            let value: f64 = landings
                .iter()
                .map(|&(q, w)| {
                    chosen
                        .iter()
                        .rev()
                        .find(|&&i| positions[i] <= q)
                        .map_or(0.0, |&i| w * savings[i])
                })
                .sum();
            let top = positions[*chosen.last().unwrap()] as f64;
            best = best.max(value - lambda * (2.0 * top + 50.0 * chosen.len() as f64));
        }
        best
    }

    #[test]
    fn the_state_choice_matches_exhaustive_search() {
        let mut seed = 7u64;
        let mut next = move |modulus: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % modulus
        };
        for _ in 0..300 {
            let count = 1 + next(6) as usize;
            let mut positions: Vec<i64> = (0..count).map(|_| 64 * (1 + next(40) as i64)).collect();
            positions.sort_unstable();
            positions.dedup();
            let savings: Vec<f64> = positions.iter().map(|&p| p as f64 * 0.01).collect();
            let mut landings: Vec<(i64, f64)> = (0..4)
                .map(|_| (64 * (1 + next(44) as i64), next(100) as f64 / 10.0))
                .collect();
            landings.sort_by_key(|x| x.0);
            let lambda = next(20) as f64 * 1e-4;
            let (score, _) = choose_states(&positions, &savings, &landings, lambda, 2.0, 50.0);
            let expected = brute_force(&positions, &savings, &landings, lambda);
            assert!((score - expected).abs() < 1e-9, "{score} vs {expected}");
        }
    }
}
