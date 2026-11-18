//! Per-component LRU, the replacement SGLang's unified radix cache runs.
//!
//! The coldest leaf KV goes first and takes the state at its end with it; a
//! state further up the tree only goes when no KV can. With `state_share` each
//! kind is limited to its own pool instead, which is the default static split.
//! An evicted unit is demoted under `write_back` when the host has room.

use std::cmp::Reverse;

use super::{ContextCache, ContextCachePolicy, MinHeap, HBM, HOST};

#[derive(Default)]
pub(super) struct LruHeaps {
    kv: MinHeap<(u64, u32)>,
    state: MinHeap<(u64, u32)>,
}

impl ContextCache {
    /// (Re)list node `id`. Entries are validated when popped, so a changed node
    /// only has to be pushed again: `touch` and every residency gain do.
    pub(super) fn lru_push(&mut self, id: u32) {
        if self.config.policy != ContextCachePolicy::Lru {
            return;
        }
        let node = &self.nodes[id as usize];
        let (last_used, kv, state) = (node.last_used, node.kv, node.state);
        for tier in [HBM, HOST] {
            if kv & (1 << tier) != 0 {
                self.lru[tier].kv.push(Reverse((last_used, id)));
            }
            if state & (1 << tier) != 0 {
                self.lru[tier].state.push(Reverse((last_used, id)));
            }
        }
    }

    pub(super) fn lru_note_split(&mut self, lower: u32) {
        self.lru_push(lower);
    }

    pub(super) fn lru_shrink(&mut self, limit_bytes: u64, now_s: f64) {
        match self.config.state_share {
            None => {
                while self.kv_bytes[HBM] + self.state_bytes[HBM] > limit_bytes {
                    if !self.evict_hbm_kv(now_s) && !self.evict_hbm_state(now_s) {
                        break;
                    }
                }
            }
            Some(share) => {
                let total = self.sizes.capacity_units * self.sizes.bytes_per_unit;
                let state_limit = (share * total as f64) as u64;
                let kv_limit = limit_bytes.saturating_sub(state_limit);
                while self.kv_bytes[HBM] > kv_limit && self.evict_hbm_kv(now_s) {}
                while self.state_bytes[HBM] > state_limit && self.evict_hbm_state(now_s) {}
            }
        }
    }

    /// Make host room for node `id`'s HBM data.
    pub(super) fn lru_host_room(&mut self, id: u32) -> bool {
        let node = &self.nodes[id as usize];
        let kv_needed = if node.kv == 1 << HBM {
            u64::from(node.tokens) * self.sizes.bytes_per_token
        } else {
            0
        };
        let state_needed = if node.state == 1 << HBM {
            self.sizes.bytes_per_state
        } else {
            0
        };
        self.reclaim_duplicates(kv_needed + state_needed);
        match self.config.state_share {
            None => {
                while !self.host_fits(kv_needed + state_needed) {
                    if !self.evict_host_kv() && !self.evict_host_state() {
                        return false;
                    }
                }
                true
            }
            Some(share) => {
                let total = self.config.host_capacity_bytes;
                let state_limit = (share * total as f64) as u64;
                let kv_limit = total - state_limit;
                while self.kv_bytes[HOST] + kv_needed > kv_limit {
                    if !self.evict_host_kv() {
                        return false;
                    }
                }
                while self.state_bytes[HOST] + state_needed > state_limit {
                    if !self.evict_host_state() {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn pop_kv_victim(&mut self, tier: usize) -> Option<u32> {
        while let Some(Reverse((last_used, id))) = self.lru[tier].kv.pop() {
            let node = &self.nodes[id as usize];
            if node.last_used == last_used
                && node.kv & (1 << tier) != 0
                && node.locks == 0
                && node.kv_children[tier] == 0
            {
                return Some(id);
            }
        }
        None
    }

    fn pop_state_victim(&mut self, tier: usize) -> Option<u32> {
        while let Some(Reverse((last_used, id))) = self.lru[tier].state.pop() {
            let node = &self.nodes[id as usize];
            if node.last_used == last_used && node.state & (1 << tier) != 0 && node.locks == 0 {
                return Some(id);
            }
        }
        None
    }

    fn demote_if_write_back(&mut self, id: u32, now_s: f64) {
        if !self.config.write_through && self.lru_host_room(id) {
            self.copy_to_host(id, now_s);
        }
    }

    fn evict_hbm_kv(&mut self, now_s: f64) -> bool {
        let Some(id) = self.pop_kv_victim(HBM) else {
            return false;
        };
        let used_before = self.used_tokens;
        self.demote_if_write_back(id, now_s);
        self.set_kv(id, HBM, false);
        self.set_state(id, HBM, false);
        self.record_eviction(self.nodes[id as usize].session, used_before);
        true
    }

    fn evict_hbm_state(&mut self, now_s: f64) -> bool {
        let Some(id) = self.pop_state_victim(HBM) else {
            return false;
        };
        if !self.config.write_through && self.nodes[id as usize].state == 1 << HBM {
            let bytes = self.sizes.bytes_per_state;
            self.reclaim_duplicates(bytes);
            let mut room = self.host_fits(bytes);
            while !room && (self.evict_host_kv() || self.evict_host_state()) {
                room = self.host_fits(bytes);
            }
            if room {
                let ready = self.queue_demotion(bytes, now_s);
                self.nodes[id as usize].host_ready_s = ready;
                self.set_state(id, HOST, true);
            }
        }
        self.set_state(id, HBM, false);
        true
    }

    fn evict_host_kv(&mut self) -> bool {
        let Some(id) = self.pop_kv_victim(HOST) else {
            return false;
        };
        self.set_kv(id, HOST, false);
        self.set_state(id, HOST, false);
        true
    }

    fn evict_host_state(&mut self) -> bool {
        let Some(id) = self.pop_state_victim(HOST) else {
            return false;
        };
        self.set_state(id, HOST, false);
        true
    }
}
