use super::*;
use crate::common::Time;

const TOKEN_BYTES: u64 = 10;
const STATE_BYTES: u64 = 640;

fn sizes(capacity_bytes: u64, checkpoint_interval: Option<u32>) -> ContextCacheSizes {
    ContextCacheSizes {
        bytes_per_token: TOKEN_BYTES,
        bytes_per_state: STATE_BYTES,
        bytes_per_unit: 1,
        capacity_units: capacity_bytes,
        checkpoint_interval,
    }
}

fn config(policy: ContextCachePolicy, host_capacity_bytes: u64) -> ContextCacheConfig {
    ContextCacheConfig {
        policy,
        host_capacity_bytes,
        link_bytes_per_s: 1e6,
        recompute_alpha_s: 1e-3,
        recompute_beta_s: 1e-7,
        write_through: false,
        state_share: None,
        page_tokens: 64,
        bytes_per_token: None,
    }
}

fn session(session_id: u32, declared: u32, round: u32, prefix_round: Option<u32>) -> SessionInput {
    SessionInput::Session {
        session_id,
        session_start_time: Time::ZERO,
        declared_prefix_tokens: declared,
        context: ContextRef {
            round,
            prefix_round,
        },
    }
}

struct Driver {
    cache: ContextCache,
    next_request: u32,
    limit: u64,
}

impl Driver {
    fn new(cache: ContextCache, limit: u64) -> Self {
        Self {
            cache,
            next_request: 0,
            limit,
        }
    }

    /// Run one call to completion at `now`; returns the hit.
    fn call(&mut self, input: SessionInput, prompt: u32, now: f64) -> u32 {
        let request = RequestId(self.next_request);
        self.next_request += 1;
        let hit = self.cache.begin(request, input, now).hit_tokens;
        self.cache.finish(request, prompt, now);
        self.cache.shrink_to(self.limit, now);
        assert!(self.cache.bytes_consistent());
        hit
    }
}

fn lru(capacity: u64, host: u64) -> Driver {
    Driver::new(
        ContextCache::new(config(ContextCachePolicy::Lru, host), sizes(capacity, None)),
        capacity,
    )
}

#[test]
fn an_append_resumes_at_the_previous_prompt_end() {
    let mut driver = lru(1 << 30, 0);
    assert_eq!(driver.call(session(1, 0, 0, None), 640, 0.0), 0);
    assert_eq!(driver.call(session(1, 640, 1, None), 1280, 1.0), 640);
    assert_eq!(driver.call(session(1, 1280, 2, None), 1920, 2.0), 1280);
}

#[test]
fn a_hit_lands_only_on_a_state_at_or_below_the_declared_prefix() {
    let mut driver = lru(1 << 30, 0);
    driver.call(session(1, 0, 0, None), 640, 0.0);
    driver.call(session(1, 640, 1, None), 1280, 1.0);
    // Rewind into the second call's tail: only the first prompt end has a state.
    assert_eq!(driver.call(session(1, 960, 2, None), 1600, 2.0), 640);
    // The new call placed a state at the branch point, for the next rewind.
    assert_eq!(driver.call(session(1, 960, 3, Some(1)), 1000, 3.0), 960);
}

#[test]
fn an_older_round_is_reachable_through_its_own_branch() {
    let mut driver = lru(1 << 30, 0);
    driver.call(session(1, 0, 0, None), 640, 0.0);
    driver.call(session(1, 640, 1, None), 1280, 1.0);
    driver.call(session(1, 640, 2, Some(0)), 960, 2.0);
    assert_eq!(driver.call(session(1, 1280, 3, Some(1)), 1500, 3.0), 1280);
}

#[test]
fn checkpoints_give_a_deep_rewind_a_nearby_state() {
    let mut driver = Driver::new(
        ContextCache::new(
            config(ContextCachePolicy::Lru, 0),
            sizes(1 << 30, Some(320)),
        ),
        1 << 30,
    );
    driver.call(session(1, 0, 0, None), 1280, 0.0);
    assert_eq!(driver.call(session(1, 700, 1, None), 900, 1.0), 640);
}

#[test]
fn hbm_lru_demotes_to_the_host_and_a_hit_then_pays_a_load() {
    let context = 640 * TOKEN_BYTES + STATE_BYTES;
    let mut driver = lru(context, 10 * context);
    driver.call(session(1, 0, 0, None), 640, 0.0);
    driver.call(session(2, 0, 0, None), 640, 1.0);
    assert_eq!(driver.cache.kv_bytes[HOST], 640 * TOKEN_BYTES);
    let request = RequestId(99);
    let result = driver.cache.begin(request, session(1, 640, 1, None), 2.0);
    assert_eq!(result.hit_tokens, 640);
    let load_end = driver.cache.load_end_s(request).unwrap();
    assert!(load_end > 2.0 + (context as f64 / 1e6) * 0.99, "{load_end}");
}

#[test]
fn with_no_host_an_evicted_session_misses() {
    let context = 640 * TOKEN_BYTES + STATE_BYTES;
    let mut driver = lru(context, 0);
    driver.call(session(1, 0, 0, None), 640, 0.0);
    driver.call(session(2, 0, 0, None), 640, 1.0);
    assert_eq!(driver.call(session(1, 640, 1, None), 700, 2.0), 0);
}

#[test]
fn running_requests_own_their_path() {
    let mut driver = lru(1 << 30, 0);
    driver.call(session(1, 0, 0, None), 640, 0.0);
    let used = driver.cache.used_units();
    driver
        .cache
        .begin(RequestId(50), session(1, 640, 1, None), 1.0);
    assert_eq!(driver.cache.used_units(), 0);
    assert!(driver.cache.shrink_to(0, 1.0).is_empty());
    driver.cache.finish(RequestId(50), 640, 1.0);
    assert_eq!(driver.cache.used_units(), used);
}

#[test]
fn a_static_split_limits_each_kind_to_its_pool() {
    let mut config = config(ContextCachePolicy::Lru, 0);
    config.state_share = Some(0.5);
    let capacity = 4 * STATE_BYTES;
    let mut driver = Driver::new(ContextCache::new(config, sizes(capacity, None)), capacity);
    for session_id in 0..20 {
        driver.call(session(session_id, 0, 0, None), 64, f64::from(session_id));
    }
    assert!(driver.cache.state_bytes[HBM] <= 2 * STATE_BYTES);
    assert!(driver.cache.kv_bytes[HBM] <= 2 * STATE_BYTES);
}

#[test]
fn price_graph_keeps_the_tier_within_its_limit_and_serves_returning_sessions() {
    let context = 6400 * TOKEN_BYTES + STATE_BYTES;
    let limit = 8 * context;
    let mut driver = Driver::new(
        ContextCache::new(
            config(ContextCachePolicy::PriceGraph, 40 * context),
            sizes(limit, Some(3200)),
        ),
        limit,
    );
    let mut now = 0.0;
    let (mut declared_total, mut hit_total) = (0u64, 0u64);
    for round in 0..40u32 {
        for session_id in 0..12u32 {
            now += 0.5;
            let declared = 640 * round;
            let hit = driver.call(
                session(session_id, declared, round, None),
                declared + 640,
                now,
            );
            if round >= 20 {
                declared_total += u64::from(declared);
                hit_total += u64::from(hit);
            }
            assert!(driver.cache.used_units() <= limit);
        }
    }
    assert!(
        hit_total * 10 >= declared_total * 8,
        "{hit_total} of {declared_total}"
    );
    assert!(driver.cache.kv_bytes[HOST] + driver.cache.state_bytes[HOST] <= 40 * context);
}

/// Six one-shot sessions per returning call flush any recency order from a cache
/// of four contexts. The return rates tell them apart.
fn hits_with_one_shot_traffic(policy: ContextCachePolicy) -> u32 {
    let context = 640 * TOKEN_BYTES + STATE_BYTES;
    let limit = 4 * context;
    let mut driver = Driver::new(
        ContextCache::new(config(policy, 0), sizes(limit, None)),
        limit,
    );
    let mut now = 0.0;
    let mut late_hits = 0;
    for round in 0..300u32 {
        for one_shot in 0..6 {
            now += 1.0;
            driver.call(session(1000 + round * 6 + one_shot, 0, 0, None), 640, now);
        }
        now += 1.0;
        let hit = driver.call(session(0, 640, round, None), 640, now);
        if round >= 200 && hit == 640 {
            late_hits += 1;
        }
    }
    late_hits
}

#[test]
fn price_graph_keeps_what_returns_where_recency_cannot() {
    assert_eq!(hits_with_one_shot_traffic(ContextCachePolicy::Lru), 0);
    assert!(hits_with_one_shot_traffic(ContextCachePolicy::PriceGraph) > 90);
}
