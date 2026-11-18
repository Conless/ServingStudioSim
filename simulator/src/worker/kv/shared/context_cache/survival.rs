//! Online return-time estimator for idle contexts.
//!
//! A *spell* is the time a demand point waits, from the moment its request
//! finished, for the next request to land on it. Spells are grouped by class
//! into octave age bins. Each class's hazard is shrunk toward the pooled hazard,
//! and [`Survival::return_density`] reads the resulting life table in closed
//! form: the probability of returning within a horizon, per second survived.
//!
//! Only observed arrivals feed the table. It never reads the future or the kind
//! of the response that ended a call.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

const BINS: usize = 18;

/// Pseudo-count of the shrinkage toward the pooled hazard.
const SHRINKAGE: f64 = 1.0;

/// `f64` with the total order the event heap needs. Spell times are never NaN.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct Ordered(pub(super) f64);
impl Eq for Ordered {}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn bin_upper(bin: usize) -> f64 {
    if bin + 1 >= BINS {
        f64::INFINITY
    } else {
        (1u64 << bin) as f64
    }
}

fn bin_lower(bin: usize) -> f64 {
    if bin == 0 {
        0.0
    } else {
        bin_upper(bin - 1)
    }
}

/// The octave bin of an age in seconds: `[0, 1)`, `[1, 2)`, `[2, 4)`, ...
fn age_bin(age: f64) -> usize {
    if age < 1.0 {
        return 0;
    }
    let bits = (age as u64).max(1);
    ((u64::BITS - bits.leading_zeros()) as usize).min(BINS - 1)
}

#[derive(Clone, Copy)]
struct Spell {
    class: usize,
    last: f64,
    bin: usize,
    entered: f64,
    live: bool,
}

#[derive(Clone, Default)]
struct LifeTable {
    log_survival: Vec<f64>,
    integral: Vec<f64>,
    hazard: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
struct ClassBins {
    events: [i64; BINS],
    closed_exposure: [f64; BINS],
    open_spells: [i64; BINS],
    open_entered: [f64; BINS],
}

pub(super) struct Survival {
    classes: Vec<ClassBins>,
    spells: Vec<Spell>,
    crossings: BinaryHeap<Reverse<(Ordered, u64, u32)>>,
    sequence: u64,
    events: i64,
    oldest_return: f64,
    return_age_limit: f64,
    pooled: Option<LifeTable>,
    tables: Vec<LifeTable>,
    closed_total: f64,
    open_count: i64,
    open_sum: f64,
    events_at_build: i64,
    exposure_at_build: f64,
    hazard_at_build: f64,
}

impl Survival {
    pub(super) fn new() -> Self {
        Self {
            classes: Vec::new(),
            spells: Vec::new(),
            crossings: BinaryHeap::new(),
            sequence: 0,
            events: 0,
            oldest_return: f64::INFINITY,
            return_age_limit: f64::INFINITY,
            pooled: None,
            tables: Vec::new(),
            closed_total: 0.0,
            open_count: 0,
            open_sum: 0.0,
            events_at_build: 0,
            exposure_at_build: 0.0,
            hazard_at_build: 0.0,
        }
    }

    pub(super) fn open(&mut self, class: usize, last: f64, now: f64) -> u32 {
        if self.classes.len() <= class {
            self.classes.resize(class + 1, ClassBins::default());
        }
        let bin = age_bin(now - last);
        let id = self.spells.len() as u32;
        self.spells.push(Spell {
            class,
            last,
            bin,
            entered: now,
            live: true,
        });
        self.classes[class].open_spells[bin] += 1;
        self.classes[class].open_entered[bin] += now;
        self.open_count += 1;
        self.open_sum += now;
        self.push_crossing(id);
        id
    }

    /// End a spell. `returned` says whether a request landed on the point, as
    /// opposed to the spell being superseded.
    pub(super) fn close(&mut self, id: u32, now: f64, returned: bool) {
        self.advance(now);
        let spell = self.spells[id as usize];
        let class = &mut self.classes[spell.class];
        class.closed_exposure[spell.bin] += now - spell.entered;
        class.open_spells[spell.bin] -= 1;
        class.open_entered[spell.bin] -= spell.entered;
        self.closed_total += now - spell.entered;
        self.open_count -= 1;
        self.open_sum -= spell.entered;
        self.spells[id as usize].live = false;
        if returned {
            let age = now - spell.last;
            self.classes[spell.class].events[age_bin(age)] += 1;
            self.events += 1;
            if self.events == 1 || age > self.oldest_return {
                self.oldest_return = age;
            }
        }
    }

    /// True when enough evidence arrived since the last build: the event count
    /// moved by its own Poisson noise, or the exposure predicts that it should have.
    pub(super) fn wants_rebuild(&self, now: f64) -> bool {
        let threshold = (self.events_at_build as f64).sqrt();
        let new_events = self.events - self.events_at_build;
        (new_events > 0 && new_events as f64 >= threshold)
            || (self.hazard_at_build > 0.0
                && self.hazard_at_build * (self.total_exposure(now) - self.exposure_at_build)
                    >= threshold)
    }

    pub(super) fn rebuild(&mut self, now: f64) {
        self.advance(now);
        self.events_at_build = self.events;
        self.exposure_at_build = self.total_exposure(now);
        self.hazard_at_build = if self.exposure_at_build > 0.0 {
            self.events as f64 / self.exposure_at_build
        } else {
            0.0
        };
        if self.events == 0 {
            return;
        }
        let exposure: Vec<[f64; BINS]> = self
            .classes
            .iter()
            .map(|class| {
                let mut bins = [0.0; BINS];
                for (bin, value) in bins.iter_mut().enumerate() {
                    *value = class.closed_exposure[bin] + class.open_spells[bin] as f64 * now
                        - class.open_entered[bin];
                }
                bins
            })
            .collect();
        let mut pooled = [0.0; BINS];
        let mut pooled_exposure = [0.0; BINS];
        let mut last = 0.0;
        for bin in 0..BINS {
            let events: i64 = self.classes.iter().map(|class| class.events[bin]).sum();
            let time: f64 = exposure.iter().map(|class| class[bin]).sum();
            pooled_exposure[bin] = time;
            if events > 0 && time > 0.0 {
                last = events as f64 / time;
            }
            pooled[bin] = last;
        }
        // Past the oldest return, zero hazard, unless the exposure beyond it is
        // still too small to say that nothing returns there.
        let oldest_bin = age_bin(self.oldest_return);
        let exposure_beyond: f64 = pooled_exposure[oldest_bin + 1..].iter().sum();
        self.return_age_limit = if pooled[oldest_bin] * exposure_beyond >= 1.0 {
            self.oldest_return
        } else {
            f64::INFINITY
        };
        self.pooled = Some(self.life_table(&pooled));
        self.tables = (0..self.classes.len())
            .map(|class| {
                let hazard: Vec<f64> = (0..BINS)
                    .map(|bin| {
                        if pooled[bin] > 0.0 {
                            (self.classes[class].events[bin] as f64 + SHRINKAGE)
                                / (exposure[class][bin] + SHRINKAGE / pooled[bin])
                        } else {
                            0.0
                        }
                    })
                    .collect();
                self.life_table(&hazard)
            })
            .collect();
    }

    /// Return density of `class` at `age` over `horizon`: the chance of a return
    /// in `[age, age + horizon]` per unit of expected survival there.
    pub(super) fn return_density(&self, class: usize, age: f64, horizon: f64) -> f64 {
        let unknown = if horizon > 0.0 {
            1.0 / horizon
        } else {
            f64::INFINITY
        };
        if self.events == 0 {
            return unknown;
        }
        if age >= self.return_age_limit {
            return 0.0;
        }
        let Some(table) = self.tables.get(class).or(self.pooled.as_ref()) else {
            return unknown;
        };
        let age = age.max(0.0);
        let bin = age_bin(age);
        if (horizon <= 0.0 || age_bin(age + horizon) == bin)
            && age + horizon <= self.return_age_limit
        {
            return table.hazard[bin];
        }
        let (survival_0, integral_0) = self.survival_at(table, age);
        let (survival_1, integral_1) = self.survival_at(table, age + horizon);
        let spent = integral_1 - integral_0;
        if spent > 0.0 {
            (survival_0 - survival_1) / spent
        } else {
            0.0
        }
    }

    fn push_crossing(&mut self, id: u32) {
        let spell = self.spells[id as usize];
        if spell.bin < BINS - 1 {
            self.sequence += 1;
            self.crossings.push(Reverse((
                Ordered(spell.last + bin_upper(spell.bin)),
                self.sequence,
                id,
            )));
        }
    }

    /// Move every spell that aged past its bin into the next one.
    fn advance(&mut self, now: f64) {
        while let Some(&Reverse((Ordered(crossing), _, id))) = self.crossings.peek() {
            if crossing > now {
                break;
            }
            self.crossings.pop();
            let spell = self.spells[id as usize];
            if !spell.live {
                continue;
            }
            let class = &mut self.classes[spell.class];
            class.closed_exposure[spell.bin] += crossing - spell.entered;
            class.open_spells[spell.bin] -= 1;
            class.open_entered[spell.bin] -= spell.entered;
            class.open_spells[spell.bin + 1] += 1;
            class.open_entered[spell.bin + 1] += crossing;
            self.closed_total += crossing - spell.entered;
            self.open_sum += crossing - spell.entered;
            let moved = &mut self.spells[id as usize];
            moved.bin += 1;
            moved.entered = crossing;
            self.push_crossing(id);
        }
    }

    fn total_exposure(&self, now: f64) -> f64 {
        self.closed_total + self.open_count as f64 * now - self.open_sum
    }

    fn life_table(&self, hazard: &[f64]) -> LifeTable {
        let limit = self.return_age_limit;
        let mut log_survival = vec![0.0f64];
        let mut integral = vec![0.0];
        for bin in 0..BINS - 1 {
            let (lower, upper) = (bin_lower(bin), bin_upper(bin));
            let rate = if lower < limit { hazard[bin] } else { 0.0 };
            let (log_0, integral_0) = (log_survival[bin], integral[bin]);
            let survival_0 = log_0.exp();
            if lower < limit && limit < upper {
                let before = limit - lower;
                let log_1 = log_0 - rate * before;
                integral.push(
                    integral_0
                        + survival_0 * spent_in(rate, before)
                        + log_1.exp() * (upper - limit),
                );
                log_survival.push(log_1);
            } else {
                let width = upper - lower;
                integral.push(integral_0 + survival_0 * spent_in(rate, width));
                log_survival.push(log_0 - rate * width);
            }
        }
        LifeTable {
            log_survival,
            integral,
            hazard: hazard.to_vec(),
        }
    }

    /// `(S(x), integral of S over [0, x])`.
    fn survival_at(&self, table: &LifeTable, x: f64) -> (f64, f64) {
        let bin = age_bin(x);
        let lower = bin_lower(bin);
        let limit = self.return_age_limit;
        let rate = if lower < limit {
            table.hazard[bin]
        } else {
            0.0
        };
        let survival_0 = table.log_survival[bin].exp();
        if lower < limit && limit < x {
            let before = limit - lower;
            let survival_1 = (table.log_survival[bin] - rate * before).exp();
            return (
                survival_1,
                table.integral[bin]
                    + survival_0 * spent_in(rate, before)
                    + survival_1 * (x - limit),
            );
        }
        let width = x - lower;
        (
            survival_0 * (-rate * width).exp(),
            table.integral[bin] + survival_0 * spent_in(rate, width),
        )
    }
}

/// `integral of exp(-rate t) over [0, width]`.
fn spent_in(rate: f64, width: f64) -> f64 {
    if rate > 0.0 {
        -(-rate * width).exp_m1() / rate
    } else {
        width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn age_bins_are_octaves() {
        assert_eq!(age_bin(0.5), 0);
        assert_eq!(age_bin(1.0), 1);
        assert_eq!(age_bin(1.9), 1);
        assert_eq!(age_bin(2.0), 2);
        assert_eq!(age_bin(1e12), BINS - 1);
    }

    #[test]
    fn no_evidence_defaults_to_one_return_per_horizon() {
        assert_eq!(Survival::new().return_density(0, 5.0, 10.0), 0.1);
    }

    #[test]
    fn a_context_that_keeps_returning_quickly_outranks_a_stale_one() {
        let mut survival = Survival::new();
        for start in 0..200 {
            let start = f64::from(start) * 100.0;
            let spell = survival.open(0, start, start);
            survival.close(spell, start + 1.5, true);
        }
        // Spells that never return add exposure at old ages.
        let now = 20_000.0;
        for start in 0..50 {
            survival.open(0, now - 4000.0 - f64::from(start), now);
        }
        survival.rebuild(now + 4000.0);
        let young = survival.return_density(0, 1.2, 1.0);
        let old = survival.return_density(0, 4000.0, 1.0);
        assert!(young > 100.0 * old, "{young} vs {old}");
    }

    #[test]
    fn a_closed_spell_stops_accumulating_exposure() {
        let mut survival = Survival::new();
        let spell = survival.open(0, 0.0, 0.0);
        survival.close(spell, 10.0, false);
        assert_eq!(survival.total_exposure(1000.0), 10.0);
    }
}
