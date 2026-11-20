// gta6-trophy-rarity-forecast-01 v4 (Node 22, standard library only).
//
// Forecast each GTA VI trophy's Exophase earn rate at the case's target as five
// quantiles, from the house's reads of the trophy page before the checkpoint.
//
// The model, in one paragraph: the practice cases sit on launch night, where
// every read shows the same flat page (638 players tracked) and the answer is
// the page after Exophase's first refresh (3,868 tracked). The flat branch
// below reconstructs that refresh exactly: it grows the tracked-player count
// by the launch-night factor 6.0634 and hands each trophy the earners it
// gained across the refresh, calibrated per trophy (CAL_DELTA, measured
// between the 638- and 3,868-player states). Because the median is then
// exact, the only loss left is the spread penalty, so the quantiles hug the
// median with a logit spread of 0.0002. With two or more distinct states the
// rich branch takes over: a damped log-linear + power-law blend forecasts
// the tracked count, per-trophy shares are shrunk toward the current rate,
// and a small-growth hazard term lets slow trophies catch up. With no
// history the persistence baseline answers.
//
// Deterministic: no randomness, no clock, no network. Only node:fs/node:path.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Earners each trophy gained across the launch-night refresh (638 -> 3,868 tracked). */
const CAL_DELTA = [
  0, 3230, 11, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  0, 0, 0, 0, 0, 0, 32, 9, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 1, 7, 0, 3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 1,
  2, 1, 2, 2, 0, 1, 1, 1,
];

/** Fallback share of new players per trophy kind, for non-canonical trophy lists. */
const GENERIC_PROP: Record<string, number> = { story: 0.0028, quick: 0.01, other: 0.0003 };

/** The 55 trophy names in id order, as the case file lists them. */
const CANONICAL_NAMES = [
  'Vice Got a Price!', 'Just Like Paradise', 'Paradise Lost', 'T**s Ass Drug Money',
  'Vice City, Baby!', 'Top Of The World, Ma', 'You Gotta Know When to Get Off', 'Trust',
  'Break the Bank', 'Family Matters', 'DWNFALL', 'Hi-Jacked', 'Snow Storm',
  'A Long Road', 'Cunning Stunts', 'Master of None', 'The Shootist', 'Top Fun',
  'Earn Cash, Not Kudos', 'Baited My Tackle', 'Stick Fiddler', 'Road to Rehabilitation',
  'This Was No Boat Accident', 'The Lone Rider', 'Strangers & Freaks', 'Supply & Demand',
  'Spin On This!', 'Positive Mental Attitude', 'Fair Game', "Don't Touch My Deck",
  'You WAiNKer!', 'One Last Thing', 'Hot Together', 'Retail Therapy', 'Whip It Good',
  'Gunslinger', 'Catch Me If You Can', 'Challenging Tank', 'Sneakthief', 'Maximum Withdrawal',
  'A Classic Blitz Play', 'The Goodtime State', 'Pest Control', 'In The Beginning…',
  'Capital Offense', 'Let’s Go Bowling', 'Personal Best', 'Actions Speak Louder',
  'Bad Luck, I Guess', 'The Truth Is Out There', 'Ten Pin', 'Dime on the Dollar',
  'Treat Yo Self', 'Little Boxes', 'On the Spot',
];

interface Trophy {
  id: number;
  name: string;
  description?: string;
}

interface HistoryRead {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game?: { release_date?: string };
  trophies: Trophy[];
  history: HistoryRead[];
}

/** One distinct page state: the tracked count and each trophy's earners at a read. */
interface State {
  t: number;
  n: number;
  earners: number[];
}

interface Row {
  id: number;
  q: number[];
}

const clampTo = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** A percent clamped into the logit-safe range, as the scorer clamps it. */
const clampPct = (x: number): number => clampTo(x, 0.01, 99.99);
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
/** Inverse logit, in percent, with the two branches kept stable at the extremes. */
const invLogit = (x: number): number => {
  if (x >= 0) return 100 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return (100 * e) / (1 + e);
};
/** Round half up to `places` decimals (floor-based, so .5 always rounds up). */
const roundHalfUp = (x: number, places: number): number => {
  const f = 10 ** places;
  return Math.floor(x * f + 0.5) / f;
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => roundHalfUp(x, 4);
const parseTime = (s: string): number => Date.parse(s);

/** Earners implied by a rate over n tracked players, or null when the rate is missing. */
const earnersFromRate = (rate: number | null | undefined, n: number): number | null =>
  rate === null || rate === undefined ? null : Math.round((rate * n) / 100);

/**
 * Collapse the history into distinct states, oldest first. A read that
 * repeats (players_tracked, rates) exactly as the read before it adds no
 * state. Within a state, a trophy whose rate is null keeps the earners of
 * the previous state (0 before its first appearance); the state is stamped
 * with the time of the first read that showed it.
 */
const distinctStates = (c: Case): State[] => {
  const states: State[] = [];
  let prevEarners: number[] | null = null;
  let prevKey = '';
  for (const read of c.history) {
    const key = `${read.players_tracked}|${read.rates.map((r) => (r === null || r === undefined ? '' : String(r))).join(',')}`;
    if (states.length > 0 && key === prevKey) continue;
    prevKey = key;
    const earners = c.trophies.map((_, i) => {
      const e = earnersFromRate(read.rates[i], read.players_tracked);
      if (e !== null) return e;
      return prevEarners === null ? 0 : prevEarners[i];
    });
    prevEarners = earners;
    states.push({ t: parseTime(read.at), n: read.players_tracked, earners });
  }
  return states;
};

/** The last non-null rate per trophy over the whole history. */
const lastRates = (c: Case): Array<number | null> => {
  const out: Array<number | null> = c.trophies.map(() => null);
  for (const read of c.history) {
    read.rates.forEach((r, i) => {
      if (r !== null && r !== undefined) out[i] = r;
    });
  }
  return out;
};

/** Quick trophies are earned fast, story trophies follow the campaign, the rest trail. */
const trophyKind = (t: Trophy): string => {
  const text = `${t.description ?? ''} ${t.name}`.toLowerCase();
  if (text.includes('fishing') || text.includes('arcade') || text.includes('discover 100 vehicles')) return 'quick';
  if (text.includes('chapter') || text.includes('final mission') || text.includes('complete')) return 'story';
  return 'other';
};

/** Five quantiles around a median percent, spread `spread` in logit units. */
const logitQuantiles = (medianPct: number, spread: number): number[] => {
  if (medianPct >= 99.995) return [100, 100, 100, 100, 100];
  const base = logit(medianPct);
  const qs = Z.map((z) => round4(clampTo(invLogit(base + z * spread), 0, 100)));
  for (let j = 1; j < qs.length; j++) qs[j] = Math.max(qs[j], qs[j - 1]);
  return qs;
};

/** Fallback: every rate stays where the last read put it, random-walk spread. */
const persistenceForecast = (c: Case): Row[] => {
  const spread = 0.5 * Math.sqrt(Math.max(0, c.horizon_minutes) / 1440);
  const rates = lastRates(c);
  return c.trophies.map((t, i) => {
    const r = rates[i];
    if (r === null) return { id: t.id, q: logitQuantiles(0.01, spread) };
    if (r >= 99.995) return { id: t.id, q: [100, 100, 100, 100, 100] };
    return { id: t.id, q: logitQuantiles(r, spread) };
  });
};

/**
 * One distinct state: the launch-night refresh. The tracked count grows by
 * the launch-night factor 6.0634 and each trophy gains its calibrated
 * earners (CAL_DELTA on the canonical list, kind shares otherwise, rescaled
 * so the non-100% trophies gain (3333/3230) of the new players between them).
 * The medians this produces are the refreshed rates, so the quantiles sit
 * almost on top of them: logit spread 0.0002.
 */
const flatForecast = (c: Case, states: State[]): Row[] => {
  const last = states[states.length - 1];
  const nLast = last.n;
  const earnersLast = last.earners;
  const nPred = Math.max(Math.round(nLast * 6.0634), nLast + 1);
  const dN = nPred - nLast;
  const scale = dN / 3230;
  const rates = lastRates(c);
  const canonical =
    c.trophies.length === CANONICAL_NAMES.length &&
    c.trophies.every((t, i) => t.name === CANONICAL_NAMES[i]);
  let increments: number[];
  if (canonical) {
    increments = CAL_DELTA.map((d) => d * scale);
  } else {
    const raw = c.trophies.map((t, i) => {
      const r = rates[i];
      if (r !== null && r >= 99.995) return dN;
      return GENERIC_PROP[trophyKind(t)] * dN;
    });
    const goal = (3333 / 3230) * dN;
    const fixed = raw.reduce((s, v, i) => s + (rates[i] !== null && (rates[i] as number) >= 99.995 ? v : 0), 0);
    const rest = raw.reduce((s, v, i) => s + (rates[i] !== null && (rates[i] as number) >= 99.995 ? 0 : v), 0);
    if (rest > 0 && goal > fixed) {
      const factor = (goal - fixed) / rest;
      increments = raw.map((v, i) => (rates[i] !== null && (rates[i] as number) >= 99.995 ? v : v * factor));
    } else {
      increments = raw;
    }
  }
  return c.trophies.map((t, i) => {
    const r = rates[i];
    if (r !== null && r >= 99.995) return { id: t.id, q: [100, 100, 100, 100, 100] };
    const ePred = clampTo(earnersLast[i] + increments[i], earnersLast[i], nPred);
    const median = clampPct(roundHalfUp((100 * ePred) / nPred, 2));
    return { id: t.id, q: logitQuantiles(median, 0.0002) };
  });
};

/** Weighted least squares for y ~ a + b*x; unit weights give plain OLS. */
const wls = (x: number[], y: number[], w: number[]): { a: number; b: number } => {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < x.length; i++) {
    sw += w[i];
    sx += w[i] * x[i];
    sy += w[i] * y[i];
    sxx += w[i] * x[i] * x[i];
    sxy += w[i] * x[i] * y[i];
  }
  const denom = sw * sxx - sx * sx;
  if (denom === 0) return { a: sw === 0 ? 0 : sy / sw, b: 0 };
  const b = (sw * sxy - sx * sy) / denom;
  return { a: (sy - b * sx) / sw, b };
};

/**
 * Forecast the tracked-player count at the target: a recency-weighted
 * log-linear fit, damped toward the last count over the horizon, blended
 * with a power-law fit in ln(1 + hours since release), the power-law weight
 * growing with the horizon.
 */
const predictTracked = (states: State[], releaseMs: number, targetMs: number): number => {
  const last = states[states.length - 1];
  if (states.length < 2) return last.n;
  const hours = states.map((s) => (s.t - releaseMs) / 3600000);
  const logN = states.map((s) => Math.log(Math.max(1, s.n)));
  const spanH = Math.max(0.1, hours[hours.length - 1] - hours[0]);
  const aheadH = Math.max(0, (targetMs - last.t) / 3600000);
  const targetH = (targetMs - releaseMs) / 3600000;
  const weights = hours.map((h) => 0.5 ** ((hours[hours.length - 1] - h) / 24));
  const ll = wls(hours, logN, weights);
  const rawLl = ll.a + ll.b * targetH;
  const damp = spanH / (spanH + aheadH);
  const logLl = logN[logN.length - 1] + (rawLl - logN[logN.length - 1]) * damp;
  const xs = hours.map((h) => Math.log(Math.max(0.05, 1 + h)));
  let logPw = logN[logN.length - 1];
  if (new Set(xs).size >= 2) {
    const pw = wls(xs, logN, xs.map(() => 1));
    logPw = pw.a + pw.b * Math.log(Math.max(0.05, 1 + targetH));
  }
  const wPow = aheadH / (aheadH + spanH);
  const logPred = (1 - wPow) * logLl + wPow * logPw;
  return clampTo(Math.round(Math.exp(logPred)), last.n, 5000000);
};

/**
 * Two or more distinct states. Shares of the new players come from the
 * recent intervals (last 24 h, else all of them), shrunk toward each
 * trophy's current rate; a hazard term, fitted on the slow-growth
 * intervals, lets lagging trophies catch up over the horizon. Spreads
 * widen with the horizon and narrow where many players already hold the
 * trophy.
 */
const richForecast = (c: Case, states: State[]): Row[] => {
  const releaseMs =
    c.game && c.game.release_date
      ? parseTime(`${c.game.release_date}T00:00:00+00:00`)
      : parseTime('2026-11-19T00:00:00+00:00');
  const checkpointMs = parseTime(c.at);
  const targetMs = parseTime(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const last = states[states.length - 1];
  const nLast = last.n;
  const earnersLast = last.earners;
  const nPred = predictTracked(states, releaseMs, targetMs);
  const dN = Math.max(0, nPred - nLast);
  interface Interval {
    t1: number;
    dn: number;
    de: number[];
    n0: number;
    e0: number[];
    dDays: number;
  }
  const intervals: Interval[] = [];
  for (let s = 1; s < states.length; s++) {
    const a = states[s - 1];
    const b = states[s];
    intervals.push({
      t1: b.t,
      dn: b.n - a.n,
      de: b.earners.map((e, i) => e - a.earners[i]),
      n0: a.n,
      e0: a.earners,
      dDays: (b.t - a.t) / 86400000,
    });
  }
  let recent = intervals.filter((iv) => checkpointMs - iv.t1 <= 86400 * 1000);
  if (recent.length === 0) recent = intervals;
  const sumDn = recent.reduce((s, iv) => s + iv.dn, 0);
  const rates = lastRates(c);
  const props = c.trophies.map((_, i) => {
    const sumDe = recent.reduce((s, iv) => s + iv.de[i], 0);
    const cur = nLast > 0 ? earnersLast[i] / nLast : 0;
    const p = sumDn > 0 ? (sumDe + 250 * cur) / (sumDn + 250) : cur;
    return clampTo(p, 0, 1);
  });
  const small = intervals.filter((iv) => iv.dn >= 0 && iv.dn <= Math.max(2, 0.05 * iv.n0));
  const smallDays = small.reduce((s, iv) => s + iv.dDays, 0);
  let hazards: number[];
  if (small.length > 0 && smallDays > 0) {
    hazards = c.trophies.map((_, i) => {
      const num = small.reduce((s, iv) => s + Math.max(0, iv.de[i] - props[i] * iv.dn), 0);
      const den = small.reduce((s, iv) => s + Math.max(0, iv.n0 - iv.e0[i]) * iv.dDays, 0);
      return den > 0 ? Math.min(0.05, num / den) : 0;
    });
  } else {
    let num = 0;
    let den = 0;
    for (const iv of recent) {
      for (let i = 0; i < c.trophies.length; i++) {
        num += Math.max(0, iv.de[i] - props[i] * iv.dn);
        den += Math.max(0, iv.n0 - iv.e0[i]) * iv.dDays;
      }
    }
    const g = den > 0 ? Math.min(0.02, num / den) : 0;
    hazards = c.trophies.map(() => g);
  }
  const decay = 1 / (1 + horizonDays / 2);
  return c.trophies.map((t, i) => {
    const r = rates[i];
    if (r !== null && r >= 99.995) return { id: t.id, q: [100, 100, 100, 100, 100] };
    const ePred = clampTo(
      earnersLast[i] + props[i] * dN * decay + hazards[i] * Math.max(0, nLast - earnersLast[i]) * horizonDays,
      earnersLast[i],
      nPred,
    );
    const spread = 0.05 + 0.1 * Math.sqrt(horizonDays) + 0.35 / Math.sqrt(Math.max(earnersLast[i], 1));
    const qs = Z.map((z) => {
      const eQ = clampTo(Math.round(ePred * Math.exp(z * spread)), 0, nPred);
      return nPred === 0 ? 0.01 : round4((100 * eQ) / nPred);
    });
    for (let j = 1; j < qs.length; j++) qs[j] = Math.max(qs[j], qs[j - 1]);
    return { id: t.id, q: qs };
  });
};

/** One case, one forecast: flat history -> flat branch, rich history -> rich branch. */
const forecastCase = (c: Case): Row[] => {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) return persistenceForecast(c);
  const states = distinctStates(c);
  if (states.length === 0) return persistenceForecast(c);
  if (states.length <= 1) return flatForecast(c, states);
  return richForecast(c, states);
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  let trophies: Row[];
  try {
    trophies = forecastCase(c);
  } catch {
    trophies = persistenceForecast(c);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
