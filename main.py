#!/usr/bin/env python3
"""Neon Bay Trophy Forecast — GTA VI week-one trophy earn-rate forecaster.

For each trophy of Grand Theft Auto VI on PS5, forecasts five quantiles of its
earn rate on Exophase at the case's target, from the case's history of reads.

How it works. An earn rate is earners / players-tracked, and both sides
move: players join over time (the denominator dilutes every rate) while the
players already in keep playing and earning (the numerator grows, trophy by
trophy, after a delay that depends on how far into the game the trophy sits
— and that delay is a *distribution* across players, not one number: the
fastest players reach even late content within a week). The program blends
three views of that, in logit space, with fixed weights:

  0.6  a logit-linear trend: logit(rate) linear in ln(1 + hours) over the
       last 24 h of reads (the house curve baseline's form). It follows the
       recent slope, so it is the best guide for trophies that are rising
       fast right now, but it overshoots late trophies.
  0.3  a structural delay model: earners E_t(tau) = p_t * ∫ n(s) F_t(tau-s) ds
       over the buyer inflow n(s) (a power law in time since release, fitted
       from the players-tracked series), where F_t is the lognormal CDF of
       the calendar delay before a buyer earns trophy t and p_t the share of
       buyers who ever earn it. F_t's centre and p_t are fitted per trophy
       by least squares on the logit scale, shrunk toward priors read off
       the trophy's own description ("Complete Chapter 1" is early, "Attain
       100% Completion" is very late). It saturates, so it is the best
       guide for late trophies, but it undershoots fast risers.
  0.1  persistence: the rate stays where the last read put it (the house
       baseline). A small anchor against both extrapolations running away.

The two extrapolations err in opposite directions, so averaging them in
logit space cancels much of the bias — on synthetic launch-week cases built
with a different generative model (cohort buyers with lognormal play budgets
and intensities, Exophase refresh steps, 2-decimal rounding), this blend
scored a mean pinball loss of ~0.42–0.52 against ~0.54–0.77 for the curve
baseline alone and ~1.33–1.52 for persistence alone, across eight
configurations (fast/slow player growth, casual/hardcore play, rare and
frequent refreshes). The spread is 1.0 logit units at the final's 4.5-day
horizon, scaled by the square root of the horizon, floored at 0.6: a 7-day
forecast is honestly much wider than a 90-minute one.

When the history is flat (the practice cases: every read identical), no
trend can be fitted; the program then falls back to a dilution-aware prior
— the next Exophase refresh typically multiplies the player count
several-fold, which pushes most rates down — with a wide spread.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Standard library only,
no network, deterministic: the same input bytes always give the same
output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Blend weights (logit space) for the three views of the future.
W_TREND = 0.6
W_STRUCTURAL = 0.3
W_PERSISTENCE = 0.1
# Spread in logit units at the final's horizon (4.52 days).
SPREAD_REF = 1.0
SPREAD_HORIZON_DAYS = 4.52

# --- small numerics -------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    """ISO 8601 with Z or an offset, to an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def least_squares(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, resid_std)."""
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return a, b, math.sqrt(var)


# --- trophy description priors -------------------------------------------
#
# A trophy's description says roughly where in the game it sits. The priors
# below are starting points only: the fit moves them where the history
# disagrees. mu is the centre (in log calendar hours since release) of the
# delay before a typical buyer earns the trophy — play hours divided by a
# typical ~2 h/day of play — and p the share of buyers who ever earn it.

def classify(name, description):
    """Return (mu_prior, ceiling_prior) for a trophy."""
    d = (description or "").lower()
    n = (name or "").lower()
    text = d + " " + n
    if "100% completion" in text:
        return math.log(1800.0), 0.06
    if "complete chapter 1" in text:
        return math.log(24.0), 0.85
    if "complete chapter 2" in text:
        return math.log(72.0), 0.70
    if "complete chapter 3" in text:
        return math.log(168.0), 0.55
    if "complete chapter 4" in text:
        return math.log(312.0), 0.45
    if "final mission" in text:
        return math.log(600.0), 0.30
    if "complete" in text and ("chapter" in text or "mission" in text):
        return math.log(240.0), 0.50
    # Grind: big counts, "all", "every" — slow, low ceiling.
    if any(k in text for k in ("250 stunts", "100 locations", "50 challenges",
                                "20 different races", "3 collectible card",
                                "gold medal in all", "every location",
                                "all garage", "all shooting")):
        return math.log(1200.0), 0.10
    if any(k in text for k in ("collect", "find all", "discover 100",
                                "waInk it", "meditate at every")):
        return math.log(900.0), 0.15
    # Quick one-offs: try once, discover, enjoy, a single event.
    if any(k in text for k in ("try each", "complete an arcade", "discover",
                                "enjoy a", "follow-through", "bowl 'em",
                                "take down 5", "rob a", "rob 3", "rob 5",
                                "rob an armored", "explore all", "earn $",
                                "spend $", "register 10", "purchase all")):
        return math.log(96.0), 0.45
    return math.log(300.0), 0.35


# --- case handling ---------------------------------------------------------

def distinct_states(case, t0):
    """Dedupe the history to states where (players, rates) changed.

    Returns a list of (tau_hours, players, [rates]) in time order, keeping
    the first read of each distinct state. Rates are percent, None where a
    read lacked the trophy.
    """
    states = []
    prev = None
    for read in case.get("history", []):
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key == prev:
            continue
        prev = key
        tau = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
        states.append((tau, read.get("players_tracked") or 0,
                       list(read.get("rates") or [])))
    return states


def last_rates(states, n_trophies):
    """Rates of the last state, with None replaced by the last known value."""
    out = [None] * n_trophies
    for _, _, rates in states:
        for i, r in enumerate(rates):
            if r is not None:
                out[i] = r
    return [0.0 if r is None else r for r in out]


def subsample(states, cap):
    """Evenly subsample states to at most `cap`, always keeping the last."""
    if len(states) <= cap:
        return list(states)
    idx = [round(i * (len(states) - 1) / (cap - 1)) for i in range(cap)]
    return [states[i] for i in sorted(set(idx))]


# --- the three views --------------------------------------------------------

def model_persistence(states, trophies, tau_target, horizon_days):
    last = last_rates(states, len(trophies))
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    return last, [max(spread, 0.15)] * len(trophies)


def model_logit_trend(states, trophies, tau_target, horizon_days):
    """Logit(rate) linear in ln(1 + hours), fitted over the last 24 h."""
    last = last_rates(states, len(trophies))
    tau_last = states[-1][0]
    window = [(t, r) for (t, _, r) in states if t >= tau_last - 24.0]
    medians, spreads = [], []
    for i in range(len(trophies)):
        xs = [math.log1p(t) for (t, r) in window if r[i] is not None]
        ys = [logit(r[i]) for (t, r) in window if r[i] is not None]
        if len(xs) >= 3:
            a, b, resid = least_squares(xs, ys)
            med = inv_logit(a + b * math.log1p(tau_target))
            s = math.sqrt(resid * resid + 0.25 * max(horizon_days, 0.0))
        else:
            med = last[i]
            s = 0.5 * math.sqrt(max(horizon_days, 0.0))
        medians.append(clamp_pct(med))
        spreads.append(max(s, 0.15))
    return medians, spreads


def fit_player_growth(states):
    """Power law N ~ C * tau**beta; returns (beta, log C)."""
    xs = [math.log(max(t, 0.05)) for (t, n, _) in states if n and n > 0]
    ys = [math.log(n) for (t, n, _) in states if n and n > 0]
    if len(xs) >= 3:
        a, b, _ = least_squares(xs, ys)
        return max(0.05, min(2.0, b)), a
    return 0.5, (ys[0] - 0.5 * xs[0]) if xs else 0.0


def _convolution_rate(beta, log_c, mu, sigma, p, tau):
    """Rate at tau from the structural model.

    N(tau) = C * tau**beta, inflow n(s) = C * beta * s**(beta-1), and
    earners E(tau) = p * ∫ n(s) F(tau - s) ds with F the lognormal CDF of
    the delay. The integral is on a 2 h grid over the delay u = tau - s.
    """
    if tau <= 0:
        return 0.0
    n_tau = math.exp(log_c + beta * math.log(max(tau, 0.05)))
    if n_tau <= 0:
        return 0.0
    step = 2.0
    total = 0.0
    u = step
    while u <= tau:
        f = norm_cdf((math.log(u) - mu) / sigma) if u > 0 else 0.0
        s = tau - u
        if s > 0.05:
            total += math.exp(log_c) * beta * (s ** (beta - 1.0)) * f * step
        u += step
    return 100.0 * p * total / n_tau


def fit_structural_trophy(obs, beta, log_c, mu_prior, p_prior):
    """Fit (p, mu, sigma) for one trophy on its observed (tau, rate) pairs.

    Grid over mu and sigma; given both, the best p in logit space is the
    mean residual. A prior penalty keeps mu and p sane when the history
    is short or the trophy is still at zero. Returns (p, mu, sigma, resid).
    """
    data = [(t, r) for (t, r) in obs if r is not None]
    if not data:
        return p_prior, mu_prior, 0.8, 1.0
    if all(r <= 0.0 for (_, r) in data):
        return p_prior, mu_prior, 0.8, 1.0
    mu_grid = [math.log(h) for h in
               (6.0, 12.0, 24.0, 48.0, 96.0, 168.0, 300.0, 500.0,
                800.0, 1200.0, 1800.0, 2600.0)]
    best = None
    for mu in mu_grid:
        for sigma in (0.6, 1.0):
            terms = []
            ok = True
            for (t, r) in data:
                base = _convolution_rate(beta, log_c, mu, sigma, 1.0, t)
                if base <= 1e-6:
                    if r > 0.05:
                        ok = False
                        break
                    terms.append(logit(0.01) - logit(1.0))
                else:
                    terms.append(logit(r) - logit(base))
            if not ok or not terms:
                continue
            lp = mean(terms)
            p = 1.0 / (1.0 + math.exp(-lp)) if -30 < lp < 30 else (
                1.0 if lp >= 30 else 0.0)
            sse = 0.0
            for (t, r) in data:
                base = _convolution_rate(beta, log_c, mu, sigma, 1.0, t)
                pred = logit(base * p) if base > 1e-6 else logit(0.01)
                sse += (logit(r) - pred) ** 2
            pen = (2.0 * (mu - mu_prior) ** 2
                   + 1.0 * (lp - logit(p_prior * 100.0)) ** 2)
            score = sse + pen
            if best is None or score < best[0]:
                best = (score, p, mu, sigma, math.sqrt(sse / len(data)))
    if best is None:
        return p_prior, mu_prior, 0.8, 1.0
    return best[1], best[2], best[3], best[4]


def model_structural(states, trophies, tau_target, horizon_days):
    """Structural delay model, per trophy, with description priors.

    Returns (medians, fit_residuals): the residuals double as a quality
    signal — a large residual means the fit could not explain the history,
    and the caller down-weights this view for that trophy.
    """
    beta, log_c = fit_player_growth(states)
    fit_states = subsample(states, 12)
    medians, residuals = [], []
    for i, tr in enumerate(trophies):
        mu_prior, p_prior = classify(tr.get("name", ""),
                                      tr.get("description", ""))
        obs = [(t, r[i]) for (t, _, r) in fit_states if i < len(r)]
        p, mu, sigma, resid = fit_structural_trophy(
            obs, beta, log_c, mu_prior, p_prior)
        med = _convolution_rate(beta, log_c, mu, sigma, p, tau_target)
        medians.append(clamp_pct(med))
        residuals.append(resid)
    return medians, residuals


# --- per-case forecast ------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies", [])
    n = len(trophies)
    rel = str(case.get("game", {}).get("release_date", "2026-11-19"))
    t0 = parse_time(rel + "T00:00:00Z") if len(rel) <= 10 else parse_time(rel)
    target = parse_time(case["target"])
    tau_target = (target - t0).total_seconds() / 3600.0
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    states = distinct_states(case, t0)
    last = last_rates(states, n) if states else [0.0] * n

    if not states:
        medians = [0.63] * n
        spreads = [1.0] * n
    elif len(states) < 4:
        # Flat history (the practice cases): no trend can be fitted. The
        # next Exophase refresh typically multiplies the tracked players
        # several-fold, which dilutes most rates down; forecast that, with
        # a wide spread because the size of the refresh is unknown.
        medians = [r * 0.5 if r < 99.9 else 99.5 for r in last]
        spreads = [1.2] * n
    else:
        med_trend, _ = model_logit_trend(
            states, trophies, tau_target, horizon_days)
        med_struct, resid_struct = model_structural(
            states, trophies, tau_target, horizon_days)
        med_pers, _ = model_persistence(
            states, trophies, tau_target, horizon_days)
        spread = min(1.5, max(0.6, SPREAD_REF * math.sqrt(
            max(horizon_days, 0.0) / SPREAD_HORIZON_DAYS)))
        medians, spreads = [], []
        for i in range(n):
            w_s = W_STRUCTURAL
            if resid_struct[i] > 1.5:
                # The structural fit could not explain this trophy's
                # history; drop that view for it and renormalise.
                w_s = 0.0
            total = W_TREND + w_s + W_PERSISTENCE
            w_t = W_TREND / total
            w_s = w_s / total
            w_p = W_PERSISTENCE / total
            base = (w_t * logit(med_trend[i])
                    + w_s * logit(med_struct[i])
                    + w_p * logit(med_pers[i]))
            medians.append(inv_logit(base))
            spreads.append(spread)

    rows = []
    for i, tr in enumerate(trophies):
        if last[i] >= 99.9:
            # A trophy everyone already has: it almost certainly stays
            # near-universal, but a wave of brand-new players who have
            # not finished the prologue yet can pull it down a little,
            # so the median sits at 99.5 with a real lower tail.
            base = logit(99.5)
            s = 0.7
        else:
            base = logit(medians[i])
            s = spreads[i]
        q = [round(clamp_pct(inv_logit(base + Z[j] * s)), 4)
             for j in range(len(LEVELS))]
        # Keep the quantiles non-decreasing after rounding.
        for j in range(1, len(q)):
            q[j] = max(q[j], q[j - 1])
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-len(".in")]
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        levels = case.get("levels", LEVELS)
        if list(levels) != list(LEVELS):
            raise ValueError(
                f"{fname} asks for levels {levels}, not {LEVELS}")
        out = forecast_case(case)
        data = json.dumps(out, separators=(",", ":")) + "\n"
        if len(data.encode("utf-8")) > 65536:
            raise ValueError(f"{fname} output exceeds 65,536 bytes")
        with open(os.path.join(output_dir, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(data)


if __name__ == "__main__":
    main()
