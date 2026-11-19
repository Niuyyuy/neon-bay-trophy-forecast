[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Niuyyuy/neon-bay-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Niuyyuy/neon-bay-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# Neon Bay Trophy Forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — one week after launch. An entry for the
[make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
Try make.fun yourself: https://make.fun/r/0wvmb34n

## The program

`main.py` (Python 3.13, standard library only, no network) is the whole
entry. It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): one row per
trophy, `{"id": n, "q": [q05, q25, q50, q75, q95]}`, five quantiles of the
trophy's earn rate at the case's target, in percent.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

An earn rate is earners / players-tracked, and both sides move: players
join over time (the denominator dilutes every rate) while the players
already in keep playing and earning (the numerator grows, trophy by trophy,
after a delay that depends on how far into the game the trophy sits — and
that delay is a *distribution* across players, not one number). The program
blends three views of that, in logit space:

- **0.6 — logit-linear trend:** `logit(rate)` linear in `ln(1 + hours)`
  over the last 24 h of reads. Follows the recent slope; best for trophies
  rising fast right now, but overshoots late ones.
- **0.3 — structural delay model:** earners
  `E_t(tau) = p_t * ∫ n(s) F_t(tau - s) ds` over the buyer inflow `n(s)`
  (a power law in time since release, fitted from the players-tracked
  series), where `F_t` is the lognormal CDF of the calendar delay before a
  buyer earns trophy `t` and `p_t` the share of buyers who ever earn it.
  `F_t`'s centre and `p_t` are fitted per trophy, shrunk toward priors read
  off the trophy's own description ("Complete Chapter 1" is early, "Attain
  100% Completion" is very late). It saturates; best for late trophies,
  but undershoots fast risers.
- **0.1 — persistence:** the rate stays where the last read put it. A small
  anchor against both extrapolations running away.

The two extrapolations err in opposite directions, so averaging them in
logit space cancels much of the bias. The spread is 1.0 logit units at the
final's 4.5-day horizon, scaled by the square root of the horizon. When the
history is flat (the practice cases: every read identical), no trend can be
fitted, so the program falls back to a dilution-aware prior — the next
Exophase refresh typically multiplies the player count several-fold, which
pushes most rates down — with a wide spread.

## Validation

There is no local grade: the server grades every entry. Locally, the
program was checked on the kit's two practice cases (valid output shape,
55 rows, byte-identical across two runs, ~0.05 s per run) and on synthetic
launch-week cases built with a different generative model than the entry's
own (cohort buyers with lognormal play budgets and intensities, Exophase
refresh steps, 2-decimal rounding). Across twelve such cases the blend
scored a mean pinball loss of 0.34–0.49, against 0.43–0.84 for the house
curve baseline alone and 1.28–1.54 for persistence alone.

## License

MIT — see [LICENSE](LICENSE).
