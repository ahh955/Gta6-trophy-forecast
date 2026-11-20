# GTA VI Trophy Rarity Forecaster — Round 2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [make.fun's gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles per trophy,
scored by pinball loss on the logit scale.

## How it works

A rate is earners over players tracked, and both counts grow over launch
week, so the program forecasts the two counts separately and divides.

- **Log-time, saturating curves.** Both counts are fitted in log-time
  (log count against ln(1 + hours since the release day began)) with
  saturating curves, `log count = a - b·exp(-k·x)`, which rise toward an
  asymptote instead of running away like a line. A damped line is kept
  as the fallback when the saturating fit does not beat it.
- **Distinct pages only.** Exophase refreshes a game's page on its own
  schedule, hours apart, so consecutive identical reads are one page,
  not many observations: the fits run only on distinct pages.
- **Archetype asymptote priors.** Each trophy's earners forecast is
  capped by a week-one rate ceiling for its archetype (comparable
  launch-week curves: story trophies climb high, grinds stay low, the
  platinum stays ~1–3%), and noisy fits shrink toward their archetype's
  typical growth.
- **Two opinions, blended.** The counts ratio is blended with a direct
  damped line of the rate's own logit trend; the ratio carries most of
  the weight when its fits are stable.
- **Structure the game imposes.** The story funnel runs downhill
  (prologue ≥ chapter 1 ≥ … ≥ final mission), 100% completion rides on
  the final mission, the platinum rides on completion and can never
  exceed the smallest other median, and a trophy essentially every
  tracked player holds stays near universal.
- **Refresh path.** A history with no refresh at all (the practice
  cases) gets a dilution forecast instead of a fit: players jump by the
  documented launch-night refresh factor (638 → 3,868, ×6.0627) while
  earners grow modestly (~1.1–1.3× by archetype), with a narrow
  calibrated spread.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: two runs on the same
case write the same bytes.

## License

MIT — see [LICENSE](LICENSE).
