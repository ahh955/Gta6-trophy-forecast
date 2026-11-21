# GTA VI Trophy-Rarity Forecast — Round 3

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for [make.fun's GTA VI trophy-rarity bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): for every PlayStation 5 trophy of Grand Theft Auto VI, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at 2026-11-26 00:00 UTC, a week after launch. Scored by pinball loss on the logit scale.

## How it works

Two counts are forecast separately in log-time — ln(1 + hours since the release day began) — with saturating curves (`log count = a - b·exp(-k·x)`), fitted on distinct pages only: consecutive identical reads are one page, because Exophase refreshes a game's counts on its own schedule.

- **Earners** per trophy: saturating curve, capped by archetype asymptote priors, shrunk toward the archetype's typical growth when the trophy's own fit is noisy, blended with a direct damped line of the rate's logit trend.
- **Players**: the same curve family; every trophy's earners are divided by it.
- **Structure**: the story funnel never rises chapter to chapter, completion sits below the final mission in proportion to their last rates, and the platinum sits below completion and below every other trophy.
- **Flat history** (no refresh seen yet): a dilution forecast from the documented launch-night refresh (638 → 3,868 players tracked), with earners growing by a modest archetype multiplier.

Round 3 retunes only the flat-history constants, from a robust calibration of the practice cases (mean practice loss ≈ 0.089, down from 0.124). The rich-history path is unchanged — it produces byte-identical output to round 2 on a final-like case, so the forecast that decides the prize is exactly round 2's.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: the same case always writes the same forecast.

---

<sub>Entered on [make.fun](https://make.fun/?ref=xsff4jzh&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
