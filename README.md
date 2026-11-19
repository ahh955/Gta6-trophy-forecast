[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ahh955/Gta6-trophy-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

An entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/c/gta6-trophy-rarity-forecast-01) on make.fun: a program that reads Exophase's recorded earn rates for all 55 PlayStation 5 trophies of Grand Theft Auto VI and forecasts each trophy's rate at 2026-11-26 00:00 UTC — one week after launch — as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

## The model

An earn rate is earners divided by players tracked, and both counts grow over launch week, so the program forecasts the two counts separately and divides. Both are fitted in log-log time (log count against log hours since the release day began), because launch growth is multiplicative and slows in clock time.

- **Distinct pages only.** Exophase refreshes a game's page on its own schedule, hours apart, so consecutive identical reads are one page, not many observations. Every fit runs only on distinct pages (deduped by players, trophies earned, and the full rate vector).
- **Players:** a two-window fit — the full history and the last 24 hours, averaged — with slopes damped, so the launch spike is not extrapolated forever. The windows' disagreement feeds the spread.
- **Earners, per trophy:** the full history as a line blended with a quadratic (capped against straying far from the line), averaged with the last day's damped line, weighted 80% to the recent window. The recent window matters most: a saturating trophy's early log-log slope (its rate rising from zero) is far steeper than its slope at the deadline, so a full-window line overshoots badly. Noisy fits shrink toward their archetype's typical growth, with the shrinkage weight set by the fit's t-statistic. Archetypes (prologue / story / side / grind / misc / completion / platinum) are classified from each trophy's own text.
- **A second opinion on the rate.** The ratio forecast is blended, in log rate (70/30), with a damped line of the rate's own logit trend over the last day. The logit trend never compounds two forecasts' errors and saturates the way a rate does. Its damping is adaptive: a trophy whose recent slope already runs below its full-history slope is decelerating and gets damped toward 0.8; a steady or accelerating one is read undamped.
- **Structure the game imposes.** The story funnel runs downhill (prologue ≥ chapter 1 ≥ … ≥ final mission); 100% completion rides on the final mission (blended with the historical ratio × 1.25, capped at the final mission); the platinum rides on completion and can never exceed the smallest other median. A near-universal trophy (last rate ≥ 97%) — the prologue mission, which launch night's sixfold refresh left at 100% — is forecast at the top of the scale (median 99.95, quantiles 98.5–99.99): the logit scale punishes an undershoot near 100% far more than an overshoot.
- **Flat history** (no refresh observed, as in the practice cases) gets a dilution haircut instead of a fit: a refresh multiplies the tracked players far faster than the earners, so rates fall — launch night's refresh grew players 6.06× while non-prologue earners grew only 1.34×. The haircut is archetype-weighted (early story trophies keep the most) with a wide spread.
- **Spread:** a random walk over the horizon (0.5 logit units per √day) combined with the recent fit's residual and the players windows' disagreement, a 0.25 floor, and an archetype multiplier (story 0.9, grind 1.05, completion/platinum 1.1).

The program is deterministic (two runs on the same case write identical bytes), standard library only, no network, and runs in ~0.1 s per case. Every forecast passes a strict format check before it is written: 55 rows, exact trophy ids, five finite 0–100 non-decreasing quantiles each, under 65,536 bytes.

## Backtests

The kit ships no practice answers, so the practice was scored against plausible answers reconstructed from the facts the competition documents (the 04:30 UTC refresh: 3,868 players tracked, 4,276 trophies earned, every rate but the 100% one lower). The starter's documented practice loss is ~1.51; this reconstruction reproduces it (1.51, typical miss 0.68 pp, coverage 0), which is a check on the reconstruction.

| Test | This entry | Persistence baseline | Curve baseline |
| --- | --- | --- | --- |
| Practice, scenario A | 0.19 | 1.51 | 1.51 |
| Practice, scenario B | 0.17 | 1.40 | 1.40 |
| Practice, scenario C | 0.17 | 1.41 | 1.41 |
| Synthetic final, slow launch (3 seeds) | **0.25** | 0.38 | 0.25 |
| Synthetic final, fast launch (3 seeds) | **0.27** | 0.31 | 0.27 |

The synthetic finals are my own simulation (a plausible launch week: players growing log-log with a decelerating slope, each trophy's rate saturating toward an archetype ceiling, Exophase refreshing every 2–4 hours), used to test the machinery end to end — not official results. The entry beats both house baselines in both simulated worlds, with a much smaller typical miss than either (1.1–1.2 pp vs 1.8–2.1 pp for the curve baseline in the slow world).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input directory and writes one `<id>.out` per case. The grader runs `python3 /bot/main.py` with `/input` and `/output`.

## Files

- `main.py` — the forecaster (Python 3.13, standard library only)
- `README.md` — this file
- `LICENSE` — MIT
