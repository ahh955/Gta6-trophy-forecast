"""GTA VI trophy-rarity forecaster, round 3.

Reads one case per run from ``/input/<id>.in`` (schema
``makefun/gta6-trophies-case/v1``) and writes ``/output/<id>.out``: for every
PlayStation 5 trophy of Grand Theft Auto VI, five quantiles (5%, 25%, 50%,
75%, 95%) of its Exophase earn rate at the case's target. The grade is a
pinball loss on the logit scale, so the medians must be honest and the
spreads calibrated, not just narrow.

How the forecast is built
-------------------------
Earners and players are forecast separately, in log-time
``ln(1 + hours since the release day began)``, with saturating curves of the
form ``log count = a - b * exp(-k * x)``. Only distinct pages are fitted:
consecutive identical reads are one page, because Exophase refreshes a
game's counts on its own schedule and re-reading the same page adds no
information.

Each trophy's earners curve is capped by an archetype asymptote prior
(story chapters cannot out-earn their funnel, a platinum cannot out-earn
the rarest trophy), shrunk toward its archetype's typical growth when its
own fit is noisy, and blended with a direct damped line of the rate's logit
trend. Story-funnel, completion and platinum structure is enforced on the
medians afterwards.

A flat history -- every read identical, no refresh seen yet -- gets a
dilution forecast from the documented launch-night refresh (638 players
tracked before it, 3868 after): the rates fall by the refresh factor while
earners grow by a modest archetype multiplier. Round 3 retunes those flat
constants from a robust calibration of the practice cases; the rich-history
path is unchanged.

Standard library only, no network, deterministic: the same case always
writes the same forecast, byte for byte.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Model constants, rich-history path (unchanged from round 2) ---------
DAMP = 0.8            # how much of a fitted slope survives past the last read
RECENT_HOURS = 24.0   # the recent window blended into every forecast
SHRINK_K = 4.0        # shrinkage strength for noisy per-trophy growth
RW_SIGMA = 0.25       # random-walk spread per sqrt(day) of horizon
SPREAD_FLOOR = 0.30   # narrowest logit spread any trophy is allowed
SPREAD_CAP = 1.5      # widest logit spread any trophy is allowed
SAT_MIN_POINTS = 5    # points needed before a saturating fit is tried
SAT_GRID = 48         # grid size for the saturating curve's decay rate

# Asymptote priors: no trophy of an archetype is forecast above its cap,
# unless its last rate already sits above it.
RATE_CEILING = {
    "prologue": 0.999,
    "story": 0.60,
    "side": 0.35,
    "grind": 0.25,
    "misc": 0.45,
    "completion": 0.08,
    "platinum": 0.03,
}
CHAPTER_CEILING = {1: 0.90, 2: 0.80, 3: 0.70, 4: 0.60, 99: 0.45}

# Per-archetype spread multipliers on the combined base spread.
ARCH_SPREAD = {
    "prologue": 0.9,
    "story": 0.9,
    "side": 1.0,
    "grind": 1.05,
    "misc": 1.0,
    "completion": 1.1,
    "platinum": 1.1,
}

# A trophy the launch crowd already holds almost universally keeps a
# near-universal row. Do not tighten this: it is shared with the final path.
NEAR_UNIV_Q = [98.0, 99.6, 99.95, 99.99, 99.99]

# --- Flat-history path (retuned in round 3) ------------------------------
# The documented launch-night refresh: 638 players tracked before it,
# 3868 after. Rates dilute by this factor; earners grow by the multipliers.
REFRESH_PLAYERS = 3868.0 / 638.0
FLAT_SIGMA = 0.1337
FLAT_EARNERS = {
    "prologue": REFRESH_PLAYERS,
    "story": 1.1515625,
    "side": 1.24296875,
    "grind": 1.515625,
    "misc": 1.2234375,
    "completion": 1.0171875,
    "platinum": 1.3,
}
FLAT_CHAPTER = {1: 1.08984375, 2: 1.3, 3: 1.23984375, 4: 1.3}


def clamp_rate(x):
    """Keep a percent inside the logit's domain, [0.01, 99.99]."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    """Logit of a percent rate, clamped first."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(lg):
    """Inverse logit as a percent, stably on both sides of zero."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero; drop a trailing .0."""
    scaled = x * 10000.0
    if scaled >= 0:
        whole = math.floor(scaled + 0.5)
    else:
        whole = math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_ts(text):
    """Parse an ISO timestamp, or None when it is not one."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def mid(values):
    """Median of a list; 0.0 for an empty one."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def fit_line(xs, ys):
    """Least-squares line; returns (intercept, slope, rms, t_slope).

    With no spread in x there is no slope to fit: the mean, flat. The
    t-statistic is 0 when there are too few points to judge the slope by.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, 0.0)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    rms = math.sqrt(rss / n)
    if n <= 2:
        return (intercept, slope, rms, 0.0)
    se = math.sqrt(rss / (n - 2) / sxx)
    t = slope / se if se > 0 else 0.0
    return (intercept, slope, rms, t)


def fit_saturating(xs, ys):
    """Fit y = a - b * exp(-k * x); returns (a, b, k, rms) or None.

    The decay rate k is gridded log-spaced from 0.05 to 4; given k the fit
    is linear in (a, b). Only rising curves count (b > 0): a falling
    saturating curve is not a launch curve. The best grid point wins on
    residual sum of squares, the first of any tie.
    """
    n = len(xs)
    best = None
    for gi in range(SAT_GRID):
        k = 0.05 * (80.0 ** (gi / (SAT_GRID - 1)))
        zs = [math.exp(-k * x) for x in xs]
        mz = sum(zs) / n
        my = sum(ys) / n
        szz = sum((z - mz) ** 2 for z in zs)
        if szz <= 0:
            continue
        # y = a + c * z with c = -b, fitted as a line in z.
        c = sum((z - mz) * (y - my) for z, y in zip(zs, ys)) / szz
        if c >= 0:
            continue
        a = my - c * mz
        rss = sum((y - (a + c * z)) ** 2 for z, y in zip(zs, ys))
        if best is None or rss < best[0]:
            best = (rss, a, -c, k)
    if best is None:
        return None
    rss, a, b, k = best
    return (a, b, k, math.sqrt(rss / n))


def line_at(intercept, slope, x_anchor, x_target, damp=DAMP):
    """A fitted line read at the target, its slope damped past the anchor."""
    return intercept + slope * x_anchor + damp * slope * (x_target - x_anchor)


def series_forecast(xs, ys, x_target, x_recent):
    """Forecast one log-count series; returns (pred, rms, t_slope).

    A damped line is the base. With enough points a saturating curve
    replaces it when it fits at least as well. A recent-window line, when
    it has enough points, is blended half and half, and its residual can
    only widen the spread.
    """
    intercept, slope, rms, t = fit_line(xs, ys)
    pred = line_at(intercept, slope, xs[-1], x_target)
    if len(xs) >= SAT_MIN_POINTS:
        sat = fit_saturating(xs, ys)
        if sat is not None and sat[3] <= rms:
            a, b, k, sat_rms = sat
            pred = a - b * math.exp(-k * x_target)
            rms = sat_rms
    if x_recent is not None:
        recent = [(x, y) for x, y in zip(xs, ys) if x >= x_recent]
        if len(recent) >= 3:
            rxs = [p[0] for p in recent]
            rys = [p[1] for p in recent]
            ri, rs, rrms, _rt = fit_line(rxs, rys)
            recent_pred = line_at(ri, rs, rxs[-1], x_target)
            pred = 0.5 * pred + 0.5 * recent_pred
            rms = max(rms, rrms)
    return (pred, rms, t)


def classify_trophy(trophy):
    """Sort a trophy into (archetype, chapter or None, is_final_mission).

    First match wins, so the order is the classifier: platinum before
    completion, the prologue marker before the chapter scan, a "final
    mission for ..." side strand before the story's final mission.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return ("platinum", None, False)
    if "100% completion" in desc or "100% completion" in name:
        return ("completion", None, False)
    if "gone fishing" in desc:
        return ("prologue", 0, False)
    if "final mission for" in desc or "strangers" in desc:
        return ("side", None, False)
    if "final mission" in desc:
        return ("story", 99, True)
    for ch in (1, 2, 3, 4, 5, 6):
        if ("chapter %d" % ch) in desc:
            return ("story", ch, False)
    if "chapter" in desc:
        return ("story", None, False)
    for marker in (
        "win ",
        "earn $",
        "spend $",
        "purchase all",
        "register",
        "explore all",
        "meditate at every",
        "discover",
        "rob ",
    ):
        if marker in text:
            return ("grind", None, False)
    if desc.startswith("complete") and any(ch.isdigit() for ch in desc):
        return ("grind", None, False)
    if desc.startswith("complete") and trophy.get("secret"):
        return ("story", None, False)
    return ("misc", None, False)


def ceiling_for(arch, chapter, last_rate):
    """The highest rate a trophy is forecast at: its chapter cap, else its
    archetype cap, never below the rate it already holds."""
    cap = CHAPTER_CEILING.get(chapter, RATE_CEILING.get(arch, 0.45))
    return max(cap, last_rate / 100.0)


def distinct_pages(history):
    """Collapse consecutive identical reads to one page each, first seen.

    A read counts only when it carries a player count and a rates list.
    Two reads are the same page when the player count, the trophies-earned
    count and every rate match.
    """
    pages = []
    last_sig = None
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or not isinstance(rates, list):
            continue
        sig = (players, read.get("trophies_earned"), tuple(rates))
        if sig == last_sig:
            continue
        last_sig = sig
        pages.append(read)
    return pages


def read_time(read):
    """When a read was taken: its `at`, else its clock slot."""
    return parse_ts(read.get("at")) or parse_ts(read.get("slot"))


def log_time(t, release_start):
    """ln(1 + hours from the release day's start to t); 0 without a time."""
    if t is None or release_start is None:
        return 0.0
    hours = (t - release_start).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def direct_rate(pages, xs, idx, x_target, x_recent, last_rate):
    """A trophy's rate from a damped line of its logit trend, recent reads
    only when a recent window is set. Falls back to the last rate when
    there are too few points to fit a line."""
    pts = []
    for x, page in zip(xs, pages):
        rates = page.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        if x_recent is not None and x < x_recent:
            continue
        pts.append((x, logit_of(rates[idx])))
    if len(pts) < 3:
        return last_rate
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    intercept, slope, _rms, _t = fit_line(px, py)
    return clamp_rate(rate_of_logit(line_at(intercept, slope, px[-1], x_target)))


def apply_structure(trophies, kinds, last_rates, medians):
    """Enforce what the trophies' structure says about the medians, in
    place: near-universal rates stay near-universal, the story funnel
    never rises chapter to chapter, completion sits below the final
    mission in proportion to their last rates, and the platinum sits
    below completion and below every other trophy."""
    for i in range(len(trophies)):
        if last_rates[i] >= 97.0:
            medians[i] = 99.95
    chain = []
    for i, (arch, chapter, _fm) in enumerate(kinds):
        if arch == "prologue":
            chain.append((0, i))
        elif arch == "story" and chapter is not None:
            chain.append((chapter, i))
    chain.sort()
    running = None
    for _ch, i in chain:
        if running is None:
            running = medians[i]
        else:
            running = min(running, medians[i])
            medians[i] = running
    final_idx = None
    completion_idx = None
    platinum_idx = None
    for i, (arch, _chapter, is_final) in enumerate(kinds):
        if arch == "story" and is_final:
            if final_idx is None:
                final_idx = i
        elif arch == "completion":
            if completion_idx is None:
                completion_idx = i
        elif arch == "platinum":
            if platinum_idx is None:
                platinum_idx = i
    if completion_idx is not None and final_idx is not None:
        fm_last = last_rates[final_idx]
        ratio = last_rates[completion_idx] / fm_last if fm_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = min(medians[final_idx] * ratio * 1.25, medians[final_idx])
        blended = math.sqrt(
            max(medians[completion_idx], 0.01) * max(anchor, 0.01)
        )
        medians[completion_idx] = min(blended, medians[final_idx])
    if platinum_idx is not None and completion_idx is not None:
        comp_last = last_rates[completion_idx]
        ratio = last_rates[platinum_idx] / comp_last if comp_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = medians[completion_idx] * ratio
        medians[platinum_idx] = math.sqrt(
            max(medians[platinum_idx], 0.01) * max(anchor, 0.01)
        )
    if platinum_idx is not None:
        others = [medians[i] for i in range(len(medians)) if i != platinum_idx]
        if others:
            medians[platinum_idx] = min(medians[platinum_idx], min(others))


def validate(forecast, trophies):
    """Raise unless the forecast is exactly the shape the grader wants:
    one `trophies` key, one row per trophy, ids unique and from the case,
    five finite non-decreasing quantiles in [0, 100] each, and a payload
    of at most 65,536 bytes."""
    if set(forecast.keys()) != {"trophies"}:
        raise ValueError("forecast must have exactly the key 'trophies'")
    rows = forecast["trophies"]
    if len(rows) != len(trophies):
        raise ValueError("row count must match trophy count")
    trophy_ids = {t["id"] for t in trophies}
    seen = set()
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            raise ValueError("each row must have exactly 'id' and 'q'")
        if row["id"] in seen or row["id"] not in trophy_ids:
            raise ValueError("ids must be unique and from the case")
        seen.add(row["id"])
        q = row["q"]
        if len(q) != 5:
            raise ValueError("each row needs five quantiles")
        for j, value in enumerate(q):
            if not math.isfinite(value) or value < 0 or value > 100:
                raise ValueError("quantiles must be finite numbers in [0, 100]")
            if j > 0 and value < q[j - 1]:
                raise ValueError("quantiles must be non-decreasing")
    payload = json.dumps(forecast, separators=(",", ":")) + "\n"
    if len(payload.encode("utf-8")) > 65536:
        raise ValueError("forecast payload exceeds 65536 bytes")


def forecast_case(case):
    """Forecast one case: medians and spreads per trophy, then quantiles."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    checkpoint = parse_ts(case.get("at"))
    target = parse_ts(case.get("target"))
    game = case.get("game") or {}
    release_start = parse_ts((game.get("release_date") or "") + "T00:00:00Z")
    if release_start is None and history:
        first = read_time(history[0])
        if first is not None:
            release_start = first.replace(hour=0, minute=0, second=0, microsecond=0)
    if target is None and checkpoint is not None:
        target = checkpoint + timedelta(minutes=case.get("horizon_minutes") or 0)
    x_target = log_time(target, release_start)
    kinds = [classify_trophy(t) for t in trophies]
    last_rates = []
    for i in range(len(trophies)):
        rate = 1.0
        for read in reversed(history):
            rates = read.get("rates")
            if rates is not None and i < len(rates) and rates[i] is not None:
                rate = rates[i]
                break
        last_rates.append(rate)
    pages = distinct_pages(history)
    medians = [None] * len(trophies)
    sigmas = [None] * len(trophies)
    if len(pages) <= 1:
        # Flat history: no refresh seen yet. Dilute by the documented
        # refresh factor while earners grow by their archetype multiplier.
        for i in range(len(trophies)):
            arch, chapter, _fm = kinds[i]
            mult = FLAT_EARNERS.get(arch, 1.2)
            if arch == "story" and chapter in FLAT_CHAPTER:
                mult = FLAT_CHAPTER[chapter]
            rate = last_rates[i] * mult / REFRESH_PLAYERS
            medians[i] = clamp_rate(rate)
            sigmas[i] = math.sqrt(
                FLAT_SIGMA ** 2
                + (RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2
            )
    else:
        xs = [log_time(read_time(p), release_start) for p in pages]
        for j in range(1, len(xs)):
            if xs[j] <= xs[j - 1]:
                xs[j] = xs[j - 1] + 1e-6
        if checkpoint is not None:
            x_recent = log_time(
                checkpoint - timedelta(hours=RECENT_HOURS), release_start
            )
        else:
            x_recent = None
        # Players first: every trophy's earners are divided by them.
        ys_players = [math.log(max(1, p["players_tracked"])) for p in pages]
        log_players, _players_rms, _pt = series_forecast(
            xs, ys_players, x_target, x_recent
        )
        log_players = max(log_players, ys_players[-1])
        _fi, _fs, _frms, _ft = fit_line(xs, ys_players)
        full_only = line_at(_fi, _fs, xs[-1], x_target)
        players_disagree = abs(full_only - log_players)
        # Pass 1: each trophy's earners curve on its own.
        raw = []
        for i in range(len(trophies)):
            series = []
            for x, page in zip(xs, pages):
                rates = page.get("rates")
                if rates is None or i >= len(rates) or rates[i] is None:
                    continue
                earners = rates[i] / 100.0 * page["players_tracked"]
                series.append((x, math.log(max(0.5, earners))))
            if not series:
                raw.append(None)
                continue
            sx = [p[0] for p in series]
            sy = [p[1] for p in series]
            if len(series) == 1:
                raw.append((sy[0], 0.3, 0.0, sy[0]))
                continue
            pred, rms, t = series_forecast(sx, sy, x_target, x_recent)
            pred = max(pred, sy[-1] - 0.02)
            raw.append((pred, rms, t, sy[-1]))
        # Typical growth per archetype, and overall, for shrinkage.
        growths = {}
        all_growths = []
        for i, out in enumerate(raw):
            if out is None:
                continue
            arch = kinds[i][0]
            growth = out[0] - out[3]
            growths.setdefault(arch, []).append(growth)
            all_growths.append(growth)
        arch_growth = {a: mid(g) for a, g in growths.items()}
        global_growth = mid(all_growths) or 0.0
        # Pass 2: shrink, cap, blend with the direct rate line, spread.
        rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
        for i in range(len(trophies)):
            out = raw[i]
            if out is None:
                medians[i] = clamp_rate(last_rates[i])
                sigmas[i] = max(SPREAD_FLOOR, rw)
                continue
            pred, rms, t, last_y = out
            arch, chapter, _fm = kinds[i]
            w = (t * t) / (t * t + SHRINK_K) if t else 0.0
            norm = arch_growth.get(arch, global_growth)
            if len(growths.get(arch, [])) <= 1:
                norm = global_growth
            growth = pred - last_y
            shrunk = w * growth + (1.0 - w) * norm
            log_earners = last_y + shrunk
            cap_rate = ceiling_for(arch, chapter, last_rates[i])
            log_cap = math.log(max(0.5, cap_rate * math.exp(log_players)))
            log_earners = min(log_earners, log_cap)
            rate_counts = clamp_rate(100.0 * math.exp(log_earners - log_players))
            rate_direct = direct_rate(
                pages, xs, i, x_target, x_recent, last_rates[i]
            )
            w_direct = 0.2 + 0.3 * min(1.0, rms / 0.5)
            rate = math.exp(
                (1.0 - w_direct) * math.log(rate_counts)
                + w_direct * math.log(clamp_rate(rate_direct))
            )
            medians[i] = clamp_rate(rate)
            base = math.sqrt(
                rw * rw
                + min(rms, 1.0) ** 2
                + min(players_disagree, 1.0) ** 2
            )
            sigmas[i] = min(
                max(base, SPREAD_FLOOR) * ARCH_SPREAD.get(arch, 1.0),
                SPREAD_CAP,
            )
    apply_structure(trophies, kinds, last_rates, medians)
    rows = []
    for i, trophy in enumerate(trophies):
        med = clamp_rate(medians[i])
        sigma = sigmas[i]
        if sigma is None or not math.isfinite(sigma):
            sigma = max(SPREAD_FLOOR, RW_SIGMA * math.sqrt(max(horizon_days, 0.0)))
        if last_rates[i] >= 97.0:
            rows.append({"id": trophy["id"], "q": list(NEAR_UNIV_Q)})
            continue
        centre = logit_of(med)
        qs = [
            round4(min(100.0, max(0.0, rate_of_logit(centre + z * sigma))))
            for z in Z
        ]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    forecast = {"trophies": rows}
    validate(forecast, trophies)
    return forecast


def main():
    """Run every case in the input directory, one forecast file each."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
