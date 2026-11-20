"""GTA VI trophy-rarity forecaster, round 2 (make.fun gta6-trophy-rarity-forecast-01).

Reads one case (/input/<id>.in, schema makefun/gta6-trophies-case/v1) and
writes /output/<id>.out: five quantiles (5/25/50/75/95%) of every PS5
trophy's Exophase earn rate at the case's target, scored by pinball loss
on the logit scale.

Model
-----
A rate is earners over players tracked, and both counts grow over launch
week, so the program forecasts the two counts separately and divides.
Both are fitted in log-time -- log count against ln(1 + hours since the
release day began) -- because launch growth is multiplicative and slows
in clock time, and with *saturating* curves, log count = a - b*exp(-k*x),
which rise toward an asymptote instead of running away the way a line
does.  Exophase refreshes a game's page on its own schedule, hours apart,
so consecutive identical reads are one page, not many observations: the
fits run only on distinct pages.

Each trophy's earners forecast is capped by an archetype asymptote prior
(comparable launch-week curves: story trophies climb high, the platinum
stays ~1-3% in week one), shrunk toward its archetype's typical growth
when the fit is noisy, and blended with a direct damped line of the
rate's own logit trend -- the counts ratio carries most of the weight
when its fits are stable.  Structure the game itself imposes is then
enforced: the story funnel runs downhill, 100% completion rides on the
final mission, the platinum rides on completion and can never exceed the
smallest other median, and a trophy essentially every tracked player
holds stays near universal.

A history with no refresh at all (the practice cases) gets a dilution
forecast instead of a fit: a refresh multiplies the tracked players far
faster than the earners, so rates fall.  The documented launch-night
refresh (638 -> 3,868 players, factor 6.0627) sets the players jump;
earners grow modestly, fastest for the trophies new players reach within
hours, with a narrow calibrated spread.

Standard library only, no network, deterministic: two runs on the same
case write the same bytes.
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

# --- refresh path (flat history): the documented launch-night refresh ---
REFRESH_PLAYERS = 3868.0 / 638.0   # 6.0627: Exophase's first refresh
FLAT_SIGMA = 0.25                  # narrow calibrated spread for one refresh
# Earners growth through one refresh, by archetype.  Modest: the refresh
# grew non-prologue earners only ~1.1-1.3x overall while players grew 6x.
FLAT_EARNERS = {
    "prologue": REFRESH_PLAYERS,  # newcomers arrive holding the first trophy
    "story": 1.25,
    "side": 1.25,
    "grind": 1.15,
    "misc": 1.2,
    "completion": 1.05,
    "platinum": 1.0,
}
FLAT_CHAPTER = {1: 1.5, 2: 1.3, 3: 1.2, 4: 1.15}  # early chapters taper

# --- rich-history path ---
DAMP = 0.8                # extrapolation damps every fitted line slope
RECENT_HOURS = 24.0       # recent window for the second-opinion fits
SHRINK_K = 4.0            # t^2/(t^2+K) weights a fit against its archetype
RW_SIGMA = 0.25           # random-walk spread, logit units per sqrt(day):
                          # backtested on synthetic final-like cases (see
                          # notes): 0.25 beats 0.35/0.45 on pinball in both
                          # a curve-matched and a misspecified simulator,
                          # with 90% coverage still ~0.99; the emitted
                          # spread stays ~0.7-0.8 via the residual terms.
SPREAD_FLOOR = 0.30       # minimum logit spread
SPREAD_CAP = 1.5          # maximum logit spread
SAT_MIN_POINTS = 5        # distinct pages needed before a saturating fit
SAT_GRID = 48             # k values tried in the saturating fit

# Week-one rate ceilings by archetype, from comparable launch-week curves
# (GTA V / RDR2): story trophies climb high, grinds stay low, the
# platinum stays ~1-3%.  A trophy already above its ceiling keeps its own
# level -- the ceiling is a prior, not a lid on observed progress.
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

ARCH_SPREAD = {
    "prologue": 0.9, "story": 0.9, "side": 1.0, "grind": 1.05,
    "misc": 1.0, "completion": 1.1, "platinum": 1.1,
}

# Fixed quantiles for a near-universal trophy: most of the mass at 100%,
# a left tail for the chance dilution finally bites.
NEAR_UNIV_Q = [98.0, 99.6, 99.95, 99.99, 99.99]


def clamp_rate(x):
    """A rate clamped into the scorer's logit domain, percent."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero, JSON-friendly."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_ts(text):
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def mid(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    half = n // 2
    if n % 2:
        return vals[half]
    return 0.5 * (vals[half - 1] + vals[half])


# ---------------------------------------------------------------------------
# Least squares: a line, and a saturating curve in log-time.
# ---------------------------------------------------------------------------

def fit_line(xs, ys):
    """Least-squares line; returns (intercept, slope, rms, t_slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    rms = math.sqrt(rss / n)
    if n <= 2:
        return intercept, slope, rms, 0.0
    se = math.sqrt(rss / (n - 2) / sxx)
    t = slope / se if se > 0 else 0.0
    return intercept, slope, rms, t


def fit_saturating(xs, ys):
    """Fit y = a - b*exp(-k*x) by gridding k, linear in (a, b) given k.

    Returns (a, b, k, rms) or None when no rising saturating curve fits
    (b must be positive: a count that grows toward an asymptote).
    """
    n = len(xs)
    best = None
    # k log-spaced from 0.05 to 4: slow saturation to fast.
    for gi in range(SAT_GRID):
        k = 0.05 * (80.0 ** (gi / (SAT_GRID - 1)))
        zs = [math.exp(-k * x) for x in xs]
        mz = sum(zs) / n
        my = sum(ys) / n
        szz = sum((z - mz) ** 2 for z in zs)
        if szz <= 0:
            continue
        # y = a + c*z with c = -b; need c < 0 for a rising curve.
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
    return a, b, k, math.sqrt(rss / n)


def line_at(intercept, slope, x_anchor, x_target, damp=DAMP):
    """A line read at x_target, anchored at the last observation, slope damped."""
    return intercept + slope * x_anchor + damp * slope * (x_target - x_anchor)


def series_forecast(xs, ys, x_target, x_recent):
    """Extrapolate one log-count series to x_target.

    Full history: the saturating curve when it beats the line on residual,
    else the damped line.  Recent window (last 24h): a damped line, read
    at the target.  The two are averaged -- the recent window reads the
    current slope, which is what matters for a saturating series, while
    the full fit steadies it.  Returns (prediction, rms, t_slope).
    """
    intercept, slope, rms, t = fit_line(xs, ys)
    line_pred = line_at(intercept, slope, xs[-1], x_target)
    pred = line_pred
    sat = fit_saturating(xs, ys) if len(xs) >= SAT_MIN_POINTS else None
    if sat is not None:
        a, b, k, sat_rms = sat
        if sat_rms <= rms:
            pred = a - b * math.exp(-k * x_target)
            rms = sat_rms
    if x_recent is not None:
        recent = [(x, y) for x, y in zip(xs, ys) if x >= x_recent]
        if len(recent) >= 3:
            rxs = [r[0] for r in recent]
            rys = [r[1] for r in recent]
            ri, rs, rrms, _rt = fit_line(rxs, rys)
            recent_pred = line_at(ri, rs, rxs[-1], x_target)
            pred = 0.5 * pred + 0.5 * recent_pred
            rms = max(rms, rrms)
    return pred, rms, t


# ---------------------------------------------------------------------------
# Trophy archetypes, from the trophy's own text.
# ---------------------------------------------------------------------------

def classify_trophy(trophy):
    """Return (archetype, chapter or None, is_final_mission)."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return "platinum", None, False
    if "100% completion" in desc or "100% completion" in name:
        return "completion", None, False
    if "gone fishing" in desc:
        return "prologue", 0, False
    if "final mission for" in desc or "strangers" in desc:
        return "side", None, False
    if "final mission" in desc:
        return "story", 99, True
    for ch in (1, 2, 3, 4, 5, 6):
        if "chapter %d" % ch in desc:
            return "story", ch, False
    if "chapter" in desc:
        return "story", None, False
    grind_markers = ("win ", "earn $", "spend $",
                     "purchase all", "register", "explore all",
                     "meditate at every", "discover", "rob ")
    if any(m in text for m in grind_markers):
        return "grind", None, False
    if desc.startswith("complete") and any(ch.isdigit() for ch in desc):
        return "grind", None, False
    if desc.startswith("complete") and trophy.get("secret"):
        return "story", None, False
    return "misc", None, False


def ceiling_for(arch, chapter, last_rate):
    """Week-one rate ceiling for this trophy, never below its last rate."""
    cap = CHAPTER_CEILING.get(chapter, RATE_CEILING.get(arch, 0.45))
    return max(cap, last_rate / 100.0)


# ---------------------------------------------------------------------------
# History: distinct Exophase pages only.
# ---------------------------------------------------------------------------

def distinct_pages(history):
    """Collapse consecutive identical reads to one page each (first seen)."""
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
    return parse_ts(read.get("at")) or parse_ts(read.get("slot"))


def log_time(t, release_start):
    """x-coordinate: ln(1 + hours since the release day began)."""
    if t is None or release_start is None:
        return 0.0
    hours = (t - release_start).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


# ---------------------------------------------------------------------------
# One case, end to end.
# ---------------------------------------------------------------------------

def forecast_case(case):
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
            release_start = first.replace(hour=0, minute=0, second=0,
                                          microsecond=0)
    if target is None and checkpoint is not None:
        target = checkpoint + timedelta(minutes=case.get("horizon_minutes") or 0)
    x_target = log_time(target, release_start)

    kinds = [classify_trophy(t) for t in trophies]

    # Last observed rate per trophy, scanning backwards over nulls.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rs = read.get("rates") or []
            if i < len(rs) and rs[i] is not None:
                rate = rs[i]
                break
        last_rates.append(rate if rate is not None else 1.0)

    pages = distinct_pages(history)
    medians = [None] * len(trophies)
    sigmas = [None] * len(trophies)

    if len(pages) <= 1:
        # No refresh observed: expect one before the target.  Players jump
        # by the documented factor; earners grow modestly, fastest for the
        # trophies new players reach within hours.  Narrow spread: the
        # refresh's size is known, the earners growth is the uncertainty.
        for i in range(len(trophies)):
            arch, chapter, _fm = kinds[i]
            mult = FLAT_EARNERS.get(arch, 1.2)
            if arch == "story" and chapter in FLAT_CHAPTER:
                mult = FLAT_CHAPTER[chapter]
            rate = last_rates[i] * mult / REFRESH_PLAYERS
            medians[i] = clamp_rate(rate)
            sigmas[i] = math.sqrt(FLAT_SIGMA ** 2
                                  + (RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2)
    else:
        xs = [log_time(read_time(p), release_start) for p in pages]
        for j in range(1, len(xs)):
            if xs[j] <= xs[j - 1]:
                xs[j] = xs[j - 1] + 1e-6
        x_recent = None
        if checkpoint is not None:
            x_recent = log_time(checkpoint - timedelta(hours=RECENT_HOURS),
                                release_start)

        # Players: saturating fit on the distinct pages, floored at the
        # last count (tracking only accumulates).
        ys_players = [math.log(max(1, p["players_tracked"])) for p in pages]
        log_players, players_rms, _pt = series_forecast(
            xs, ys_players, x_target, x_recent)
        log_players = max(log_players, ys_players[-1])
        # Window disagreement, for the spread: full fit vs recent line.
        _i, _s, _r, _t = fit_line(xs, ys_players)
        full_only = line_at(_i, _s, xs[-1], x_target)
        players_disagree = abs(full_only - log_players)

        # Pass 1: unshrunk earners forecasts, for the archetype growth norms.
        raw = []
        for i in range(len(trophies)):
            series = []
            for x, page in zip(xs, pages):
                rates = page.get("rates") or []
                if i >= len(rates) or rates[i] is None:
                    continue
                earners = max(0.5, rates[i] / 100.0 * page["players_tracked"])
                series.append((x, math.log(earners)))
            if not series:
                raw.append(None)
                continue
            sxs = [s[0] for s in series]
            sys = [s[1] for s in series]
            if len(series) == 1:
                raw.append((sys[0], 0.3, 0.0, sys[0]))
                continue
            pred, rms, t = series_forecast(sxs, sys, x_target, x_recent)
            # Earners are cumulative: never forecast below the last count.
            pred = max(pred, sys[-1] - 0.02)
            raw.append((pred, rms, t, sys[-1]))

        growths = {}
        all_growth = []
        for i, out in enumerate(raw):
            if out is None:
                continue
            growth = out[0] - out[3]
            all_growth.append(growth)
            growths.setdefault(kinds[i][0], []).append(growth)
        arch_growth = {a: mid(g) for a, g in growths.items()}
        global_growth = mid(all_growth) if all_growth else 0.0

        # Pass 2: shrink each fit toward its archetype's typical growth,
        # cap by the archetype ceiling, then rate = earners / players.
        rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
        for i, out in enumerate(raw):
            arch, chapter, _fm = kinds[i]
            if out is None:
                medians[i] = clamp_rate(last_rates[i])
                sigmas[i] = max(SPREAD_FLOOR, rw)
                continue
            pred, rms, t, last_y = out
            w = (t * t) / (t * t + SHRINK_K) if t else 0.0
            norm = arch_growth.get(arch, global_growth)
            if len(growths.get(arch, [])) <= 1:
                norm = global_growth
            growth = pred - last_y
            shrunk = w * growth + (1.0 - w) * norm
            log_earners = last_y + shrunk
            # Archetype ceiling on the rate at the target.
            cap_rate = ceiling_for(arch, chapter, last_rates[i])
            log_cap = math.log(max(0.5, cap_rate * math.exp(log_players)))
            log_earners = min(log_earners, log_cap)
            rate_counts = clamp_rate(100.0 * math.exp(log_earners - log_players))

            # Second opinion: a damped line of the rate's own logit trend
            # over the last day.  The counts ratio carries most of the
            # weight when its fits are stable; the direct trend gets more
            # say when the earners fit is noisy.
            rate_direct = direct_rate(pages, xs, i, x_target, x_recent,
                                      last_rates[i])
            w_direct = 0.2 + 0.3 * min(1.0, rms / 0.5)
            rate = math.exp((1.0 - w_direct) * math.log(rate_counts)
                            + w_direct * math.log(clamp_rate(rate_direct)))
            medians[i] = clamp_rate(rate)
            base = math.sqrt(rw * rw + min(rms, 1.0) ** 2
                             + min(players_disagree, 1.0) ** 2)
            sigmas[i] = min(max(base, SPREAD_FLOOR) * ARCH_SPREAD.get(arch, 1.0),
                            SPREAD_CAP)

    apply_structure(trophies, kinds, last_rates, medians)

    # --- quantiles, with a strict format check on every row ---
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
        qs = []
        for z in Z:
            q = rate_of_logit(centre + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    forecast = {"trophies": rows}
    validate(forecast, trophies)
    return forecast


def direct_rate(pages, xs, idx, x_target, x_recent, last_rate):
    """Second opinion on the rate: a damped line of its logit trend.

    Fits logit(rate) against log-time over the recent window and reads it
    at the target.  Unlike the earners/players ratio it never compounds
    two forecasts' errors, and the logit scale saturates the way a rate
    does.  Falls back to the last rate when the recent window holds too
    few distinct pages to fit.
    """
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
    """Enforce the structure the game itself imposes on the medians."""
    # Near-universal trophies stay near universal: Exophase tracks a
    # player for the game once they hold a trophy, and the prologue is
    # the first thing played, so newcomers arrive already holding it --
    # launch night's refresh proved it, growing the tracked players
    # sixfold while the rate stayed at 100%.
    for i in range(len(trophies)):
        if last_rates[i] >= 97.0:
            medians[i] = 99.95

    # Story funnel: prologue >= chapter 1 >= ... >= final mission.
    chain = []
    for i in range(len(trophies)):
        arch, chapter, _fm = kinds[i]
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

    final_idx = completion_idx = platinum_idx = None
    for i in range(len(trophies)):
        arch, _ch, is_final = kinds[i]
        if arch == "story" and is_final:
            final_idx = i
        elif arch == "completion":
            completion_idx = i
        elif arch == "platinum":
            platinum_idx = i

    # 100% completion rides on the final mission: blend the model with
    # (final mission x historical ratio x 1.25), capped at the final mission.
    if completion_idx is not None and final_idx is not None:
        fm_last = last_rates[final_idx]
        ratio = last_rates[completion_idx] / fm_last if fm_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = min(medians[final_idx] * ratio * 1.25, medians[final_idx])
        blended = math.sqrt(max(medians[completion_idx], 0.01)
                            * max(anchor, 0.01))
        medians[completion_idx] = min(blended, medians[final_idx])

    # The platinum rides on completion: same geometric blend, then capped
    # at the smallest other median (it needs every other trophy).
    if platinum_idx is not None and completion_idx is not None:
        comp_last = last_rates[completion_idx]
        ratio = last_rates[platinum_idx] / comp_last if comp_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = medians[completion_idx] * ratio
        medians[platinum_idx] = math.sqrt(max(medians[platinum_idx], 0.01)
                                          * max(anchor, 0.01))
    if platinum_idx is not None:
        others = [medians[i] for i in range(len(trophies)) if i != platinum_idx]
        if others:
            medians[platinum_idx] = min(medians[platinum_idx], min(others))


def validate(forecast, trophies):
    """Raise unless the forecast is exactly the shape the rules demand."""
    if set(forecast.keys()) != {"trophies"}:
        raise ValueError("top level must hold only 'trophies'")
    rows = forecast["trophies"]
    if len(rows) != len(trophies):
        raise ValueError("row count does not match the trophy count")
    want = {t["id"] for t in trophies}
    seen = set()
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            raise ValueError("row keys must be exactly 'id' and 'q'")
        if row["id"] in seen or row["id"] not in want:
            raise ValueError("duplicate or unknown trophy id")
        seen.add(row["id"])
        qs = row["q"]
        if len(qs) != 5:
            raise ValueError("each row needs five quantiles")
        prev = -1.0
        for q in qs:
            if not isinstance(q, (int, float)) or not math.isfinite(q):
                raise ValueError("quantile is not a finite number")
            if q < 0 or q > 100:
                raise ValueError("quantile outside 0..100")
            if q < prev:
                raise ValueError("quantiles must be non-decreasing")
            prev = q
    payload = json.dumps(forecast, separators=(",", ":")) + "\n"
    if len(payload.encode("utf-8")) > 65536:
        raise ValueError("forecast exceeds 65,536 bytes")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
