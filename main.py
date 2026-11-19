"""Forecast GTA VI's PS5 trophy earn rates on Exophase a week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01: read one case file
(/input/<id>.in, schema makefun/gta6-trophies-case/v1) and write
/output/<id>.out with five quantiles (5/25/50/75/95%) of every trophy's
earn rate at the case's target, scored by pinball loss on the logit scale.

Model, in one paragraph: an earn rate is earners divided by players
tracked, and both counts grow over launch week, so the program forecasts
the two counts separately and divides.  Both are fitted in log-log time
(log count against log hours since the release day began), because launch
growth is multiplicative and slows in clock time.  Exophase refreshes a
game's page on its own schedule, hours apart, so consecutive identical
reads are one page, not many observations: the fits run only on distinct
pages.  Players get a two-window fit (recent day plus full history,
averaged, slopes damped); each trophy's earners get the same two windows,
the full history as a line blended with a quadratic, capped against
straying far from the line, averaged with the last day's damped line --
the recent window matters most, because a saturating trophy's early
log-log slope is far steeper than its slope at the deadline.  The ratio
forecast is then blended, in log rate, with a second opinion: a damped
line of the rate's own logit trend over the last day, which never
compounds two forecasts' errors and saturates the way a rate does.
Noisy fits shrink toward their archetype's typical growth, with the
shrinkage weight set by the fit's t-statistic.  Structure the game itself imposes is then enforced: the
story funnel runs downhill, 100% completion rides on the final mission,
the platinum rides on completion and can never exceed the smallest other
median, and a trophy essentially every tracked player holds stays near
universal.  The spread combines a random walk over the horizon, the fit
residual and the players windows' disagreement, with a floor and an
archetype multiplier.  A history with no refresh at all (the practice
cases) gets a dilution haircut instead of a fit: a refresh multiplies the
tracked players far faster than the earners, so rates fall.

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

# --- tuning constants (chosen by backtesting on simulated launch weeks,
# never on the practice answers, which the kit does not ship) ---
SLOPE_DAMP = 0.8        # extrapolation damps every fitted slope by this
QUAD_WEIGHT = 0.5       # weight of the quadratic in the line/quadratic blend
QUAD_CAP = 0.5          # log units the blend may stray from the damped line
SHRINK_K = 4.0          # t^2 / (t^2 + K) is a fit's weight against its archetype
EARNERS_RECENT_W = 0.8  # weight of the last day's line in the earners forecast
RATE_BLEND_W = 0.7      # weight of the direct rate trend in the final rate
# Fixed quantiles for a near-universal trophy (see the median step below):
# most of the mass at 100%, a left tail for dilution.
NEAR_UNIV_Q = [98.5, 99.7, 99.95, 99.99, 99.99]
RW_SIGMA = 0.5          # random-walk spread, logit units per sqrt(day)
SPREAD_FLOOR = 0.25     # minimum logit spread, whatever the fits say
SPREAD_CAP = 2.0        # maximum logit spread
FLAT_SIGMA = 0.7        # spread when the history shows no refresh at all
FLAT_PLAYERS_MULT = 6.0  # launch night's observed refresh: 638 -> 3,868 tracked
RECENT_HOURS = 24.0     # the recent window for the players, earners and rate fits

ARCH_SPREAD = {
    "prologue": 0.9,
    "story": 0.9,
    "side": 1.0,
    "grind": 1.05,
    "misc": 1.0,
    "completion": 1.1,
    "platinum": 1.1,
}

# Earners multiplier per archetype when the history is flat (one refresh
# expected before the target).  Launch night's refresh grew non-prologue
# earners 1.34x overall while players grew 6.06x; early story trophies grow
# fastest because new players reach them within hours.
FLAT_EARNERS_MULT = {
    "prologue": 6.0,
    "story": 1.5,
    "side": 1.3,
    "grind": 1.2,
    "misc": 1.3,
    "completion": 1.1,
    "platinum": 1.0,
}
FLAT_CHAPTER_MULT = {1: 3.0, 2: 2.0, 3: 1.6, 4: 1.4}


def clamp_pct(x):
    """A rate clamped into the scorer's logit domain."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero, as a JSON-friendly number."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparseable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


# --------------------------------------------------------------------------
# Small least-squares fits (closed form; no third-party code needed).
# --------------------------------------------------------------------------

def ols_line(xs, ys):
    """Least-squares line. Returns (intercept, slope, resid_rms, t_slope).

    t_slope is the slope's t-statistic (0 when it cannot be estimated).
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    rss = sum(r * r for r in resid)
    rms = math.sqrt(rss / n)
    if n <= 2:
        return intercept, slope, rms, 0.0
    se = math.sqrt(rss / (n - 2) / sxx)
    t = slope / se if se > 0 else 0.0
    return intercept, slope, rms, t


def solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(3):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][3] for i in range(3)]


def ols_quad(xs, ys):
    """Least-squares quadratic. Returns (c0, c1, c2) or None (too few points)."""
    n = len(xs)
    if n < 3:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]
    rhs = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    mat = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    return solve3(mat, rhs)


def eval_quad(coef, x):
    return coef[0] + coef[1] * x + coef[2] * x * x


# --------------------------------------------------------------------------
# Trophy archetypes, classified from the trophy's own text.
# --------------------------------------------------------------------------

def classify(trophy):
    """Return (archetype, chapter or None, is_final_mission)."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return "platinum", None, False
    if "100% completion" in desc:
        return "completion", None, False
    if "gone fishing" in desc:
        return "prologue", 0, False
    if "chapter" in desc:
        for ch in (1, 2, 3, 4, 5, 6):
            if "chapter %d" % ch in desc:
                return "story", ch, False
        return "story", None, False
    if "final mission for" in desc or "strangers" in desc:
        return "side", None, False
    if "final mission" in desc:
        return "story", 99, True
    if desc.startswith("complete") and any(c.isdigit() for c in desc):
        return "grind", None, False
    grind_words = ("win ", "earn $", "spend $", "purchase all", "register",
                   "explore all", "meditate at every", "discover", "rob ")
    if any(w in text for w in grind_words):
        return "grind", None, False
    if desc.startswith("complete") and trophy.get("secret"):
        return "story", None, False
    return "misc", None, False


# --------------------------------------------------------------------------
# History handling: distinct Exophase pages only.
# --------------------------------------------------------------------------

def distinct_pages(history):
    """Collapse consecutive identical reads to one page each (first seen)."""
    pages = []
    last_sig = None
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or not isinstance(rates, list):
            continue
        sig = (players, read.get("trophies_earned"),
               tuple(rates))
        if sig == last_sig:
            continue
        last_sig = sig
        pages.append(read)
    return pages


def log_hours(t, release_start):
    """x-coordinate: ln(1 + hours since the release day began)."""
    if t is None or release_start is None:
        return 0.0
    hours = (t - release_start).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


# --------------------------------------------------------------------------
# Players forecast: two windows, averaged, slopes damped.
# --------------------------------------------------------------------------

def damped_line_forecast(xs, ys, x_target, damp=SLOPE_DAMP):
    """Line fit evaluated at x_target with its slope damped past the data."""
    intercept, slope, _rms, _t = ols_line(xs, ys)
    anchor = intercept + slope * xs[-1]
    return anchor + damp * slope * (x_target - xs[-1])


def forecast_players(pages, xs, checkpoint, x_target):
    """Return (log players at target, window disagreement in log units)."""
    ys = [math.log(max(1, p["players_tracked"])) for p in pages]
    full = damped_line_forecast(xs, ys, x_target)
    cutoff = checkpoint.timestamp() - RECENT_HOURS * 3600 if checkpoint else None
    recent_idx = [i for i, p in enumerate(pages)
                  if cutoff is None or (parse_time(p.get("slot") or p.get("at")) or checkpoint).timestamp() >= cutoff]
    if len(recent_idx) >= 3:
        rxs = [xs[i] for i in recent_idx]
        rys = [ys[i] for i in recent_idx]
        recent = damped_line_forecast(rxs, rys, x_target)
    elif len(xs) >= 3:
        recent = damped_line_forecast(xs[-3:], ys[-3:], x_target)
    else:
        recent = full
    pred = 0.5 * (full + recent)
    # Players never shrink in the forecast: tracking only accumulates.
    pred = max(pred, ys[-1])
    return pred, abs(full - recent)


# --------------------------------------------------------------------------
# Earners forecast per trophy.
# --------------------------------------------------------------------------

def fit_earners(xs, ys, x_target):
    """Line/quadratic blend for one earners series.

    Returns (log earners at target, line residual rms, slope t-statistic).
    """
    intercept, slope, rms, t = ols_line(xs, ys)
    line_pred = intercept + slope * xs[-1] + SLOPE_DAMP * slope * (x_target - xs[-1])
    quad = ols_quad(xs, ys)
    if quad is None:
        return line_pred, rms, t
    blend = (1.0 - QUAD_WEIGHT) * line_pred + QUAD_WEIGHT * eval_quad(quad, x_target)
    dev = max(-QUAD_CAP, min(QUAD_CAP, blend - line_pred))
    return line_pred + dev, rms, t


def forecast_earners(pages, xs, idx, x_target, checkpoint):
    """Forecast one trophy's log earners at the target.

    Two windows, like the players forecast: the full history's
    line/quadratic blend, averaged with the last day's damped line.  The
    full window alone is badly biased for a trophy whose rate saturates:
    its early log-log slope (rate rising from zero) is far steeper than
    the slope at the deadline, so a full-window line overshoots and its
    quadratic whipsaws.  The last day reads the current slope instead --
    accelerating series keep accelerating, decelerating ones slow -- and
    averaging the two windows trades a little variance for much less bias.
    A last-day quadratic is not used: over a one-day window it is too
    unstable to extrapolate four days out.

    Returns (log earners, residual rms, t-statistic, last log earners).
    """
    series = []
    for x, page in zip(xs, pages):
        rates = page.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        earners = max(0.5, rates[idx] / 100.0 * page["players_tracked"])
        series.append((x, math.log(earners), page))
    if not series:
        return None
    xs_s = [s[0] for s in series]
    ys_s = [s[1] for s in series]
    last_y = ys_s[-1]
    if len(series) == 1:
        return last_y, 0.3, 0.0, last_y
    pred, rms, t = fit_earners(xs_s, ys_s, x_target)
    # Second window: the last day's damped line, averaged with the full
    # blend whatever the direction of travel (see the docstring).
    if len(series) >= 4 and checkpoint is not None:
        cutoff = checkpoint.timestamp() - RECENT_HOURS * 3600
        day = [s for s in series
               if (parse_time(s[2].get("slot") or s[2].get("at")) or checkpoint).timestamp() >= cutoff]
        if len(day) >= 3:
            dxs = [s[0] for s in day]
            dys = [s[1] for s in day]
            day_pred = damped_line_forecast(dxs, dys, x_target)
            pred = ((1.0 - EARNERS_RECENT_W) * pred
                    + EARNERS_RECENT_W * day_pred)
            # The spread's residual is the recent fit's, not the full
            # window's: the forecast leans on the recent window, and the
            # full line's residual mostly measures how curved the early
            # history was, not how uncertain the forecast is.
            _i, _s, rms, _t = ols_line(dxs, dys)
    return pred, rms, t, last_y


def direct_rate_forecast(pages, xs, idx, x_target, checkpoint, last_rate):
    """A second opinion on the rate: a damped line of the logit's trend.

    Fits logit(rate) against log hours over the last day's distinct pages
    and reads it at the target.  Unlike the earners/players ratio it never
    compounds two forecasts' errors, and the logit scale saturates the way
    a rate does, so it is the steadier read for a trophy whose rate is
    flattening near its ceiling.  Falls back to the last rate when the
    last day holds too few distinct pages to fit.

    The damping is adaptive: a trophy whose recent slope already runs
    below its full-history slope is decelerating, so its slope is damped
    toward 0.8; one whose recent slope matches or beats its history is
    steady or accelerating, so its slope is read undamped.  That keeps
    saturating story trophies from overshooting while letting late-game
    trophies (grinds, completion, platinum) keep climbing.
    """
    pts = []
    all_pts = []
    cutoff = checkpoint.timestamp() - RECENT_HOURS * 3600 if checkpoint else None
    for x, page in zip(xs, pages):
        rates = page.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        all_pts.append((x, logit(rates[idx])))
        if cutoff is not None:
            when = parse_time(page.get("slot") or page.get("at")) or checkpoint
            if when.timestamp() < cutoff:
                continue
        pts.append((x, logit(rates[idx])))
    if len(pts) < 3:
        return last_rate
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    _i, recent_slope, _r, _t = ols_line(px, py)
    damp = SLOPE_DAMP
    if len(all_pts) >= 3:
        ax = [p[0] for p in all_pts]
        ay = [p[1] for p in all_pts]
        _i2, full_slope, _r2, _t2 = ols_line(ax, ay)
        if full_slope > 1e-9:
            ratio = recent_slope / full_slope
        else:
            ratio = 1.0 if recent_slope > 0 else 0.0
        damp = SLOPE_DAMP + (1.0 - SLOPE_DAMP) * min(1.0, max(0.0, ratio))
    return clamp_pct(inv_logit(damped_line_forecast(px, py, x_target, damp)))


# --------------------------------------------------------------------------
# One case, end to end.
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    game = case.get("game") or {}
    release_start = parse_time((game.get("release_date") or "") + "T00:00:00Z")
    if release_start is None and history:
        first = parse_time(history[0].get("slot") or history[0].get("at"))
        release_start = first.replace(hour=0, minute=0, second=0, microsecond=0) if first else None
    x_target = log_hours(target, release_start)
    if target is None and checkpoint is not None:
        x_target = log_hours(
            checkpoint + timedelta(minutes=case.get("horizon_minutes") or 0),
            release_start)

    kinds = [classify(t) for t in trophies]
    last_rates = []
    if history:
        last_read = history[-1]
        rates = last_read.get("rates") or []
        for i in range(len(trophies)):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                for read in reversed(history[:-1]):
                    rs = read.get("rates") or []
                    if i < len(rs) and rs[i] is not None:
                        rate = rs[i]
                        break
            last_rates.append(rate if rate is not None else 1.0)
    else:
        last_rates = [1.0] * len(trophies)

    pages = distinct_pages(history)
    flat = len(pages) <= 1

    medians = [None] * len(trophies)
    sigmas = [None] * len(trophies)

    if flat:
        # No refresh observed: expect one before the target.  Players jump
        # (launch night: x6.06); earners grow much less, fastest for the
        # trophies new players reach within hours.  Wide spread: the
        # refresh's size is a guess.
        horizon_scale = 1.0 + 0.3 * max(0.0, horizon_days)
        players_mult = FLAT_PLAYERS_MULT
        if horizon_days > 1.0:
            players_mult = FLAT_PLAYERS_MULT * math.sqrt(horizon_days)
        for i, trophy in enumerate(trophies):
            arch, chapter, _fm = kinds[i]
            mult = FLAT_EARNERS_MULT.get(arch, 1.3)
            if arch == "story" and chapter in FLAT_CHAPTER_MULT:
                mult = FLAT_CHAPTER_MULT[chapter]
            if arch == "story" and _fm:
                mult = 1.3
            rate = last_rates[i] * mult * horizon_scale / players_mult
            medians[i] = clamp_pct(rate)
            sigmas[i] = max(FLAT_SIGMA, RW_SIGMA * math.sqrt(max(horizon_days, 0.0)))
    else:
        xs = [log_hours(parse_time(p.get("slot") or p.get("at")), release_start)
              for p in pages]
        # Guard against non-increasing x (duplicate slots after parsing).
        for j in range(1, len(xs)):
            if xs[j] <= xs[j - 1]:
                xs[j] = xs[j - 1] + 1e-6
        log_players, players_disagree = forecast_players(
            pages, xs, checkpoint, x_target)

        # Pass 1: unshrunk earners forecasts, for the archetype growth norms.
        raw = []
        for i in range(len(trophies)):
            out = forecast_earners(pages, xs, i, x_target, checkpoint)
            if out is None:
                raw.append(None)
            else:
                raw.append(out)
        growths = {}
        all_growth = []
        for i, out in enumerate(raw):
            if out is None:
                continue
            growth = out[0] - out[3]
            all_growth.append(growth)
            growths.setdefault(kinds[i][0], []).append(growth)
        arch_growth = {a: median(g) for a, g in growths.items()}
        global_growth = median(all_growth) if all_growth else 0.0

        # Pass 2: shrink each fit toward its archetype's typical growth,
        # then rate = earners / players.
        rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
        for i, out in enumerate(raw):
            arch = kinds[i][0]
            if out is None:
                medians[i] = clamp_pct(last_rates[i])
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
            rate_ep = 100.0 * math.exp(log_earners - log_players)
            rate_ep = clamp_pct(rate_ep)
            # Blend the ratio forecast with the direct rate trend, in log
            # rate: the ratio sees the counts, the trend sees the ceiling.
            rate_dr = direct_rate_forecast(pages, xs, i, x_target,
                                           checkpoint, last_rates[i])
            rate = math.exp((1.0 - RATE_BLEND_W) * math.log(rate_ep)
                            + RATE_BLEND_W * math.log(clamp_pct(rate_dr)))
            medians[i] = clamp_pct(rate)
            base = math.sqrt(rw * rw + min(rms, 1.0) ** 2
                             + min(players_disagree, 1.0) ** 2)
            sigmas[i] = max(base, SPREAD_FLOOR) * ARCH_SPREAD.get(arch, 1.0)
            sigmas[i] = min(sigmas[i], SPREAD_CAP)

    # --- structure the game imposes, applied to the medians ---
    # Near-universal trophies stay near universal.  A trophy every tracked
    # player holds is the prologue mission: Exophase tracks a player for
    # the game once they hold a trophy, and the prologue is the first thing
    # played, so newcomers arrive already holding it -- launch night's
    # refresh proved it, growing the tracked players sixfold while the
    # rate stayed at 100%.  The logit scale punishes an undershoot near
    # 100% far more than an overshoot, so the forecast sits high: median
    # 99.95, with most of the mass at 100% and a left tail (q05 98.5)
    # for the chance that dilution finally bites.
    near_univ = set()
    for i in range(len(trophies)):
        if last_rates[i] >= 97.0:
            near_univ.add(i)
            medians[i] = 99.95

    # Story funnel: prologue >= chapter 1 >= ... >= final mission.
    chain = []
    for i, t in enumerate(trophies):
        arch, chapter, is_final = kinds[i]
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

    # 100% completion rides on the final mission: blend the model with
    # (final mission x historical ratio x 1.25), capped at the final mission.
    final_idx = None
    completion_idx = None
    platinum_idx = None
    for i in range(len(trophies)):
        arch, _ch, is_final = kinds[i]
        if arch == "story" and is_final:
            final_idx = i
        elif arch == "completion":
            completion_idx = i
        elif arch == "platinum":
            platinum_idx = i
    if completion_idx is not None and final_idx is not None:
        fm_last = last_rates[final_idx]
        ratio = (last_rates[completion_idx] / fm_last) if fm_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = medians[final_idx] * ratio * 1.25
        anchor = min(anchor, medians[final_idx])
        medians[completion_idx] = math.sqrt(
            max(medians[completion_idx], 0.01) * max(anchor, 0.01))
        medians[completion_idx] = min(medians[completion_idx], medians[final_idx])
    if platinum_idx is not None and completion_idx is not None:
        comp_last = last_rates[completion_idx]
        ratio = (last_rates[platinum_idx] / comp_last) if comp_last > 0 else 0.5
        ratio = min(1.0, max(0.0, ratio))
        anchor = medians[completion_idx] * ratio
        medians[platinum_idx] = math.sqrt(
            max(medians[platinum_idx], 0.01) * max(anchor, 0.01))
    if platinum_idx is not None:
        others = [medians[i] for i in range(len(trophies)) if i != platinum_idx]
        if others:
            medians[platinum_idx] = min(medians[platinum_idx], min(others))

    # --- quantiles, with a strict format check on every row ---
    rows = []
    for i, trophy in enumerate(trophies):
        med = clamp_pct(medians[i])
        sigma = sigmas[i]
        if sigma is None or not math.isfinite(sigma):
            sigma = max(SPREAD_FLOOR, RW_SIGMA * math.sqrt(max(horizon_days, 0.0)))
        if i in near_univ:
            rows.append({"id": trophy["id"], "q": list(NEAR_UNIV_Q)})
            continue
        centre = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(centre + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        # Rounding must never break the non-decreasing order.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    forecast = {"trophies": rows}
    validate(forecast, trophies)
    return forecast


def validate(forecast, trophies):
    """Raise unless the forecast is exactly the shape the rules demand."""
    if set(forecast.keys()) != {"trophies"}:
        raise ValueError("top level must hold only 'trophies'")
    rows = forecast["trophies"]
    if len(rows) != len(trophies):
        raise ValueError("row count does not match the trophy count")
    seen = set()
    want = {t["id"] for t in trophies}
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
