"""Transparent 0-100 setup quality score.

The score is a *ranking* of how well a setup matches the configured
break-and-retest criteria. It is NOT a probability of winning: an 80/100
setup does not have an 80% chance of success.

Each factor produces a raw sub-score in [0, 1] plus a plain-English
explanation. ``points = weight x raw``. Missing factors are handled by an
explicit policy (never silently awarded full points):

* ``renormalize`` (default): total = sum(points of available factors) /
  sum(weights of available factors) x 100. The setup is rejected if the
  available weight is below ``min_available_weight`` or any *essential*
  factor is missing.
* ``zero``: missing factors score 0 points (most conservative).

Factor definitions (all deterministic):

* htf_alignment - per enabled HTF: aligned=1, neutral=0.5, opposed=0; mean.
* zone_quality - zone strength (0..1) frozen at breakout time.
* breakout_quality - mean of body ratio, close position in range,
  range-size suitability (1 inside 0.8-2.0 ATR, tapering outside) and
  extension (1 at the zone edge, 0.5 at the max extension).
* retest_quality - mean of rejection-wick ratio (saturating at 0.6),
  retest precision (1 inside the zone, 0.8 front-run, penalised for deep
  pierces), timing (1 for 2-8 bars, lower when immediate or late) and
  confirmation close position.
* vwap_alignment - 1 on the correct side of VWAP (0.6 if more than
  ``vwap_stretch_atr`` ATR away, default 4), else 0.
* relative_volume - linear from 0 at 0.8x to 1 at 1.5x average volume.
* reward_risk - 0 below the minimum R:R, then 0.5 -> 1 as R:R rises from
  the minimum to minimum + 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core.config import SCORE_FACTORS, ScoringSettings


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class FactorResult:
    factor: str
    label: str
    weight: float
    raw: float | None  # None = unavailable
    points: float
    available: bool
    explanation: str


@dataclass
class ScoreResult:
    total: float
    policy: str
    available_weight: float
    components: list[FactorResult]
    passed: bool
    min_score: float
    missing: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # blocking (why it failed)
    notes: list[str] = field(default_factory=list)  # informational

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    def component(self, name: str) -> FactorResult | None:
        for c in self.components:
            if c.factor == name:
                return c
        return None


# ---------------------------------------------------------------------------
# Factor helpers
# ---------------------------------------------------------------------------


def factor_htf(trends: dict[str, str | None], direction: str) -> tuple[float | None, str]:
    want = "up" if direction == "long" else "down"
    against = "down" if direction == "long" else "up"
    vals, parts = [], []
    for tf, tr in trends.items():
        if tr is None:
            parts.append(f"{tf}: unavailable")
            continue
        v = 1.0 if tr == want else (0.0 if tr == against else 0.5)
        vals.append(v)
        parts.append(f"{tf}: {tr}")
    if not vals:
        return None, "No higher-timeframe trend available (" + ", ".join(parts or ["all disabled"]) + ")."
    raw = sum(vals) / len(vals)
    return raw, f"{raw:.0%} aligned with a {direction} trade (" + ", ".join(parts) + ")."


def factor_zone(
    strength: float | None, parts: dict | None, reactions: int | None
) -> tuple[float | None, str]:
    if strength is None:
        return None, "Zone strength unavailable."
    desc = ", ".join(f"{k} {v:.2f}" for k, v in (parts or {}).items())
    return clip01(strength), f"Zone strength {strength:.2f} from {reactions} reaction(s) [{desc}]."


def factor_breakout(metrics: dict, max_extension_atr: float) -> tuple[float | None, str]:
    br = metrics.get("body_ratio")
    cp = metrics.get("close_position")
    ra = metrics.get("range_atr")
    ex = metrics.get("close_distance_atr")
    if br is None or cp is None or ra is None or ex is None:
        return None, "Breakout metrics unavailable."
    if 0.8 <= ra <= 2.0:
        size = 1.0
    elif ra < 0.8:
        size = clip01(ra / 0.8)
    else:
        size = clip01(1.0 - (ra - 2.0) / 2.0)
    ext = clip01(1.0 - 0.5 * clip01(ex / max_extension_atr))
    raw = (clip01(br) + clip01(cp) + size + ext) / 4.0
    return raw, (
        f"Body {br:.0%} of range, close in top {cp:.0%} of range, size {ra:.2f} ATR, extension {ex:.2f} ATR."
    )


def factor_retest(
    wick_ratio: float | None,
    precision: float | None,
    bars_to_retest: int | None,
    confirm_close_pos: float | None,
) -> tuple[float | None, str]:
    if wick_ratio is None or precision is None or bars_to_retest is None or confirm_close_pos is None:
        return None, "Retest metrics unavailable."
    wick = clip01(wick_ratio / 0.6)
    if 2 <= bars_to_retest <= 8:
        timing = 1.0
    elif bars_to_retest < 2:
        timing = 0.7
    else:
        timing = clip01(1.0 - 0.05 * (bars_to_retest - 8))
        timing = max(timing, 0.5)
    raw = (wick + clip01(precision) + timing + clip01(confirm_close_pos)) / 4.0
    return raw, (
        f"Rejection wick {wick_ratio:.0%}, precision {precision:.2f}, retest after {bars_to_retest} bar(s), "
        f"confirmation close in top {confirm_close_pos:.0%} of range."
    )


def factor_vwap(distance_atr: float | None, stretch_atr: float = 4.0) -> tuple[float | None, str]:
    if distance_atr is None:
        return None, "VWAP unavailable (volume missing or unreliable)."
    if distance_atr <= 0:
        return 0.0, f"Price on the wrong side of VWAP ({distance_atr:+.2f} ATR)."
    if distance_atr > stretch_atr:
        return 0.6, f"Correct side of VWAP but stretched ({distance_atr:+.2f} ATR)."
    return 1.0, f"Correct side of VWAP ({distance_atr:+.2f} ATR)."


def factor_rvol(rvol: float | None) -> tuple[float | None, str]:
    if rvol is None:
        return None, "Relative volume unavailable (volume missing or unreliable)."
    return clip01((rvol - 0.8) / 0.7), f"Breakout volume {rvol:.2f}x recent average."


def factor_rr(rr: float | None, min_rr: float) -> tuple[float | None, str]:
    if rr is None:
        return None, "No valid trade plan - reward/risk unavailable."
    if rr < min_rr:
        return 0.0, f"Planned R:R {rr:.2f} is below the minimum {min_rr:.2f}."
    return 0.5 + 0.5 * clip01((rr - min_rr) / 2.0), f"Planned R:R {rr:.2f} (minimum {min_rr:.2f})."


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_score(factors: dict[str, tuple[float | None, str]], cfg: ScoringSettings) -> ScoreResult:
    components: list[FactorResult] = []
    missing: list[str] = []
    reasons: list[str] = []
    notes: list[str] = []
    avail_w = 0.0
    pts_total = 0.0
    for name, label in SCORE_FACTORS.items():
        w = float(cfg.weights[name])
        raw, expl = factors.get(name, (None, "Not computed."))
        if raw is None:
            missing.append(name)
            components.append(FactorResult(name, label, w, None, 0.0, False, expl))
            continue
        raw = clip01(raw)
        pts = w * raw
        avail_w += w
        pts_total += pts
        components.append(FactorResult(name, label, w, round(raw, 4), round(pts, 2), True, expl))

    passed = True
    if cfg.missing_factor_policy == "renormalize":
        total = (pts_total / avail_w * 100.0) if avail_w > 0 else 0.0
        if avail_w < cfg.min_available_weight:
            passed = False
            reasons.append(
                f"Only {avail_w:g} of 100 weight points have data (minimum {cfg.min_available_weight:g})."
            )
    else:
        total = pts_total
    for name in cfg.essential_factors:
        if name in missing:
            passed = False
            reasons.append(f"Essential factor '{SCORE_FACTORS[name]}' is unavailable.")
    total = round(total, 1)
    if total < cfg.min_score:
        passed = False
        reasons.append(f"Score {total:.1f} is below the minimum {cfg.min_score:g}.")
    if missing:
        policy_txt = (
            "re-normalised over available factors"
            if cfg.missing_factor_policy == "renormalize"
            else "missing factors scored as zero"
        )
        notes.append(f"Missing data: {', '.join(SCORE_FACTORS[m] for m in missing)} ({policy_txt}).")
    return ScoreResult(
        total=total,
        policy=cfg.missing_factor_policy,
        available_weight=avail_w,
        components=components,
        passed=passed,
        min_score=cfg.min_score,
        missing=missing,
        reasons=reasons,
        notes=notes,
    )
