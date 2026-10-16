"""End-to-end scanner tests: determinism, historical replay and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import AppSettings
from core.resampling import compute_trend, htf_value_at, resample_session_aware
from core.retest_engine import TERMINAL_STATES, SetupState
from core.scanner import run_scan
from tests.conftest import flat_bars


@pytest.fixture(scope="module")
def full_scan(synthetic_nq):
    from core.instruments import get_instrument

    return run_scan(synthetic_nq, get_instrument("NQ"), AppSettings())


def snapshot(setups, cutoff=None):
    out = {}
    for s in setups:
        if cutoff is not None and not (
            s.is_terminal and s.terminal_ts is not None and s.terminal_ts <= cutoff
        ):
            continue
        out[s.setup_id] = (
            s.state,
            s.breakout_ts,
            s.confirm_ts,
            None if s.score is None else s.score.total,
            None if s.plan is None else (s.plan.entry_price, s.plan.stop, s.plan.target1),
        )
    return out


def test_scan_is_deterministic(synthetic_nq, full_scan, nq):
    again = run_scan(synthetic_nq, nq, AppSettings())
    assert snapshot(again.setups) == snapshot(full_scan.setups)
    assert len(full_scan.setups) > 50
    assert any(s.state == SetupState.QUALIFIED for s in full_scan.setups)


@pytest.mark.parametrize("cut", [2500, 4100, 6000])
def test_truncated_history_reproduces_past_signals(synthetic_nq, full_scan, nq, cut):
    """Every setup that was finished by time T must be identical whether or not later data exists."""
    part = run_scan(synthetic_nq.iloc[:cut], nq, AppSettings())
    cutoff = synthetic_nq.index[cut - 1] + pd.Timedelta(minutes=5)
    assert snapshot(part.setups, cutoff) == snapshot(full_scan.setups, cutoff)


def test_future_perturbation_does_not_change_past(synthetic_nq, full_scan, nq):
    cut = 4000
    df = synthetic_nq.copy()
    rng = np.random.default_rng(0)
    shift = np.cumsum(rng.normal(0, 3, len(df) - cut)).round() * 0.25
    for col in ("open", "high", "low", "close"):
        df.iloc[cut:, df.columns.get_loc(col)] = df[col].to_numpy()[cut:] + shift
    perturbed = run_scan(df, nq, AppSettings())
    cutoff = df.index[cut - 1] + pd.Timedelta(minutes=5)
    assert snapshot(perturbed.setups, cutoff) == snapshot(full_scan.setups, cutoff)


def test_signal_timestamps_never_precede_their_information(full_scan):
    for s in full_scan.setups:
        times = [h.ts for h in s.history[1:]]
        assert times == sorted(times)
        if s.confirm_ts is not None:
            assert s.retest_ts is not None and s.breakout_ts <= s.retest_ts <= s.confirm_ts
        if s.state in TERMINAL_STATES:
            assert s.terminal_ts == s.history[-1].ts


def test_zone_used_by_setup_was_confirmed_before_breakout(full_scan):
    sw = full_scan.swings.set_index("swing_id")
    for s in full_scan.setups:
        # strength/reaction counts are frozen when the setup is created
        assert s.zone["reactions"] >= AppSettings().zones.min_reactions
    assert (sw["confirm_ts"] > sw["pivot_ts"]).all()


def test_htf_bars_only_used_after_close(synthetic_nq):
    df = synthetic_nq.iloc[:3000]
    h1 = resample_session_aware(df, "1h")
    tr = compute_trend(h1, 20, 50)
    probe = h1["close_ts"].iloc[100]
    before = htf_value_at(tr, pd.DatetimeIndex([probe - pd.Timedelta(minutes=5)]), "close_ts")[0]
    at = htf_value_at(tr, pd.DatetimeIndex([probe]), "close_ts")[0]
    assert at == probe and before < probe


def test_qualified_setups_meet_every_rule(full_scan):
    s = AppSettings()
    for x in full_scan.setups:
        if x.state == SetupState.QUALIFIED:
            assert x.score.total >= s.scoring.min_score
            assert x.plan.valid and x.plan.rr >= s.planning.min_rr - 1e-9
            assert all(x.context["qual_checks"].values())


def test_no_duplicate_setup_ids_or_alert_keys(full_scan):
    ids = [s.setup_id for s in full_scan.setups]
    assert len(ids) == len(set(ids))
    keys = [e.dedup_key for e in full_scan.events]
    assert len(keys) == len(set(keys))


def test_multi_contract_series_refused(nq):
    a = flat_bars(300, contract="NQH26")
    b = flat_bars(300, start=a.index[-1] + pd.Timedelta(minutes=5), price=120, contract="NQM26")
    res = run_scan(pd.concat([a, b]), nq, AppSettings())
    assert not res.scanned and "contract" in res.message


def test_malformed_data_disables_scan(synthetic_nq, nq):
    bad = synthetic_nq.iloc[:500].copy()
    bad.iloc[100, bad.columns.get_loc("high")] = bad["low"].iloc[100] - 10
    res = run_scan(bad, nq, AppSettings())
    assert not res.scanned and not res.setups


def test_stale_feed_blocks_new_qualification(synthetic_nq, full_scan, nq):
    q = [s for s in full_scan.setups if s.state == SetupState.QUALIFIED]
    target = q[0]
    cut = target.confirm_idx + 1
    blocked = run_scan(
        synthetic_nq.iloc[:cut],
        nq,
        AppSettings(),
        block_qualification_from=target.confirm_ts,
        block_reason="stale feed",
    )
    s = blocked.setup(target.setup_id)
    assert s.state == SetupState.REJECTED and "stale feed" in s.rejection_reasons
