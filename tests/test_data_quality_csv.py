"""Data validation, freshness and CSV import."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data_quality import assess_freshness, validate_candles
from core.indicators import atr, relative_volume, session_vwap, volume_is_reliable
from core.scanner import split_contract_segments
from data.csv_import import ImportOptions, import_csv
from tests.conftest import T0, flat_bars, make_bars


def codes(report) -> set[str]:
    return {i.code for i in report.issues}


class TestValidation:
    def test_clean_data_ok(self):
        r = validate_candles(flat_bars(50), "5min", 0.25)
        assert r.ok_for_signals and not r.errors
        assert r.last_bar_close == flat_bars(50).index[-1] + pd.Timedelta(minutes=5)

    def test_duplicates_and_unsorted(self):
        df = flat_bars(10)
        dup = pd.concat([df, df.iloc[[3]]]).sort_index()
        r = validate_candles(dup)
        assert "duplicates" in codes(r) and not r.ok_for_signals
        r2 = validate_candles(df.iloc[::-1])
        assert "unsorted" in codes(r2) and not r2.ok_for_signals

    def test_invalid_ohlc(self):
        df = flat_bars(10)
        df.iloc[4, df.columns.get_loc("high")] = 90.0  # high below open/close
        r = validate_candles(df)
        assert "invalid_ohlc" in codes(r) and not r.ok_for_signals

    def test_missing_bars_in_session_flagged(self):
        df = flat_bars(30)
        gapped = df.drop(df.index[10:13])
        r = validate_candles(gapped)
        assert r.missing_bars == 3
        assert r.gap_positions == [10]

    def test_maintenance_halt_and_weekend_not_flagged(self):
        # Friday 16:30 ET -> Sunday 18:00 ET plus the daily 17:00-18:00 halt are not gaps.
        fri = pd.date_range(pd.Timestamp("2026-03-13 16:30", tz="America/New_York"), periods=6, freq="5min")
        sun = pd.date_range(pd.Timestamp("2026-03-15 18:00", tz="America/New_York"), periods=6, freq="5min")
        idx = fri.append(sun).tz_convert("UTC")
        df = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0, "contract": "NQM26"},
            index=idx,
        )
        r = validate_candles(df)
        assert r.missing_bars == 0

    def test_multiple_contracts_blocked_and_split(self):
        a = flat_bars(10, contract="NQH26")
        b = flat_bars(10, start=a.index[-1] + pd.Timedelta(minutes=5), price=120, contract="NQM26")
        both = pd.concat([a, b])
        r = validate_candles(both)
        assert "multiple_contracts" in codes(r) and not r.ok_for_signals
        segs = split_contract_segments(both)
        assert [c for c, _ in segs] == ["NQH26", "NQM26"]

    def test_missing_volume_reported_not_fabricated(self):
        df = flat_bars(40, volume=None)
        ok, _ = volume_is_reliable(df)
        assert not ok
        assert session_vwap(df).isna().all()
        assert relative_volume(df).isna().all()
        r = validate_candles(df)
        assert "volume_unreliable" in codes(r) and r.ok_for_signals

    def test_constant_volume_is_unreliable(self):
        df = flat_bars(40)
        df["volume"] = 1.0
        assert not volume_is_reliable(df)[0]

    def test_atr_does_not_span_contract_roll(self):
        a = flat_bars(20, price=100, contract="NQH26")
        b = flat_bars(20, start=a.index[-1] + pd.Timedelta(minutes=5), price=200, contract="NQM26")
        tr_at_roll = atr(pd.concat([a, b]), 2)
        # A 100-point price gap at the roll must not appear as true range.
        assert tr_at_roll.max() < 5

    def test_off_tick_warning(self):
        df = flat_bars(10, price=100.1)
        assert "off_tick" in codes(validate_candles(df, tick_size=0.25))


class TestFreshness:
    def test_fresh_stale_closed_historical(self):
        last = pd.Timestamp("2026-03-10 15:00", tz="UTC")
        assert assess_freshness(last, last + pd.Timedelta(minutes=3), "5min", "live").status == "fresh"
        st = assess_freshness(last, last + pd.Timedelta(minutes=30), "5min", "live")
        assert st.status == "stale" and st.is_stale and not st.allows_new_signals
        sat = pd.Timestamp("2026-03-14 12:00", tz="America/New_York").tz_convert("UTC")
        fri_close = pd.Timestamp("2026-03-13 17:00", tz="America/New_York").tz_convert("UTC")
        assert assess_freshness(fri_close, sat, "5min", "live").status == "market_closed"
        assert (
            assess_freshness(last, last + pd.Timedelta(days=30), "5min", "historical").status == "historical"
        )
        assert assess_freshness(None, last, "5min", "live").status == "no_data"

    def test_historical_replay_never_reported_fresh(self):
        last = pd.Timestamp("2026-03-10 15:00", tz="UTC")
        st = assess_freshness(last, last, "5min", "historical", replay_clock=True)
        assert st.status == "historical" and "not live" in st.message

    def test_simulated_stall_detected_on_replay_clock(self):
        last = pd.Timestamp("2026-03-10 15:00", tz="UTC")
        assert assess_freshness(last, last, "5min", "simulated", replay_clock=True).status == "simulated"
        assert (
            assess_freshness(
                last, last + pd.Timedelta(minutes=45), "5min", "simulated", replay_clock=True
            ).status
            == "stale"
        )


CSV = """Date,Time,Open,High,Low,Close,Volume
2026-03-10,10:00,100,101,99,100.5,10
2026-03-10,10:05,100.5,102,100,101.75,12
2026-03-10,10:10,101.75,102.5,101,102,9
"""


class TestCsvImport:
    def test_aliases_and_naive_timezone(self):
        res = import_csv(CSV.encode(), ImportOptions(source_tz="America/New_York", tick_size=0.25))
        assert res.ok, res.errors
        assert res.df.index[0] == pd.Timestamp("2026-03-10 14:00", tz="UTC")
        assert list(res.df.columns[:4]) == ["open", "high", "low", "close"]

    def test_explicit_offsets_converted(self):
        csv = "timestamp,open,high,low,close\n2026-03-10T10:00:00-04:00,1,2,0.5,1.5\n2026-03-10T10:05:00-04:00,1,2,0.5,1.5\n"
        res = import_csv(csv.encode())
        assert res.ok, res.errors
        assert res.df.index[0].hour == 14

    def test_bar_close_stamps_shifted(self):
        res = import_csv(CSV.encode(), ImportOptions(timestamps_are_bar_close=True))
        assert res.df.index[0] == pd.Timestamp("2026-03-10 13:55", tz="UTC")

    def test_dst_ambiguous_time_raises_unless_resolved(self):
        csv = "timestamp,open,high,low,close\n2026-11-01 01:30,1,2,0.5,1.5\n2026-11-01 01:35,1,2,0.5,1.5\n"
        res = import_csv(csv.encode(), ImportOptions(source_tz="America/New_York"))
        assert not res.ok and "DST" in res.errors[0]
        res2 = import_csv(csv.encode(), ImportOptions(source_tz="America/New_York", ambiguous="earliest"))
        assert res2.df is not None and res2.df.index[0] == pd.Timestamp("2026-11-01 05:30", tz="UTC")

    def test_nonexistent_spring_forward_time(self):
        csv = "timestamp,open,high,low,close\n2026-03-08 02:30,1,2,0.5,1.5\n"
        assert not import_csv(csv.encode(), ImportOptions(source_tz="America/New_York")).ok

    def test_unsorted_requires_explicit_sort(self):
        lines = CSV.strip().splitlines()
        shuffled = "\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n"
        res = import_csv(shuffled.encode())
        assert not res.ok and any("chronological" in e for e in res.errors)
        res2 = import_csv(shuffled.encode(), ImportOptions(sort_if_unsorted=True))
        assert res2.ok and any("sorted" in n for n in res2.notes)

    def test_conflicting_duplicates_rejected(self):
        dup = CSV + "2026-03-10,10:05,100.5,103,100,101.75,12\n"
        res = import_csv(dup.encode(), ImportOptions(sort_if_unsorted=True, drop_exact_duplicates=True))
        assert not res.ok and any("repeat" in e for e in res.errors)

    def test_missing_columns(self):
        res = import_csv(b"timestamp,open,close\n2026-03-10 10:00,1,2\n")
        assert not res.ok and any("high" in e for e in res.errors)

    def test_contract_must_match_instrument(self):
        csv = (
            CSV.replace("Volume", "Volume,Contract")
            .replace(",10\n", ",10,MNQZ26\n")
            .replace(",12\n", ",12,MNQZ26\n")
            .replace(",9\n", ",9,MNQZ26\n")
        )
        assert import_csv(csv.encode(), ImportOptions(instrument="MNQ")).ok
        bad = import_csv(csv.encode(), ImportOptions(instrument="NQ"))
        assert not bad.ok and any("wrong tick value" in e for e in bad.errors)

    def test_contract_override(self):
        res = import_csv(CSV.encode(), ImportOptions(contract_override="nqm26"))
        assert set(res.df["contract"]) == {"NQM26"}


def test_make_bars_helper_deterministic():
    a, b = make_bars([(1, 2, 0.5, 1.5)] * 3), make_bars([(1, 2, 0.5, 1.5)] * 3)
    assert a.equals(b) and a.index[0] == T0
    assert np.isfinite(a["volume"]).all()
