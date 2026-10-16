"""Session calendar, DST handling, tick math and contract calendar."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.indicators import vwap_anchor_keys
from core.instruments import (
    InstrumentSpec,
    contract_info,
    front_contract,
    get_instrument,
    is_on_tick,
    parse_contract_code,
    price_to_ticks,
    round_to_tick,
    third_friday,
    third_last_business_day,
)
from core.resampling import resample_session_aware
from core.timeutils import (
    in_time_windows,
    is_market_open,
    market_open_mask,
    session_label,
    timeframe_minutes,
    to_iso,
    trade_date,
)
from tests.conftest import make_bars


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


class TestSessionCalendar:
    def test_market_hours(self):
        assert is_market_open(ts("2026-03-10 10:00"))  # Tue RTH
        assert not is_market_open(ts("2026-03-10 17:30"))  # daily halt
        assert is_market_open(ts("2026-03-10 18:00"))  # reopen
        assert not is_market_open(ts("2026-03-14 12:00"))  # Saturday
        assert not is_market_open(ts("2026-03-15 17:59"))  # Sunday before open
        assert is_market_open(ts("2026-03-15 18:00"))  # Sunday open
        assert not is_market_open(ts("2026-03-13 17:00"))  # Friday close

    def test_vectorised_mask_matches_scalar(self):
        idx = pd.date_range("2026-03-06", "2026-03-10", freq="35min", tz="UTC")
        mask = market_open_mask(idx)
        assert list(mask) == [is_market_open(t) for t in idx]

    def test_trade_date_rolls_at_18_et(self):
        assert trade_date(ts("2026-03-10 17:55")) == date(2026, 3, 10)
        assert trade_date(ts("2026-03-10 18:00")) == date(2026, 3, 11)
        assert trade_date(ts("2026-03-15 19:00")) == date(2026, 3, 16)  # Sunday evening -> Monday

    def test_trade_date_across_dst_change(self):
        # US DST starts Sunday 2026-03-08 02:00 local; Sunday 18:00 EDT session belongs to Monday.
        assert trade_date(ts("2026-03-08 18:05")) == date(2026, 3, 9)
        assert ts("2026-03-08 18:00").hour == 22  # EDT = UTC-4
        assert ts("2026-03-06 18:00").hour == 23  # EST = UTC-5

    def test_session_labels(self):
        assert session_label(ts("2026-03-10 09:35")) == "NY AM"
        assert session_label(ts("2026-03-10 13:00")) == "NY PM"
        assert session_label(ts("2026-03-10 04:00")) == "London"
        assert session_label(ts("2026-03-10 23:00")) == "Asia"

    def test_time_windows_wrap_midnight(self):
        assert in_time_windows(ts("2026-03-10 23:30"), [("18:00", "03:00")])
        assert not in_time_windows(ts("2026-03-10 09:00"), [("18:00", "03:00")])
        assert in_time_windows(ts("2026-03-10 09:00"), [])

    def test_timeframes(self):
        assert timeframe_minutes("5min") == 5
        assert timeframe_minutes("4h") == 240
        with pytest.raises(ValueError):
            timeframe_minutes("5x")

    def test_naive_timestamps_rejected(self):
        with pytest.raises(ValueError):
            to_iso(pd.Timestamp("2026-01-01 10:00"))


class TestDST:
    def test_4h_bins_align_to_session_open_both_sides_of_dst(self):
        # Friday before DST (EST) and Tuesday after (EDT): 4h bins start at 18/22/02/06/10/14 ET.
        for start in (ts("2026-03-05 18:00"), ts("2026-03-09 18:00")):
            df = make_bars([(100, 101, 99, 100)] * (23 * 12), start=start)
            htf = resample_session_aware(df, "4h")
            local_hours = [t.tz_convert("America/New_York").hour for t in htf.index]
            assert local_hours == [18, 22, 2, 6, 10, 14]
            # last bin truncated at the 17:00 close: 3h of 5-minute bars
            assert htf["n_bars"].iloc[-1] == 36
            assert htf["expected_bars"].iloc[-1] == 36

    def test_vwap_anchor_resets_at_session_open_after_dst(self):
        idx = pd.DatetimeIndex([ts("2026-03-09 16:55"), ts("2026-03-09 18:00"), ts("2026-03-10 09:00")])
        keys = vwap_anchor_keys(idx, "globex")
        assert keys[0] != keys[1]
        assert keys[1] == keys[2]


class TestTickMath:
    def test_round_to_tick(self):
        assert round_to_tick(100.13, 0.25) == 100.25
        assert round_to_tick(100.12, 0.25) == 100.0
        assert round_to_tick(100.01, 0.25, "up") == 100.25
        assert round_to_tick(100.24, 0.25, "down") == 100.0
        assert round_to_tick(2419.7, 0.1, "down") == 2419.7  # float noise must not drop a tick

    def test_ticks_and_dollars(self, nq, mnq, gc):
        assert price_to_ticks(10.0, nq.tick_size) == 40
        assert nq.dollars(10.0) == 200.0
        assert mnq.dollars(10.0) == 20.0
        assert gc.ticks(3.7) == 37
        assert gc.dollars(3.7) == pytest.approx(370.0)
        assert is_on_tick(2419.7, 0.1)
        assert not is_on_tick(100.1, 0.25)

    def test_instruments_have_distinct_values(self):
        vals = {
            s: (get_instrument(s).tick_value, get_instrument(s).point_value)
            for s in ("NQ", "MNQ", "ES", "MES", "GC", "MGC")
        }
        assert vals["NQ"] == (5.0, 20.0) and vals["MNQ"] == (0.5, 2.0)
        assert vals["ES"] == (12.5, 50.0) and vals["MES"] == (1.25, 5.0)
        assert vals["GC"] == (10.0, 100.0) and vals["MGC"] == (1.0, 10.0)

    def test_inconsistent_metadata_rejected(self, nq):
        data = nq.model_dump()
        data["tick_value"] = 6.0
        with pytest.raises(ValueError, match="tick_value"):
            InstrumentSpec.model_validate(data)


class TestContractCalendar:
    def test_third_friday_and_gold_expiry(self):
        assert third_friday(2026, 12) == date(2026, 12, 18)
        assert third_friday(2026, 3) == date(2026, 3, 20)
        assert third_last_business_day(2026, 12) == date(2026, 12, 29)

    def test_parse_codes(self):
        assert parse_contract_code("NQZ26") == ("NQ", "Z", 2026)
        assert parse_contract_code("NQZ2026") == ("NQ", "Z", 2026)
        assert parse_contract_code("MNQH7") == ("MNQ", "H", 2027)

    def test_front_contract_rolls_before_expiry(self, nq):
        dec = contract_info(nq, "Z", 2026)
        assert dec.roll_date == date(2026, 12, 10)
        assert front_contract(nq, date(2026, 12, 9)).code == "NQZ26"
        assert front_contract(nq, date(2026, 12, 10)).code == "NQH27"

    def test_gold_rolls_before_first_notice(self, gc):
        info = contract_info(gc, "Z", 2026)
        assert info.roll_date < date(2026, 11, 30)  # before FND (last business day of November)
        assert "holidays" in info.calculation_note
