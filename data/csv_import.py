"""CSV import for historical candles.

Expected columns (case-insensitive, common aliases accepted):

    timestamp | datetime | date + time   (required)
    open, high, low, close               (required)
    volume                               (optional - filters disabled if absent)
    contract | symbol                    (optional - e.g. NQZ26)

Timestamp rules:

* Values with an explicit UTC offset (``2026-03-09T14:30:00-04:00`` or a
  trailing ``Z``) are converted to UTC directly.
* Naive values are interpreted in ``source_tz`` (the timezone your platform
  exported in). DST ambiguity (the repeated hour in November) and
  non-existent times (the skipped hour in March) are reported as errors
  unless you explicitly choose how to resolve them.
* Some platforms stamp bars with their *close* time; set
  ``timestamps_are_bar_close`` and the importer shifts them to open time.

Nothing is silently repaired. Sorting or dropping exact duplicate rows only
happens when explicitly requested, and is reported.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from core.data_quality import DataQualityReport, validate_candles
from core.instruments import parse_contract_code
from core.timeutils import timeframe_delta, validate_timezone

ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "time_stamp", "ts", "date/time", "bar_time"],
    "date": ["date", "day"],
    "time": ["time", "hour"],
    "open": ["open", "o", "open_price"],
    "high": ["high", "h", "high_price"],
    "low": ["low", "l", "low_price"],
    "close": ["close", "c", "last", "close_price", "settle"],
    "volume": ["volume", "vol", "v", "total_volume"],
    "contract": ["contract", "symbol", "instrument", "ticker", "contract_symbol"],
}


@dataclass
class ImportOptions:
    source_tz: str = "America/New_York"
    timeframe: str = "5min"
    timestamps_are_bar_close: bool = False
    ambiguous: Literal["raise", "earliest", "latest"] = "raise"
    nonexistent: Literal["raise", "shift_forward"] = "raise"
    sort_if_unsorted: bool = False
    drop_exact_duplicates: bool = False
    contract_override: str | None = None
    tick_size: float | None = None
    instrument: str | None = (
        None  # if set, contract codes must belong to this instrument (e.g. MNQZ26 -> MNQ)
    )


@dataclass
class ImportResult:
    df: pd.DataFrame | None
    report: DataQualityReport | None
    column_map: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.errors


def _norm(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


def map_columns(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    norm = {_norm(c): c for c in columns}
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for a in aliases:
            if a in norm and norm[a] not in mapping.values():
                mapping[target] = norm[a]
                break
    errors = []
    if "timestamp" not in mapping and not ("date" in mapping and "time" in mapping):
        # A lone 'time' or 'date' column may hold full datetimes (e.g. "2026-03-10 10:00").
        if "time" in mapping:
            mapping["timestamp"] = mapping.pop("time")
        elif "date" in mapping:
            mapping["timestamp"] = mapping.pop("date")
        else:
            errors.append("No timestamp column found (expected 'timestamp'/'datetime' or 'date' + 'time').")
    for col in ("open", "high", "low", "close"):
        if col not in mapping:
            errors.append(f"Missing required column '{col}'.")
    return mapping, errors


def import_csv(source: str | bytes | io.IOBase, options: ImportOptions | None = None) -> ImportResult:
    opts = options or ImportOptions()
    errors: list[str] = []
    notes: list[str] = []
    try:
        validate_timezone(opts.source_tz)
    except ValueError as exc:
        return ImportResult(None, None, errors=[str(exc)])
    try:
        if isinstance(source, bytes):
            raw = pd.read_csv(io.BytesIO(source))
        else:
            raw = pd.read_csv(source)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        return ImportResult(None, None, errors=[f"Could not parse CSV: {exc}"])
    if raw.empty:
        return ImportResult(None, None, errors=["CSV contains no rows."])

    mapping, map_errors = map_columns(list(raw.columns))
    if map_errors:
        return ImportResult(None, None, mapping, map_errors)

    # Timestamps
    if "timestamp" in mapping:
        ts_raw = raw[mapping["timestamp"]].astype(str)
    else:
        ts_raw = raw[mapping["date"]].astype(str) + " " + raw[mapping["time"]].astype(str)
    try:
        parsed = pd.to_datetime(ts_raw, utc=False, format="mixed")
    except Exception as exc:  # noqa: BLE001
        return ImportResult(None, None, mapping, [f"Unparseable timestamps: {exc}"])
    if getattr(parsed.dt, "tz", None) is not None:
        ts = parsed.dt.tz_convert("UTC")
        notes.append("Timestamps carried explicit UTC offsets; converted to UTC.")
    else:
        has_offset = ts_raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        if has_offset.any() and not has_offset.all():
            return ImportResult(
                None, None, mapping, ["Mixed timestamps: some have UTC offsets and some do not."]
            )
        try:
            amb: object = "raise"
            if opts.ambiguous != "raise":
                # Boolean flags: True = treat the repeated hour as DST (the earlier occurrence).
                amb = np.full(len(parsed), opts.ambiguous == "earliest")
            nonexist = "raise" if opts.nonexistent == "raise" else "shift_forward"
            ts = parsed.dt.tz_localize(opts.source_tz, ambiguous=amb, nonexistent=nonexist).dt.tz_convert(
                "UTC"
            )
            notes.append(f"Naive timestamps interpreted as {opts.source_tz}.")
        except Exception as exc:  # noqa: BLE001
            return ImportResult(
                None,
                None,
                mapping,
                [
                    f"Timezone localisation failed in {opts.source_tz}: {exc}. This usually means a DST-ambiguous or "
                    "non-existent local time; choose an explicit resolution or export with UTC offsets."
                ],
            )
    if opts.timestamps_are_bar_close:
        ts = ts - timeframe_delta(opts.timeframe)
        notes.append(f"Shifted timestamps by -{opts.timeframe} (input stamped bars at close time).")

    df = pd.DataFrame(
        {
            "open": pd.to_numeric(raw[mapping["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[mapping["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[mapping["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[mapping["close"]], errors="coerce"),
        }
    )
    if "volume" in mapping:
        df["volume"] = pd.to_numeric(raw[mapping["volume"]], errors="coerce")
    else:
        notes.append("No volume column: VWAP and relative-volume filters will be unavailable.")
    if opts.contract_override:
        df["contract"] = opts.contract_override.strip().upper()
    elif "contract" in mapping:
        df["contract"] = raw[mapping["contract"]].astype(str).str.strip().str.upper()
    df.index = pd.DatetimeIndex(ts, name="ts")

    if not df.index.is_monotonic_increasing:
        if opts.sort_if_unsorted:
            df = df.sort_index(kind="stable")
            notes.append("Rows were not in chronological order and were sorted (as requested).")
        else:
            errors.append(
                "Rows are not in chronological order. Enable 'sort by timestamp' to sort explicitly."
            )
    if df.index.duplicated().any():
        if opts.drop_exact_duplicates:
            before = len(df)
            tmp = df.reset_index()
            tmp = tmp.drop_duplicates(keep="first")
            df = tmp.set_index("ts")
            notes.append(f"Dropped {before - len(df)} exact duplicate row(s) (as requested).")
        if df.index.duplicated().any():
            errors.append(
                f"{int(df.index.duplicated().sum())} timestamps repeat with different values - "
                "cannot decide which bar is correct."
            )

    if opts.instrument and "contract" in df.columns:
        for code in pd.unique(df["contract"].astype(str)):
            try:
                root, _m, _y = parse_contract_code(code)
            except ValueError:
                errors.append(f"Unrecognised contract code {code!r} (expected e.g. {opts.instrument}Z26).")
                continue
            if root != opts.instrument.upper():
                errors.append(
                    f"Contract {code} belongs to {root}, not the selected instrument {opts.instrument}. "
                    "Importing it under the wrong symbol would apply the wrong tick value."
                )

    report = validate_candles(df, opts.timeframe, opts.tick_size, allow_multiple_contracts=True)
    for issue in report.errors:
        if issue.code not in ("unsorted", "duplicates"):
            errors.append(issue.message)
    return ImportResult(df, report, mapping, errors, notes)
