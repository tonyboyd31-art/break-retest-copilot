"""Settings & Data Sources: validated, persisted settings; CSV import; adapters; instrument registry."""

from __future__ import annotations

import os
import typing
from datetime import date

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.config import SCORE_FACTORS, AppSettings, diff_strategy
from core.instruments import listed_contracts
from data import repositories as repo
from data.adapters.databento_adapter import DatabentoAdapter
from data.csv_import import ImportOptions, import_csv
from ui import components as ui
from ui import state

s = state.settings()
tz = state.tz()
db = state.db()
reg = state.app().registry
ui.page_header(
    "Settings & Data",
    "Every setting is checked before it is saved. Strategy changes are flagged against saved backtests.",
    eyebrow="Preferences",
    accent="Data",
)

HELP = {
    "left_bars": "Bars to the left a swing must exceed.",
    "right_bars": "Bars to the right needed to confirm a swing (confirmation delay).",
    "min_prominence_ticks": "Minimum swing prominence in ticks.",
    "tolerance_ticks": "Zone clustering tolerance in ticks (max with the ATR tolerance).",
    "min_reactions": "Distinct reactions needed for a zone to generate breakout setups.",
    "opposing_min_reactions": "Reactions needed for a zone to count as an obstacle for room/targets.",
    "close_buffer_ticks": "Breakout close must clear the zone by this many ticks (max with ATR buffer).",
    "max_bars_to_retest": "Maximum bars between breakout and retest.",
    "retest_tolerance_ticks": "How close to the zone counts as a retest.",
    "min_score": "Minimum quality score to qualify (a ranking, not a probability).",
    "min_rr": "Minimum planned reward/risk to the first target.",
    "breakeven_thresholds_r": "Comma-separated R levels for breakeven REVIEW prompts.",
    "profit_r_milestones": "Comma-separated R levels for profit-protection REVIEW prompts.",
    "signal_windows": "Local-time windows in which setups may qualify, e.g. 09:30-11:30, 13:30-15:45",
}


def _base_type(ann):
    origin = typing.get_origin(ann)
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        return _base_type(args[0]) if args else ann, True
    return ann, False


def model_form(section: str, exclude: tuple[str, ...] = ()) -> dict | None:
    model = getattr(s, section)
    values: dict = {}
    cols = st.columns(3)
    i = 0
    for name, field in type(model).model_fields.items():
        if name in exclude:
            continue
        ann, optional = _base_type(field.annotation)
        cur = getattr(model, name)
        label = name.replace("_", " ")
        col = cols[i % 3]
        i += 1
        key = f"set_{section}_{name}"
        help_ = HELP.get(name, field.description)
        origin = typing.get_origin(ann)
        if ann is bool:
            values[name] = col.toggle(label, value=bool(cur), key=key, help=help_)
        elif origin is typing.Literal:
            opts = list(typing.get_args(ann))
            values[name] = col.selectbox(label, opts, index=opts.index(cur), key=key, help=help_)
        elif ann is int:
            v = col.number_input(
                label,
                value=int(cur) if cur is not None else 0,
                step=1,
                key=key,
                help=(help_ or "") + (" (0 = none)" if optional else ""),
            )
            values[name] = None if optional and v == 0 else int(v)
        elif ann is float:
            v = col.number_input(
                label,
                value=float(cur) if cur is not None else 0.0,
                step=0.05,
                format="%.4g",
                key=key,
                help=(help_ or "") + (" (0 = none)" if optional else ""),
            )
            values[name] = None if optional and v == 0 else float(v)
        elif ann is str:
            v = col.text_input(label, value=cur or "", key=key, help=help_)
            values[name] = (v or None) if optional else v
        elif ann is date:
            values[name] = col.date_input(label, value=cur, key=key, help=help_)
        elif origin is list and typing.get_args(ann) and typing.get_args(ann)[0] is float:
            txt = col.text_input(label, ", ".join(f"{x:g}" for x in (cur or [])), key=key, help=help_)
            try:
                parsed = [float(x) for x in txt.replace(";", ",").split(",") if x.strip()]
                values[name] = parsed if (parsed or not optional) else None
            except ValueError:
                col.error("Enter comma-separated numbers")
                values[name] = cur
        elif origin is list and typing.get_args(ann) and typing.get_origin(typing.get_args(ann)[0]) is tuple:
            txt = col.text_input(label, ", ".join(f"{a}-{b}" for a, b in cur), key=key, help=help_)
            try:
                values[name] = [tuple(p.strip().split("-")) for p in txt.split(",") if p.strip()]
            except Exception:  # noqa: BLE001
                values[name] = cur
        elif origin is list and typing.get_args(ann) and typing.get_args(ann)[0] is str:
            values[name] = col.multiselect(
                label,
                list(SCORE_FACTORS) if name == "essential_factors" else cur,
                default=cur,
                key=key,
                help=help_,
            )
        else:
            continue  # complex fields handled separately
    return values


def save_section(section: str, values: dict) -> None:
    global s
    try:
        new = s.updated(section, **values)
    except ValidationError as exc:
        st.error(
            "Not saved - invalid settings:\n\n"
            + "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        )
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Not saved: {exc}")
        return
    changes = diff_strategy(s, new)
    state.save_settings(new)
    s = new
    st.success("Settings saved.")
    if changes:
        runs = repo.list_backtests(db)
        stale = [r for r in runs if r["settings_hash"] != new.strategy_hash()]
        st.warning(
            "Strategy parameters changed: "
            + "; ".join(changes[:6])
            + (
                f". {len(stale)} saved backtest(s) were run with different settings and no longer describe the "
                "current strategy."
                if stale
                else ""
            )
            + " Previously recorded setups keep the scores they had when generated."
        )


tabs = st.tabs(
    [
        "Strategy",
        "Scoring",
        "Trade plan",
        "Sessions & display",
        "Management alerts",
        "Risk",
        "Backtest defaults",
        "Data sources",
        "Import CSV",
        "Instruments",
    ]
)

with tabs[0]:
    for sec, title in (
        ("timeframes", "Timeframes & HTF trend"),
        ("swings", "Swing detection"),
        ("zones", "Support/resistance zones"),
        ("breakout", "Breakout filters"),
        ("retest", "Retest & confirmation"),
    ):
        with st.form(f"f_{sec}"):
            st.markdown(f"**{title}**")
            vals = model_form(sec)
            if st.form_submit_button(f"Save {title.lower()}"):
                save_section(sec, vals)

with tabs[1]:
    with st.form("f_scoring"):
        st.markdown(
            "**Factor weights** (must sum to 100). Defaults are engineering starting points, not optimised "
            "values."
        )
        wcols = st.columns(4)
        weights = {}
        for j, (k, lbl) in enumerate(SCORE_FACTORS.items()):
            weights[k] = wcols[j % 4].number_input(
                lbl, 0.0, 100.0, float(s.scoring.weights[k]), step=1.0, key=f"w_{k}"
            )
        st.caption(f"Current total: {sum(weights.values()):g}")
        presets = {"70": 70.0, "75": 75.0, "80": 80.0, "85": 85.0, "90": 90.0, "custom": None}
        cur_p = next((k for k, v in presets.items() if v == s.scoring.min_score), "custom")
        p1, p2 = st.columns(2)
        preset = p1.selectbox("Minimum score preset", list(presets), index=list(presets).index(cur_p))
        custom = p2.number_input("Custom minimum score", 0.0, 100.0, float(s.scoring.min_score), step=1.0)
        vals = model_form("scoring", exclude=("weights", "min_score"))
        if st.form_submit_button("Save scoring"):
            vals["weights"] = weights
            vals["min_score"] = presets[preset] if presets[preset] is not None else custom
            save_section("scoring", vals)
    st.markdown("""
**Missing-data policy.** `renormalize`: the score is computed over factors that have data, and the setup is rejected
if less than `min available weight` has data or an essential factor is missing. `zero`: missing factors earn 0
points. A factor without data is never awarded points silently.
""")

with tabs[2], st.form("f_planning"):
    vals = model_form("planning")
    if st.form_submit_button("Save trade-plan settings"):
        save_section("planning", vals)

with tabs[3]:
    with st.form("f_general"):
        st.markdown("**Display**")
        vals = model_form("general")
        if st.form_submit_button("Save display settings"):
            save_section("general", vals)
    with st.form("f_sessions"):
        st.markdown("**Session filters & VWAP**")
        vals = model_form("sessions")
        if st.form_submit_button("Save session settings"):
            save_section("sessions", vals)

with tabs[4], st.form("f_management"):
    st.caption(
        "These control REVIEW prompts for trades you registered. Nothing here can move a stop or exit."
    )
    vals = model_form("management", exclude=("suppressed_categories",))
    from core.events import ALERT_CATEGORIES

    mgmt_cats = [
        "breakeven_review",
        "profit_protection",
        "trailing_stop",
        "target_approaching",
        "trade_warning",
    ]
    vals["suppressed_categories"] = st.multiselect(
        "Suppress these trade-management prompts",
        mgmt_cats,
        default=s.management.suppressed_categories,
        format_func=lambda c: ALERT_CATEGORIES[c][0],
    )
    if st.form_submit_button("Save management settings"):
        save_section("management", vals)

with tabs[5], st.form("f_risk"):
    vals = model_form("risk", exclude=("commission_per_contract_rt",))
    st.markdown("**Commission per contract, round trip (USD)**")
    ccols = st.columns(6)
    comm = {}
    for j, sym in enumerate(reg.symbols()):
        comm[sym] = ccols[j % 6].number_input(
            sym, 0.0, 100.0, float(s.risk.commission_for(sym)), step=0.1, key=f"comm_{sym}"
        )
    if st.form_submit_button("Save risk settings"):
        vals["commission_per_contract_rt"] = comm
        save_section("risk", vals)

with tabs[6], st.form("f_backtest"):
    vals = model_form("backtest")
    if st.form_submit_button("Save backtest defaults"):
        save_section("backtest", vals)

with tabs[7]:
    with st.form("f_data"):
        st.markdown("**Data defaults & freshness**")
        vals = model_form("data")
        if st.form_submit_button("Save data settings"):
            save_section("data", vals)
    st.markdown("#### Adapters")
    dbn = DatabentoAdapter().status()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "source": "Synthetic demo",
                    "status": "available",
                    "feed": "simulated",
                    "tested": True,
                    "notes": "Deterministic simulated candles. Not real prices.",
                },
                {
                    "source": "Imported CSV",
                    "status": f"{len(repo.list_datasets(db))} dataset(s)",
                    "feed": "historical",
                    "tested": True,
                    "notes": "Your own exported candles.",
                },
                {
                    "source": "Databento",
                    "status": "available" if dbn.available else "unavailable",
                    "feed": dbn.feed_type,
                    "tested": dbn.tested,
                    "notes": dbn.message,
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "DATABENTO_API_KEY is "
        + ("set (value hidden)." if os.environ.get("DATABENTO_API_KEY") else "not set.")
        + " Live data is only used through documented, authorised vendor APIs. No broker connection exists."
    )
    st.markdown("#### Feed status history")
    ms = repo.list_market_status(db)
    if ms:
        st.dataframe(pd.DataFrame(ms), hide_index=True, width="stretch")
    st.markdown("#### Imported datasets")
    dss = repo.list_datasets(db)
    if dss:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        k: d[k]
                        for k in (
                            "id",
                            "name",
                            "instrument",
                            "contract",
                            "rows",
                            "start_ts",
                            "end_ts",
                            "source_tz",
                            "created_at",
                        )
                    }
                    for d in dss
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        dsel = st.selectbox(
            "Delete dataset",
            [d["id"] for d in dss],
            format_func=lambda i: next(d["name"] for d in dss if d["id"] == i),
        )
        if st.checkbox("Confirm delete") and st.button("Delete dataset"):
            repo.delete_dataset(db, dsel)
            st.cache_data.clear()
            st.rerun()
    if st.button("Reset ALL settings to defaults"):
        state.save_settings(AppSettings())
        st.rerun()

with tabs[8]:
    st.markdown("""
**CSV format:** columns `timestamp` (or `date` + `time`), `open`, `high`, `low`, `close`, optional `volume`, optional
`contract` (e.g. `NQZ26`). One row per 5-minute bar. Timestamps with UTC offsets are converted; naive timestamps
are interpreted in the timezone you choose below. See `sample_data/` for an example.
""")
    up = st.file_uploader("CSV file", type=["csv"])
    i1, i2, i3 = st.columns(3)
    syms = reg.symbols()
    cur_inst = st.session_state.get("sel_instrument", s.general.default_instrument)
    inst = i1.selectbox(
        "Instrument",
        syms,
        index=syms.index(cur_inst) if cur_inst in syms else 0,
        key="imp_inst",
        help="Must match the file: tick values differ between minis and micros.",
    )
    src_tz = i2.selectbox(
        "Timezone of naive timestamps",
        ["America/New_York", "America/Chicago", "UTC", "Europe/London"],
        key="imp_tz",
    )
    contract = i3.text_input("Contract (if the file has no contract column)", key="imp_contract")
    j1, j2, j3, j4 = st.columns(4)
    close_stamp = j1.checkbox("Timestamps are bar CLOSE times", key="imp_close")
    sort_rows = j2.checkbox("Sort rows by timestamp", key="imp_sort")
    dedupe = j3.checkbox("Drop exact duplicate rows", key="imp_dedupe")
    amb = j4.selectbox("DST-ambiguous times", ["raise", "earliest", "latest"], key="imp_amb")
    if up is not None:
        res = import_csv(
            up.getvalue(),
            ImportOptions(
                source_tz=src_tz,
                timestamps_are_bar_close=close_stamp,
                sort_if_unsorted=sort_rows,
                drop_exact_duplicates=dedupe,
                ambiguous=amb,
                contract_override=contract or None,
                tick_size=reg.get(inst).tick_size,
                instrument=inst,
            ),
        )
        for n in res.notes:
            st.caption(n)
        for e in res.errors:
            st.error(e)
        if res.report is not None:
            st.markdown(
                f"**Validation:** {res.report.summary()}; contracts: {res.report.contracts or 'unknown'}; "
                f"volume reliable: {res.report.volume_reliable}"
            )
            if res.report.issues:
                st.dataframe(ui.quality_table(res.report), hide_index=True, width="stretch")
        if res.df is not None:
            st.dataframe(res.df.head(20), width="stretch")
        if res.ok:
            name = st.text_input("Dataset name", f"{inst} {up.name}")
            if st.button("Save dataset", type="primary"):
                if any(d["name"] == name for d in repo.list_datasets(db)):
                    st.error(
                        f"A dataset named '{name}' already exists - choose another name or delete the old one "
                        "under Data sources."
                    )
                    st.stop()
                try:
                    q = {"summary": res.report.summary(), "issues": [i.__dict__ for i in res.report.issues]}
                    did = repo.create_dataset(
                        db,
                        name,
                        inst,
                        "5min",
                        res.df,
                        "csv",
                        "historical",
                        src_tz,
                        q,
                        contract=(contract or None),
                    )
                    st.success(
                        f"Saved dataset #{did}. Select 'Imported CSV' as the data source in the sidebar."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not save: {exc}")
        elif res.df is not None:
            st.warning("Fix the errors above before saving (nothing is repaired silently).")

with tabs[9]:
    st.caption(
        "Metadata is validated (tick value = tick size x multiplier) before any calculation uses it. Values "
        "were entered from CME published specifications at build time - confirm them at the source."
    )
    rows_db = {r["symbol"]: r for r in repo.list_instrument_rows(db)}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "symbol": sp.symbol,
                    "name": sp.name,
                    "exchange": sp.exchange,
                    "tick size": sp.tick_size,
                    "tick value": sp.tick_value,
                    "point value": sp.point_value,
                    "months": sp.contract_months,
                    "mini-equivalent": sp.mini_equivalent,
                    "status": rows_db.get(sp.symbol, {}).get("metadata_status"),
                    "verified on": rows_db.get(sp.symbol, {}).get("verified_on"),
                    "source": sp.source_url,
                }
                for sp in reg.all()
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"source": st.column_config.LinkColumn("source")},
    )
    sym = st.selectbox("Contract calendar for", reg.symbols(), key="cal_sym")
    spc = reg.get(sym)
    cal = listed_contracts(spc, date.today(), 6)
    st.dataframe(
        pd.DataFrame(
            [
                {"contract": c.code, "expiration (calc.)": c.expiration, "roll date (calc.)": c.roll_date}
                for c in cal
            ]
        ),
        hide_index=True,
    )
    st.caption(cal[0].calculation_note if cal else "")
    v1, v2 = st.columns(2)
    vd = v1.date_input("I verified this instrument's specs on", date.today(), key="ver_date")
    if v2.button(f"Mark {sym} metadata verified"):
        repo.mark_instrument_verified(db, sym, vd)
        st.success("Recorded.")
        st.rerun()
