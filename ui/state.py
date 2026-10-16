"""Session state, cached data loading and the global market context (sidebar)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st

from core.config import AppSettings
from core.scanner import ScanResult
from data import repositories as repo
from data.database import Database
from notifications.manager import AlertManager
from services.bootstrap import AppContext, bootstrap
from services.market_service import SOURCE_LABELS, MarketData, MarketSelection, load_market
from services.scan_service import persist_scan, scan_market
from services.trade_service import monitor_open_trades


@st.cache_resource(show_spinner="Starting...")
def app() -> AppContext:
    return bootstrap()


def db() -> Database:
    return app().db


def settings() -> AppSettings:
    if "settings" not in st.session_state:
        s, warnings = repo.load_settings(db())
        st.session_state["settings"] = s
        st.session_state["settings_warnings"] = warnings
    return st.session_state["settings"]


def save_settings(new: AppSettings) -> None:
    repo.save_settings(db(), new)
    st.session_state["settings"] = new
    st.cache_data.clear()


def tz() -> str:
    return settings().general.display_timezone


def _bucket(source: str) -> int:
    return int(time.time() // 60) if source == "databento" else 0


@st.cache_data(show_spinner=False, max_entries=48)
def _cached_market(
    source: str,
    instrument: str,
    dataset_id: int | None,
    contract: str | None,
    replay_bar: int | None,
    stall: int,
    settings_json: str,
    db_path: str,
    bucket: int,
) -> MarketData:
    s = AppSettings.from_json(settings_json)
    sel = MarketSelection(source, instrument, dataset_id, contract, replay_bar, stall)
    return load_market(Database(db_path), s, sel)


@st.cache_data(show_spinner="Running strategy scan...", max_entries=48)
def _cached_scan(
    source: str,
    instrument: str,
    dataset_id: int | None,
    contract: str | None,
    replay_bar: int | None,
    stall: int,
    settings_json: str,
    db_path: str,
    bucket: int,
) -> ScanResult:
    md = _cached_market(
        source, instrument, dataset_id, contract, replay_bar, stall, settings_json, db_path, bucket
    )
    return scan_market(md, AppSettings.from_json(settings_json))


def selection() -> MarketSelection:
    s = settings()
    ss = st.session_state
    return MarketSelection(
        source=ss.get("sel_source", s.data.source),
        instrument=ss.get("sel_instrument", s.general.default_instrument),
        dataset_id=ss.get("sel_dataset"),
        contract=ss.get("sel_contract"),
        replay_bar=ss.get("replay_bar"),
        stall_minutes=int(ss.get("stall_minutes", 0)),
    )


def _args(sel: MarketSelection, replay_bar: int | None) -> tuple:
    s = settings()
    return (
        sel.source,
        sel.instrument,
        sel.dataset_id,
        sel.contract,
        replay_bar,
        sel.stall_minutes,
        s.to_json(),
        str(db().path),
        _bucket(sel.source),
    )


@dataclass
class MarketContext:
    md: MarketData
    scan: ScanResult | None


def market_context(process: bool = True) -> MarketContext:
    """Load data + scan for the current selection (cached) and process alerts once per state."""
    sel = selection()
    md = _cached_market(*_args(sel, sel.replay_bar))
    scan = _cached_scan(*_args(sel, sel.replay_bar)) if md.ok else None
    if process and md.ok and scan is not None:
        _process(md, scan)
    return MarketContext(md, scan)


def _process(md: MarketData, scan: ScanResult) -> None:
    key = (
        md.context_key,
        md.cursor,
        md.selection.stall_minutes,
        scan.settings_hash,
        settings().to_json().__hash__(),
        _bucket(md.selection.source),
        st.session_state.get("trade_rev", 0),
    )
    done = st.session_state.setdefault("processed_keys", set())
    if key in done:
        return
    s = settings()
    am = AlertManager(db(), s.notifications)
    summary = persist_scan(db(), md, scan, s, am)
    monitor_open_trades(db(), s, md, scan, am)
    done.add(key)
    new = repo.list_alerts(db(), acknowledged=False, limit=20)
    seen = st.session_state.setdefault("toasted", set())
    fresh = [a for a in new if a["id"] not in seen]
    for a in fresh[:4]:
        st.toast(f"**{a['title']}**\n\n{a['message'][:160]}", icon=":material/notifications:")
    seen.update(a["id"] for a in new)
    if s.notifications.browser_enabled and fresh:
        from ui.browser_notify import push_browser_notifications

        push_browser_notifications(fresh[:4])
    st.session_state["last_summary"] = summary


def bump_trade_rev() -> None:
    """Force trade monitoring to re-run after a trade/action changes."""
    st.session_state["trade_rev"] = st.session_state.get("trade_rev", 0) + 1


def render_sidebar() -> None:
    s = settings()
    ctx = app()
    ss = st.session_state
    with st.sidebar:
        st.markdown("### Break & Retest Copilot")
        st.caption("Analysis & decision support only. This app never places, modifies or cancels orders.")
        symbols = ctx.registry.symbols()
        default_sym = ss.get("sel_instrument", s.general.default_instrument)
        sym = st.selectbox(
            "Instrument",
            symbols,
            index=symbols.index(default_sym) if default_sym in symbols else 0,
            key="sb_instrument",
        )
        if sym != ss.get("sel_instrument"):
            ss["sel_instrument"] = sym
            ss["sel_contract"] = None
            ss["replay_bar"] = None
        sources = list(SOURCE_LABELS)
        cur_src = ss.get("sel_source", s.data.source)
        src = st.selectbox(
            "Data source",
            sources,
            index=sources.index(cur_src) if cur_src in sources else 0,
            format_func=lambda k: SOURCE_LABELS[k],
            key="sb_source",
        )
        if src != ss.get("sel_source"):
            ss["sel_source"] = src
            ss["sel_contract"] = None
            ss["replay_bar"] = None
        if src == "csv":
            ds = [d for d in repo.list_datasets(db()) if d["instrument"] == sym]
            if not ds:
                st.info("No imported CSV for this instrument. Import one under Settings & Data Sources.")
                ss["sel_dataset"] = None
            else:
                ids = [d["id"] for d in ds]
                cur = ss.get("sel_dataset")
                pick = st.selectbox(
                    "Dataset",
                    ids,
                    index=ids.index(cur) if cur in ids else 0,
                    format_func=lambda i: next(d["name"] for d in ds if d["id"] == i),
                    key="sb_dataset",
                )
                if pick != cur:
                    ss["sel_dataset"] = pick
                    ss["replay_bar"] = None
        sel = selection()
        base = _cached_market(*_args(sel, None))
        if base.error:
            st.error(base.error)
            return
        contracts = [seg["contract"] for seg in base.segments]
        if len(contracts) > 1:
            cur_c = ss.get("sel_contract") or contracts[-1]
            c = st.selectbox(
                "Contract segment",
                contracts,
                index=contracts.index(cur_c) if cur_c in contracts else len(contracts) - 1,
                key="sb_contract",
                help="Contracts are analysed separately - prices of different months are never spliced.",
            )
            if c != (ss.get("sel_contract") or contracts[-1]):
                ss["sel_contract"] = c
                ss["replay_bar"] = None
                base = _cached_market(*_args(selection(), None))
        if base.is_replay:
            n = len(base.segment)
            with st.expander("Replay clock", expanded=True):
                st.caption(
                    "Signals are computed only from bars up to this point - exactly what was knowable then."
                )
                cur = ss.get("replay_bar")
                pos = n - 1 if cur is None else min(int(cur), n - 1)
                cols = st.columns(4, gap="small")
                steps = [
                    (":material/fast_rewind:", -12, "Back 12 bars (1 hour)"),
                    (":material/chevron_left:", -1, "Back 1 bar"),
                    (":material/chevron_right:", 1, "Forward 1 bar"),
                    (":material/fast_forward:", 12, "Forward 12 bars (1 hour)"),
                ]
                for col, (icon, d, tip) in zip(cols, steps, strict=True):
                    if col.button("", icon=icon, key=f"rp{d}", width="stretch", help=tip):
                        pos = max(0, min(n - 1, pos + d))
                        ss["replay_bar"] = None if pos == n - 1 else pos
                        st.rerun()
                if pos < n - 1 and st.button("Jump to latest bar", key="rp_end", width="stretch"):
                    ss["replay_bar"] = None
                    st.rerun()
                new_pos = st.slider(
                    "Bar",
                    50,
                    n - 1,
                    max(pos, 50),
                    key=f"rp_slider_{base.context_key}_{pos}",
                    label_visibility="collapsed",
                )
                if new_pos != pos:
                    ss["replay_bar"] = None if new_pos == n - 1 else new_pos
                    st.rerun()
                ts = base.segment.index[pos]
                st.caption(
                    f"Bar {pos + 1}/{n} - {ts.tz_convert(s.general.display_timezone):%a %Y-%m-%d %H:%M %Z}"
                )
                if base.feed_type == "simulated":
                    stall = st.number_input(
                        "Simulate feed stall (minutes)",
                        0,
                        600,
                        int(ss.get("stall_minutes", 0)),
                        step=5,
                        help="Demo only: pretend no new bar arrived for this long to "
                        "see the stale-data safeguards.",
                    )
                    if stall != ss.get("stall_minutes", 0):
                        ss["stall_minutes"] = int(stall)
                        st.rerun()
        unack = repo.count_unacknowledged(db())
        total = sum(unack.values())
        if total:
            hi = unack.get("critical", 0) + unack.get("high", 0)
            st.markdown(f"**{total}** unacknowledged alert(s)" + (f" - **{hi} high/critical**" if hi else ""))
        if ss.get("settings_warnings"):
            for w in ss["settings_warnings"]:
                st.warning(w)
