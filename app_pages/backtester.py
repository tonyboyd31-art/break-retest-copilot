"""Backtesting Lab: chronological replay of the strategy with conservative fills and IS/OOS evaluation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.backtesting import run_backtest, structural_sensitivity, threshold_sensitivity
from core.config import diff_strategy
from data import repositories as repo
from services.backtest_service import config_from_settings, run_and_save, runs_frame
from ui import charts, state
from ui import components as ui

s = state.settings()
tz = state.tz()
db = state.db()
ui.page_header(
    "Backtesting Lab",
    "Test the strategy on past data using only what was knowable at each bar. Results are hypothetical.",
    eyebrow="Research",
    accent="Lab",
)

ctx = state.market_context(process=False)
md = ctx.md
if not md.ok:
    st.error(md.error or "No data.")
    st.stop()
spec = md.spec
st.markdown(
    f"Testing **{spec.symbol} {md.contract or ''}** - {md.source_label}, {len(md.segment):,} bars "
    f"({md.segment.index[0]:%Y-%m-%d} -> {md.segment.index[-1]:%Y-%m-%d} UTC). The replay clock is ignored: "
    "the full contract segment is replayed chronologically."
)
if md.feed_type == "simulated":
    st.warning(
        "SYNTHETIC DATA - useful for checking mechanics only. It says nothing about real-market profitability. "
        "Import real historical candles (More -> Settings -> Data sources) for research."
    )

tab_run, tab_sens, tab_saved = st.tabs(["Run backtest", "Sensitivity analysis", "Saved runs"])

with tab_run:
    with st.form("bt_form"):
        st.markdown("**Strategy overrides for this run** (defaults = your current settings)")
        a1, a2, a3, a4 = st.columns(4)
        min_score = a1.number_input("Minimum score", 0.0, 100.0, float(s.scoring.min_score), step=5.0)
        min_rr = a2.number_input("Minimum R:R", 0.5, 10.0, float(s.planning.min_rr), step=0.25)
        left = a3.number_input("Swing left bars", 1, 20, s.swings.left_bars)
        right = a4.number_input("Swing right bars", 1, 20, s.swings.right_bars)
        b1, b2, b3, b4 = st.columns(4)
        entry_m = b1.selectbox(
            "Entry method",
            ["confirmation_close", "zone_edge"],
            index=["confirmation_close", "zone_edge"].index(s.planning.entry_method),
        )
        target_m = b2.selectbox(
            "Target method",
            ["structure", "fixed_r"],
            index=["structure", "fixed_r"].index(s.planning.target_method),
        )
        stop_buf = b3.number_input(
            "Stop buffer (ticks)", 0.0, 100.0, float(s.planning.stop_buffer_ticks), step=1.0
        )
        windows_on = b4.toggle("Session windows", value=s.sessions.signal_windows_enabled)
        st.markdown("**Execution model**")
        c1, c2, c3, c4 = st.columns(4)
        comm = c1.number_input(
            "Commission / contract RT", 0.0, 100.0, float(s.risk.commission_for(spec.symbol)), step=0.1
        )
        slip = c2.number_input(
            "Slippage ticks / side", 0.0, 20.0, float(s.risk.slippage_ticks_per_side), step=0.5
        )
        qty = c3.number_input("Contracts", 1, 100, s.backtest.quantity)
        flat = c4.text_input("Force flat at (ET, blank = none)", s.backtest.force_flat_time or "")
        d1, d2, d3, d4 = st.columns(4)
        sim_be = d1.toggle(
            "Simulate breakeven rule",
            value=s.backtest.simulate_breakeven,
            help="Hypothetical rule for research: stop moves to entry from the bar AFTER +1R.",
        )
        be_r = d2.number_input(
            "BE trigger (R)", 0.25, 5.0, float((s.management.breakeven_thresholds_r or [1.0])[0]), step=0.25
        )
        sim_trail = d3.toggle("Simulate swing trailing", value=s.backtest.simulate_trailing)
        oos = d4.slider("Out-of-sample fraction", 0.0, 0.6, float(s.backtest.oos_fraction), 0.05)
        name = st.text_input("Run name", f"{spec.symbol} score>={min_score:g} rr>={min_rr:g}")
        go = st.form_submit_button("Run backtest", type="primary")
    if go:
        try:
            s2 = (
                s.updated("scoring", min_score=min_score)
                .updated(
                    "planning",
                    min_rr=min_rr,
                    entry_method=entry_m,
                    target_method=target_m,
                    stop_buffer_ticks=stop_buf,
                )
                .updated("swings", left_bars=int(left), right_bars=int(right))
                .updated("sessions", signal_windows_enabled=windows_on)
            )
            cfg = config_from_settings(
                s2,
                spec.symbol,
                commission_rt=comm,
                slippage_ticks_per_side=slip,
                quantity=int(qty),
                force_flat_time=flat.strip() or None,
                simulate_breakeven=sim_be,
                breakeven_r=be_r,
                simulate_trailing=sim_trail,
                oos_fraction=oos,
            )
            with st.spinner("Replaying..."):
                res, run_id = run_and_save(db, md, s2, cfg, name)
            st.session_state["bt_last"] = run_id
            st.success(f"Saved as run #{run_id}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {exc}")
    run_id = st.session_state.get("bt_last")
    if run_id:
        run = repo.get_backtest(db, run_id)
        trades = repo.get_backtest_trades(db, run_id)
        if run:
            for w in run["metrics"].get("warnings", []):
                st.warning(w)
            mets = run["metrics"]
            cols = st.columns(3)
            for col, (lbl, key) in zip(
                cols,
                [("All trades", "all"), ("In-sample", "in_sample"), ("Out-of-sample", "out_of_sample")],
                strict=True,
            ):
                m = mets.get(key, {})
                col.markdown(f"**{lbl}**")
                col.dataframe(
                    pd.DataFrame(
                        [
                            ("Trades", m.get("trades")),
                            ("Win rate", ui.pct(m.get("win_rate"))),
                            ("Net P&L", ui.money(m.get("net_pnl"))),
                            ("Gross profit", ui.money(m.get("gross_profit"))),
                            ("Gross loss", ui.money(m.get("gross_loss"))),
                            ("Profit factor", m.get("profit_factor")),
                            ("Avg win", ui.money(m.get("avg_win"))),
                            ("Avg loss", ui.money(m.get("avg_loss"))),
                            ("Expectancy / trade", ui.money(m.get("expectancy"))),
                            ("Expectancy (R)", m.get("expectancy_r")),
                            ("Max drawdown", ui.money(m.get("max_drawdown"))),
                            ("Max losing streak", m.get("max_losing_streak")),
                            ("Avg holding (min)", m.get("avg_holding_minutes")),
                            ("Ambiguous bars (counted as losses)", m.get("ambiguous_bars")),
                        ],
                        columns=["metric", "value"],
                    ).astype(str),
                    hide_index=True,
                    width="stretch",
                )
            if mets.get("oos_cutoff"):
                st.caption(
                    f"Out-of-sample = trades entered after {mets['oos_cutoff'][:16]} UTC (chronological split)."
                )
            if not trades.empty:
                eq = trades.sort_values("exit_ts")
                g1, g2 = st.columns(2)
                g1.plotly_chart(
                    charts.line_chart(
                        eq["exit_ts"].dt.tz_convert(tz).dt.tz_localize(None),
                        eq["pnl_net"].cumsum(),
                        "Equity",
                        height=280,
                        title="Cumulative net P&L (hypothetical)",
                        zero_line=True,
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                g2.plotly_chart(
                    charts.histogram(trades["r_multiple"], "R-multiple distribution", "R"),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                bds = run.get("breakdowns", {})
                bcols = st.columns(2)
                for i, (k, lbl) in enumerate(
                    [
                        ("session", "By session"),
                        ("score_bucket", "By score bucket"),
                        ("vol_regime", "By volatility regime"),
                        ("trend_regime", "By trend regime"),
                        ("exit_reason", "By exit reason"),
                        ("direction", "By direction"),
                    ]
                ):
                    df = pd.DataFrame(bds.get(k, []))
                    if not df.empty:
                        bcols[i % 2].plotly_chart(
                            charts.bar_breakdown(df, k, "expectancy_r", f"{lbl} - expectancy R", height=230),
                            width="stretch",
                            config={"displayModeBar": False},
                            key=f"bd_{k}",
                        )
                show = trades.copy()
                for c in ("entry_ts", "exit_ts"):
                    show[c] = show[c].dt.tz_convert(tz).dt.strftime("%m-%d %H:%M")
                st.dataframe(
                    show[
                        [
                            "entry_ts",
                            "direction",
                            "entry_price",
                            "stop",
                            "target",
                            "exit_ts",
                            "exit_price",
                            "exit_reason",
                            "r_multiple",
                            "pnl_net",
                            "score",
                            "session",
                            "sample",
                            "ambiguous",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    height=320,
                )
            diffs = diff_strategy(run["settings"], s)
            if diffs:
                st.info(
                    "Your current settings differ from this run's settings: "
                    + "; ".join(diffs[:8])
                    + (" ..." if len(diffs) > 8 else "")
                )

with tab_sens:
    st.warning(
        "Overfitting warning: every parameter combination you try increases the chance that the 'best' cell "
        "is luck. Prefer broad regions where neighbouring cells behave similarly, judge on OUT-OF-SAMPLE "
        "results, and distrust cells with fewer than ~30 trades."
    )
    st.markdown("**Threshold grid (score x R:R)** - one scan, re-filtered (fast).")
    t1, t2 = st.columns(2)
    scores = t1.multiselect("Minimum scores", [60, 65, 70, 75, 80, 85, 90], default=[70, 75, 80, 85, 90])
    rrs = t2.multiselect("Minimum R:R", [1.0, 1.5, 2.0, 2.5, 3.0], default=[1.5, 2.0, 2.5, 3.0])
    if st.button("Run threshold grid", type="primary") and scores and rrs:
        cfg = config_from_settings(s, spec.symbol)
        with st.spinner("Scanning once and re-simulating each cell..."):
            base = run_backtest(md.segment, spec, s, cfg)
            grid = threshold_sensitivity(base.scan, md.segment, spec, s, cfg, sorted(scores), sorted(rrs))
        st.session_state["sens_grid"] = grid
        st.session_state["sens_combos"] = len(scores) * len(rrs)
    grid = st.session_state.get("sens_grid")
    if grid is not None and not grid.empty:
        st.caption(
            f"{st.session_state.get('sens_combos')} combinations evaluated. Cell text: expectancy R and trade count."
        )
        h1, h2 = st.columns(2)
        for col, sample in ((h1, "IS"), (h2, "OOS")):
            sub = grid[grid["sample"] == sample]
            col.plotly_chart(
                charts.heatmap(
                    sub,
                    "min_rr",
                    "min_score",
                    "expectancy_r",
                    f"{'In' if sample == 'IS' else 'Out-of'}-sample expectancy (R)",
                    text_col="trades",
                ),
                width="stretch",
                config={"displayModeBar": False},
                key=f"hm_{sample}",
            )
        st.dataframe(grid, hide_index=True, width="stretch")
    st.markdown("**Structural grid (swing right bars x zone tolerance)** - re-scans every cell (slower).")
    u1, u2 = st.columns(2)
    rbs = u1.multiselect("Swing right bars", [2, 3, 4, 5], default=[2, 3, 4])
    tols = u2.multiselect("Zone tolerance (ATR)", [0.15, 0.25, 0.35, 0.5], default=[0.15, 0.25, 0.35])
    if st.button("Run structural grid") and rbs and tols:
        cfg = config_from_settings(s, spec.symbol)
        bar = st.progress(0.0, text="Re-scanning...")
        sg = structural_sensitivity(
            md.segment,
            spec,
            s,
            cfg,
            sorted(rbs),
            sorted(tols),
            progress=lambda f: bar.progress(f, text=f"Re-scanning... {f:.0%}"),
        )
        bar.empty()
        st.session_state["struct_grid"] = sg
    sg = st.session_state.get("struct_grid")
    if sg is not None and not sg.empty:
        k1, k2 = st.columns(2)
        for col, sample in ((k1, "IS"), (k2, "OOS")):
            sub = sg[sg["sample"] == sample]
            col.plotly_chart(
                charts.heatmap(
                    sub,
                    "tolerance_atr",
                    "right_bars",
                    "expectancy_r",
                    f"{sample} expectancy (R)",
                    text_col="trades",
                ),
                width="stretch",
                config={"displayModeBar": False},
                key=f"shm_{sample}",
            )
        st.dataframe(sg, hide_index=True, width="stretch")

with tab_saved:
    runs = runs_frame(db)
    if runs.empty:
        st.caption("No saved runs.")
    else:
        current = s.strategy_hash()
        runs["matches current settings"] = runs["settings_hash"] == current
        st.dataframe(runs, hide_index=True, width="stretch")
        rid = st.selectbox("Open run", runs["id"].tolist(), key="bt_open")
        o1, o2 = st.columns(2)
        if o1.button("Show in 'Run backtest' tab"):
            st.session_state["bt_last"] = int(rid)
            st.rerun()
        if o2.button("Delete run"):
            repo.delete_backtest(db, int(rid))
            st.rerun()
        if (~runs["matches current settings"]).any():
            st.info(
                "Runs marked False used different strategy settings than your current configuration - their "
                "results do not describe the current setup."
            )
