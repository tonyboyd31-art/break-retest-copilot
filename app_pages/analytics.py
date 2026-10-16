"""Analytics: backtest results, hypothetical signal outcomes and ACTUAL journaled performance - kept separate."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.performance import breakdown, compute_metrics
from core.timeutils import from_iso, session_label
from data import repositories as repo
from services.backtest_service import evaluate_outcomes, runs_frame
from ui import charts, state
from ui import components as ui

s = state.settings()
tz = state.tz()
db = state.db()
ui.page_header(
    "Analytics",
    "Three separate sources that are never mixed: backtests, hypothetical signal outcomes, "
    "and your actual recorded trades.",
)

tab_bt, tab_sig, tab_real = st.tabs(
    ["Backtest runs (hypothetical)", "Signal outcomes (hypothetical)", "My trades (actual, user-entered)"]
)


def metric_block(m: dict) -> None:
    c = st.columns(6)
    c[0].metric("Trades", m.get("trades", 0))
    c[1].metric("Win rate", ui.pct(m.get("win_rate")))
    c[2].metric("Net P&L", ui.money(m.get("net_pnl")))
    c[3].metric("Profit factor", "-" if m.get("profit_factor") is None else f"{m['profit_factor']}")
    c[4].metric("Expectancy (R)", "-" if m.get("expectancy_r") is None else f"{m['expectancy_r']:+.3f}")
    c[5].metric("Max drawdown", ui.money(m.get("max_drawdown")))


with tab_bt:
    runs = runs_frame(db)
    if runs.empty:
        st.info("No backtests yet - run one in the Backtesting Lab.")
    else:
        rid = st.selectbox(
            "Run",
            runs["id"].tolist(),
            format_func=lambda i: f"#{i} " + str(runs.loc[runs["id"] == i, "name"].iloc[0]),
            key="an_run",
        )
        run = repo.get_backtest(db, int(rid))
        trades = repo.get_backtest_trades(db, int(rid))
        if run["settings_hash"] != s.strategy_hash():
            st.info("This run used different strategy settings than your current configuration.")
        for w in run["metrics"].get("warnings", []):
            st.caption(w)
        metric_block(run["metrics"].get("all", {}))
        if not trades.empty:
            eq = trades.sort_values("exit_ts")
            cum = eq["pnl_net"].cumsum()
            dd = cum - cum.cummax().clip(lower=0)
            x = eq["exit_ts"].dt.tz_convert(tz).dt.tz_localize(None)
            g1, g2 = st.columns(2)
            g1.plotly_chart(
                charts.line_chart(
                    x, cum, "Equity", title="Equity curve (hypothetical, USD)", height=260, zero_line=True
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            g2.plotly_chart(
                charts.line_chart(
                    x,
                    dd,
                    "Drawdown",
                    title="Drawdown from peak (USD)",
                    height=260,
                    color="#e66767",
                    zero_line=True,
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            cols = st.columns(2)
            for i, k in enumerate(["score_bucket", "session", "vol_regime", "trend_regime"]):
                bd = breakdown(trades, k)
                if not bd.empty:
                    cols[i % 2].plotly_chart(
                        charts.bar_breakdown(bd, k, "expectancy_r", f"Expectancy R by {k}", height=220),
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"an_{k}",
                    )
            st.caption(
                "Score buckets show whether higher-ranked setups performed differently IN THIS SAMPLE. A rising "
                "pattern here does not make the score a probability."
            )

with tab_sig:
    st.caption(
        "Hypothetical: what would have happened if each QUALIFIED setup's proposed plan had been followed "
        "mechanically (conservative fills). These are not trades and not account profit."
    )
    ctx = state.market_context(process=False)
    if ctx.md.ok and ctx.scan is not None and st.button("Evaluate outcomes for the current data"):
        n = evaluate_outcomes(db, ctx.md, ctx.scan, s)
        st.success(f"Evaluated {n} qualified setup(s).")
    out = repo.list_signal_outcomes(db)
    if out.empty:
        st.info("No hypothetical outcomes evaluated yet.")
    else:
        closed = out[
            out["outcome"].isin(
                ["target", "target_gap", "stop", "stop_gap", "stop_ambiguous", "session_flat", "time"]
            )
        ].copy()
        closed["pnl_r"] = closed["r_result"]
        if not closed.empty:
            m = compute_metrics(closed.assign(pnl_net=closed["r_result"]), "pnl_net", "r_result")
            c = st.columns(4)
            c[0].metric("Resolved signals", m["trades"])
            c[1].metric("Hit rate", ui.pct(m["win_rate"]))
            c[2].metric("Avg outcome (R)", f"{m['expectancy_r']:+.3f}")
            c[3].metric("Ambiguous (counted as stop)", int(closed["ambiguous"].sum()))
        st.dataframe(
            out[
                [
                    "setup_id",
                    "instrument",
                    "direction",
                    "score",
                    "confirm_ts",
                    "outcome",
                    "r_result",
                    "mfe_r",
                    "mae_r",
                    "bars_held",
                    "ambiguous",
                    "context",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

with tab_real:
    st.caption(
        "Only trades you registered and closed by recording exits (or for which you entered the broker "
        "statement P&L). This is the only tab that reflects real results - and only as accurately as your "
        "entries."
    )
    trades = repo.list_trades(db, status="closed")
    rows = []
    for t_ in trades:
        if t_["realized_pnl"] is None:
            continue
        risk_pts = abs(t_["entry_price"] - t_["initial_stop"])
        try:
            spec = state.app().registry.get(t_["instrument"])
            risk_usd = risk_pts * spec.point_value * t_["quantity"]
        except KeyError:
            risk_usd = None
        rows.append(
            {
                "id": t_["id"],
                "instrument": t_["instrument"],
                "direction": t_["direction"],
                "entry_ts": from_iso(t_["entry_ts"]),
                "exit_ts": from_iso(t_["closed_ts"]),
                "pnl_net": t_["realized_pnl"],
                "r_multiple": (t_["realized_pnl"] / risk_usd) if risk_usd else None,
                "session": session_label(from_iso(t_["entry_ts"])),
                "setup linked": bool(t_["setup_id"]),
                "pnl_source": t_["pnl_source"],
                "rule_adherence": t_["rule_adherence"] or "n/a",
            }
        )
    if not rows:
        st.info("No closed trades with realized P&L yet.")
    else:
        df = pd.DataFrame(rows)
        metric_block(compute_metrics(df))
        x = df.sort_values("exit_ts")
        st.plotly_chart(
            charts.line_chart(
                x["exit_ts"].dt.tz_convert(tz).dt.tz_localize(None),
                x["pnl_net"].cumsum(),
                "Equity",
                title="Cumulative realized P&L (user-entered)",
                height=260,
                zero_line=True,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        cols = st.columns(2)
        for i, k in enumerate(["session", "instrument", "setup linked", "rule_adherence"]):
            bd = breakdown(df.assign(**{k: df[k].astype(str)}), k)
            if not bd.empty:
                cols[i % 2].plotly_chart(
                    charts.bar_breakdown(bd, k, "expectancy", f"Avg P&L by {k}", height=220),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"real_{k}",
                )
        st.dataframe(df, hide_index=True, width="stretch")
