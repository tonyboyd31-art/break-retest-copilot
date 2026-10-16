"""Deterministic strategy, risk and analytics engine.

Nothing in this package talks to a broker. Every function here is a pure
calculation over market data and user-entered records so it can be unit
tested and replayed historically without look-ahead.
"""
