"""
tests/test_underlying_data.py — UnderlyingDataStore unit tests.

All tests mock yfinance.download() to avoid network dependency.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_yf_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """Build a minimal yfinance-style DataFrame with DatetimeIndex."""
    idx = pd.to_datetime(dates).tz_localize("UTC")
    df = pd.DataFrame(
        {"Close": closes, "Open": closes, "High": closes, "Low": closes,
         "Volume": [1_000_000] * len(closes)},
        index=idx,
    )
    df.index.name = "Date"
    return df


def _make_store(tmp_path: Path):
    from orb_live.data.underlying_data import UnderlyingDataStore
    return UnderlyingDataStore(tmp_path / "underlyings")


# ── get() ─────────────────────────────────────────────────────────────────────

class TestGet:
    def test_returns_empty_when_no_file(self, tmp_path):
        store = _make_store(tmp_path)
        df = store.get("QQQ")
        assert df.empty
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_returns_sorted_data_after_write(self, tmp_path):
        store = _make_store(tmp_path)
        # Write unsorted parquet directly.
        dates  = pd.to_datetime(["2022-01-06", "2022-01-04", "2022-01-05"])
        closes = [100.0, 98.0, 99.0]
        df = pd.DataFrame({"date": dates, "open": closes, "high": closes,
                           "low": closes, "close": closes,
                           "volume": [1e6] * 3})
        df.to_parquet(store._path("QQQ"), index=False)

        result = store.get("QQQ")
        assert list(result["date"]) == sorted(result["date"].tolist())
        assert len(result) == 3


# ── update_one() ──────────────────────────────────────────────────────────────

class TestUpdateOne:
    def test_writes_parquet_and_returns_row_count(self, tmp_path):
        store = _make_store(tmp_path)
        yf_df = _make_yf_df(
            ["2022-01-03", "2022-01-04", "2022-01-05"],
            [380.0, 382.0, 381.0],
        )
        with patch("yfinance.download", return_value=yf_df) as mock_dl:
            n = store.update_one("QQQ", lookback_days=10)

        assert n == 3
        mock_dl.assert_called_once()
        result = store.get("QQQ")
        assert len(result) == 3
        assert list(result.columns[:2]) == ["date", "open"]

    def test_deduplicates_on_repeated_update(self, tmp_path):
        store = _make_store(tmp_path)
        yf_df = _make_yf_df(
            ["2022-01-03", "2022-01-04", "2022-01-05"],
            [380.0, 382.0, 381.0],
        )
        with patch("yfinance.download", return_value=yf_df):
            store.update_one("QQQ")
            n2 = store.update_one("QQQ")  # same data — expect 0 new rows

        assert n2 == 0
        assert len(store.get("QQQ")) == 3

    def test_appends_new_rows(self, tmp_path):
        store = _make_store(tmp_path)
        old = _make_yf_df(["2022-01-03", "2022-01-04"], [380.0, 382.0])
        with patch("yfinance.download", return_value=old):
            store.update_one("QQQ")

        new = _make_yf_df(
            ["2022-01-03", "2022-01-04", "2022-01-05"],
            [380.0, 382.0, 383.0],
        )
        with patch("yfinance.download", return_value=new):
            n = store.update_one("QQQ")

        assert n == 1
        assert len(store.get("QQQ")) == 3

    def test_returns_zero_on_empty_yf_response(self, tmp_path):
        store = _make_store(tmp_path)
        # Empty every attempt → exhausts retries → 0 rows (no crash).
        with patch("orb_live.data.underlying_data.time.sleep"), \
             patch("yfinance.download", return_value=pd.DataFrame()) as mock_dl:
            n = store.update_one("QQQ")
        assert n == 0
        assert mock_dl.call_count == 4          # _YF_MAX_ATTEMPTS

    def test_retries_then_succeeds_on_transient_empty(self, tmp_path):
        """A transient empty response is retried and the later good data is kept."""
        store = _make_store(tmp_path)
        good = _make_yf_df(["2022-01-03", "2022-01-04"], [380.0, 382.0])
        seq  = [pd.DataFrame(), pd.DataFrame(), good]  # fail, fail, succeed
        with patch("orb_live.data.underlying_data.time.sleep") as mock_sleep, \
             patch("yfinance.download", side_effect=seq) as mock_dl:
            n = store.update_one("QQQ")
        assert n == 2
        assert mock_dl.call_count == 3
        assert mock_sleep.call_count == 2       # slept before each retry

    def test_retries_on_exception_then_succeeds(self, tmp_path):
        """An exception (e.g. rate-limit) is treated as transient and retried."""
        store = _make_store(tmp_path)
        good = _make_yf_df(["2022-01-03"], [380.0])
        seq  = [RuntimeError("YFRateLimitError"), good]
        with patch("orb_live.data.underlying_data.time.sleep"), \
             patch("yfinance.download", side_effect=seq) as mock_dl:
            n = store.update_one("QQQ")
        assert n == 1
        assert mock_dl.call_count == 2

    def test_crypto_ticker_resolved(self, tmp_path):
        store = _make_store(tmp_path)
        yf_df = _make_yf_df(["2022-01-03"], [40000.0])
        with patch("yfinance.download", return_value=yf_df) as mock_dl:
            store.update_one("BTC", lookback_days=5)
        # Verify yfinance was called with the resolved ticker "BTC-USD".
        call_args = mock_dl.call_args
        assert call_args[0][0] == "BTC-USD" or call_args[1].get("tickers") == "BTC-USD" or \
               "BTC-USD" in str(call_args)


# ── Holiday alignment ─────────────────────────────────────────────────────────

class TestHolidayAlignment:
    """
    Verify that the backtest PS filter holiday alignment is replicated:
    ul_df[ul_df["date"] < pd.Timestamp(date_)].tail(2)

    Crypto has weekend bars; equity does not.  The store returns the raw
    history and check_prior_session_filter takes the tail(2) naturally.
    """

    def _build_store_with_data(self, tmp_path, dates, closes):
        store = _make_store(tmp_path)
        yf_df = _make_yf_df(dates, closes)
        with patch("yfinance.download", return_value=yf_df):
            store.update_one("QQQ")
        return store

    def test_tail2_excludes_trade_date(self, tmp_path):
        """The trade date itself must NOT appear in prior_rows."""
        dates  = ["2022-01-03", "2022-01-04", "2022-01-05",
                  "2022-01-06", "2022-01-07"]
        closes = [380.0, 382.0, 381.0, 383.0, 390.0]
        store = self._build_store_with_data(tmp_path, dates, closes)

        df = store.get("QQQ")
        trade_date = pd.Timestamp("2022-01-07")
        prior_rows = df[df["date"] < trade_date].tail(2)

        assert len(prior_rows) == 2
        assert prior_rows.iloc[-1]["date"] == pd.Timestamp("2022-01-06")
        assert prior_rows.iloc[-2]["date"] == pd.Timestamp("2022-01-05")
        # 2022-01-07 (trade date) must be excluded
        assert all(prior_rows["date"] < trade_date)

    def test_crypto_weekend_bars_included(self, tmp_path):
        """
        Crypto data includes Sat/Sun.  tail(2) returns the two most recent
        closes before the trade date, regardless of being a weekend.
        """
        dates  = ["2022-01-01", "2022-01-02", "2022-01-03"]  # Sat, Sun, Mon
        closes = [45000.0, 46000.0, 47000.0]
        store = _make_store(tmp_path)
        yf_df = _make_yf_df(dates, closes)
        with patch("yfinance.download", return_value=yf_df):
            store.update_one("BTC")

        df = store.get("BTC")
        trade_date = pd.Timestamp("2022-01-03")
        prior_rows = df[df["date"] < trade_date].tail(2)

        # Should include Sat + Sun (weekend bars for crypto)
        assert len(prior_rows) == 2
        assert prior_rows.iloc[-1]["date"] == pd.Timestamp("2022-01-02")

    def test_fewer_than_two_prior_rows(self, tmp_path):
        """With only one prior close, filter logic must handle gracefully."""
        dates  = ["2022-01-06"]
        closes = [383.0]
        store = self._build_store_with_data(tmp_path, dates, closes)

        df = store.get("QQQ")
        trade_date = pd.Timestamp("2022-01-07")
        prior_rows = df[df["date"] < trade_date].tail(2)

        assert len(prior_rows) == 1  # only one available


# ── check_freshness() ────────────────────────────────────────────────────────

class TestCheckFreshness:
    """Freshness gate: equity must be through prev_trading_day, crypto through yesterday."""

    def _write_parquet(self, store, ul: str, dates: list[str], closes: list[float]):
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1_000_000] * len(closes),
        })
        df.to_parquet(store._path(ul), index=False)

    def test_stale_equity_flagged(self, tmp_path):
        """Equity UL missing the prior trading day must appear in issues."""
        store = _make_store(tmp_path)
        # Session 2026-06-23 (Tue); prev_trading_day = 2026-06-22.
        # Latest bar is 2026-06-20 (Fri) — stale.
        self._write_parquet(store, "QQQ", ["2026-06-19", "2026-06-20"], [450.0, 451.0])
        issues = store.check_freshness(date(2026, 6, 23), {"QQQ"})
        assert issues, "Stale equity UL must be flagged"
        assert any("QQQ" in m for m in issues)

    def test_fresh_equity_passes(self, tmp_path):
        """Equity UL current through prev_trading_day must pass."""
        store = _make_store(tmp_path)
        self._write_parquet(store, "QQQ", ["2026-06-19", "2026-06-20", "2026-06-22"], [450.0, 451.0, 453.0])
        issues = store.check_freshness(date(2026, 6, 23), {"QQQ"})
        assert not issues, f"Fresh equity must pass, got: {issues}"

    def test_stale_crypto_flagged(self, tmp_path):
        """Crypto UL missing yesterday (calendar day) must appear in issues."""
        store = _make_store(tmp_path)
        # Session 2026-06-23 (Tue); required = 2026-06-22.
        # Latest bar is 2026-06-21 (Sun) — stale.
        self._write_parquet(store, "BTC", ["2026-06-20", "2026-06-21"], [64000.0, 64500.0])
        issues = store.check_freshness(date(2026, 6, 23), {"BTC"})
        assert issues, "Stale crypto UL must be flagged"
        assert any("BTC" in m for m in issues)

    def test_fresh_crypto_passes(self, tmp_path):
        """Crypto UL current through yesterday must pass."""
        store = _make_store(tmp_path)
        self._write_parquet(store, "BTC", ["2026-06-20", "2026-06-21", "2026-06-22"], [64000.0, 64500.0, 65000.0])
        issues = store.check_freshness(date(2026, 6, 23), {"BTC"})
        assert not issues, f"Fresh crypto must pass, got: {issues}"

    def test_missing_parquet_flagged(self, tmp_path):
        """UL with no parquet on disk must be flagged."""
        store = _make_store(tmp_path)
        issues = store.check_freshness(date(2026, 6, 23), {"QQQ"})
        assert issues
        assert any("no data" in m for m in issues)


# ── warn_if_stale() ───────────────────────────────────────────────────────────

class TestWarnIfStale:
    def test_fresh_data_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        yf_df = _make_yf_df([str(date.today())], [100.0])
        with patch("yfinance.download", return_value=yf_df):
            store.update_one("QQQ")
        result = store.warn_if_stale("QQQ", date.today(), max_age_days=3)
        assert result is None

    def test_empty_store_returns_warning(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.warn_if_stale("QQQ", date.today())
        assert result is not None
        assert "no data" in result

    def test_old_data_returns_warning(self, tmp_path):
        from datetime import timedelta
        store = _make_store(tmp_path)
        old_date = str(date.today() - timedelta(days=10))
        yf_df = _make_yf_df([old_date], [100.0])
        with patch("yfinance.download", return_value=yf_df):
            store.update_one("QQQ")
        result = store.warn_if_stale("QQQ", date.today(), max_age_days=3)
        assert result is not None
        assert "days ago" in result
