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

    def test_raises_for_unknown_symbol(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(KeyError, match="UNKNOWN_XYZ"):
            store.get("UNKNOWN_XYZ")

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
        with patch("yfinance.download", return_value=pd.DataFrame()):
            n = store.update_one("QQQ")
        assert n == 0

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
