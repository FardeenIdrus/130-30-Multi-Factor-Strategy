"""Tests for benchmark.py — mocked yfinance, no network calls."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from modules.backtest.benchmark import fetch_benchmark_monthly_returns


def _make_raw(prices: dict) -> pd.DataFrame:
    """Build a minimal yf.download-style DataFrame with a Close column."""
    idx = pd.to_datetime(list(prices.keys()))
    close = pd.Series(list(prices.values()), index=idx, name="EUSA")
    return pd.DataFrame({"Close": close})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START = date(2023, 1, 1)
END = date(2023, 3, 31)

# Two month-end prices → one monthly return
_TWO_MONTHS = _make_raw(
    {
        "2023-01-31": 100.0,
        "2023-02-28": 110.0,
    }
)

# Three month-end prices → two monthly returns
_THREE_MONTHS = _make_raw(
    {
        "2023-01-31": 100.0,
        "2023-02-28": 110.0,
        "2023-03-31": 99.0,
    }
)


# ---------------------------------------------------------------------------
# TestFetchBenchmarkMonthlyReturns
# ---------------------------------------------------------------------------


class TestFetchBenchmarkMonthlyReturns:

    def test_returns_correct_monthly_return(self):
        """100 → 110 gives a +10% monthly return."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert len(result) == 1
        assert pytest.approx(result.iloc[0], rel=1e-9) == 0.10

    def test_index_is_calendar_month_end_dates(self):
        """Index entries are datetime.date objects at calendar month-end."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert isinstance(result.index[0], date)
        assert result.index[0] == date(2023, 2, 28)

    def test_multiple_months_returns_correct_count(self):
        """Three month-end prices produce two monthly returns."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_THREE_MONTHS
        ):
            result = fetch_benchmark_monthly_returns(START, END)
        assert len(result) == 2

    def test_negative_return(self):
        """110 → 99 gives a negative monthly return."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_THREE_MONTHS
        ):
            result = fetch_benchmark_monthly_returns(START, END)
        assert result.iloc[-1] < 0
        assert pytest.approx(result.iloc[-1], rel=1e-9) == 99.0 / 110.0 - 1

    def test_empty_download_raises(self):
        """Empty yfinance response raises ValueError."""
        empty = pd.DataFrame()
        with patch("modules.backtest.benchmark.yf.download", return_value=empty):
            with pytest.raises(ValueError, match="No benchmark data"):
                fetch_benchmark_monthly_returns(START, END)

    def test_passes_correct_dates_to_yfinance(self):
        """start_date and end_date are forwarded as strings to yf.download."""
        with patch(
            "modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS
        ) as mock_dl:
            fetch_benchmark_monthly_returns(START, END)
        call_kwargs = mock_dl.call_args
        assert call_kwargs.kwargs["start"] == "2023-01-01"
        assert call_kwargs.kwargs["end"] == "2023-03-31"

    def test_returns_series(self):
        """Return type is a pandas Series."""
        with patch("modules.backtest.benchmark.yf.download", return_value=_TWO_MONTHS):
            result = fetch_benchmark_monthly_returns(START, END)
        assert isinstance(result, pd.Series)
