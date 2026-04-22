"""Step 1: Download MSCI USA Index monthly returns.

Uses the iShares MSCI USA ETF (EUSA) as a proxy for the MSCI USA Index,
which the ETF tracks closely with negligible tracking error.
"""

import logging
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MSCI_USA_TICKER = "EUSA"  # iShares MSCI USA ETF — tracks MSCI USA Index


def fetch_benchmark_monthly_returns(
    start_date: date,
    end_date: date,
) -> pd.Series:
    """Download MSCI USA monthly returns for the backtest window.

    Args:
        start_date: First date to include (one month before first return needed,
                    so pct_change has a base price).
        end_date:   Last date to include.

    Returns:
        Series indexed by month-end date (datetime.date) with monthly returns.
        Index represents the end of the return period.
    """
    # Download daily data and resample to avoid gaps in yfinance monthly feed
    raw = yf.download(
        MSCI_USA_TICKER,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No benchmark data downloaded for {MSCI_USA_TICKER} "
            f"({start_date} → {end_date})"
        )

    close = raw["Close"].squeeze().sort_index()

    # Last trading day of each calendar month
    month_end_prices = close.resample("ME").last().dropna()
    monthly_returns = month_end_prices.pct_change().dropna()

    # Normalise index to calendar month-end dates
    monthly_returns.index = (
        pd.DatetimeIndex(monthly_returns.index).to_period("M").to_timestamp("M").date
    )

    logger.info(
        "Benchmark (%s): %d monthly returns | %s → %s",
        MSCI_USA_TICKER,
        len(monthly_returns),
        monthly_returns.index[0],
        monthly_returns.index[-1],
    )
    return monthly_returns
