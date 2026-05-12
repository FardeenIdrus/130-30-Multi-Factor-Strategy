# 130/30 Multi-Factor Equity Strategy

Sector-neutral 130/30 long-short multi-factor equity strategy with full data
infrastructure, walk-forward backtesting across 23 scenarios, and an interactive
Streamlit dashboard.

## What this is

A systematic equity strategy combining Value, Quality, Momentum, and Low Volatility
factors. Long-short construction with 130% long / 30% short exposure, sector-neutral
within 11 GICS sectors, IC-weighted composite scoring on a 36-month rolling window,
EWMA volatility-adjusted weighting, and walk-forward backtesting with realistic
transaction cost modelling.

Built end-to-end: data ingestion from four independent sources (Yahoo Finance, SEC
EDGAR, SimFin, OECD), Dockerised storage stack (PostgreSQL, MongoDB, MinIO),
factor engine, portfolio construction, backtest engine, and 542 unit tests at
99% coverage.

## My role

This started as the technical implementation for an 8-person UCL coursework project
(Team Wittgenstein, Big Data in Quantitative Finance, 2025-26). I was the sole
developer responsible for the codebase:

- **Data infrastructure (CW1):** Full ETL pipeline — DataFetcher with waterfall
  fallback across yfinance, EDGAR, SimFin and OECD; DataValidator with strict-mode
  blocking on schema, completeness, and coverage checks; DataWriter with idempotent
  writes to PostgreSQL, MinIO Parquet cache with CTL companion files, and MongoDB
  audit logging. Docker Compose stack, APScheduler cron triggers, GitHub Actions CI.

- **Factor engine (CW2):** Per-sector winsorisation (5th/95th), cross-sectional
  z-scores, Low Vol orthogonalisation against Momentum, IC-weighted composite
  scoring with 36-month rolling Spearman ICs and zero-floor on negative ICs.

- **Portfolio builder (CW2):** Top/bottom 10% selection per sector with 10-20%
  buffer zone and 3-month max hold, EWMA volatility-adjusted weighting (λ=0.94),
  liquidity cap at 5% of 20-day ADTV with pro-rata redistribution, no-trade zone
  at ±1% weight change.

- **Backtest engine (CW2):** Walk-forward simulation with no look-ahead, gross/net
  returns with one-way transaction costs (25 bps baseline) and short-borrow charges
  (0.75% annual), 23 scenarios (baseline + 3 cost + 4 factor exclusion + 15
  parameter sensitivity), resume support for interrupted runs.

- **Testing & docs:** 542 unit tests at 99% coverage, Sphinx documentation, full
  pipeline README. The non-coding team owned strategy design, market analysis, and academic write-up.

## Stack

Python 3.10+ · PostgreSQL · MongoDB · MinIO · Docker · Poetry · pandas · NumPy ·
Streamlit · pytest · Sphinx · GitHub Actions

## Running it

See [`team_wittgenstein/coursework_two/README.md`](team_wittgenstein/coursework_two/README.md)
for the full pipeline and dashboard instructions. Quick-start with the committed
seed dump takes ~30 seconds; full pipeline run is ~2-3 hours.
