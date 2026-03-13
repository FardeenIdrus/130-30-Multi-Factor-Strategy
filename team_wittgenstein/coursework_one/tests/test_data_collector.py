"""Tests for modules.input.data_collector."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as req_lib

from modules.input.data_collector import DataFetcher


@pytest.fixture
def fetcher(mock_minio_conn):
    """DataFetcher with mocked MinIO."""
    return DataFetcher(mock_minio_conn)


# ===================================================================
# _reshape_price_df
# ===================================================================


class TestReshapePriceDf:

    def test_valid_transform(self):
        raw = pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-01", periods=5),
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [98.0] * 5,
                "Close": [102.0] * 5,
                "Adj Close": [102.0] * 5,
                "Volume": [1_000_000] * 5,
            }
        ).set_index("Date")

        result = DataFetcher._reshape_price_df(raw, "AAPL")
        assert result is not None
        assert "symbol" in result.columns
        assert "trade_date" in result.columns
        assert "close_price" in result.columns
        assert "currency" in result.columns
        assert (result["symbol"] == "AAPL").all()
        assert len(result) == 5

    def test_multiindex_columns(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["AAPL"] * 6,
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        data = [[100, 105, 98, 102, 102, 1e6]] * 3
        raw = pd.DataFrame(data, index=dates, columns=index)
        raw.index.name = "Date"

        result = DataFetcher._reshape_price_df(raw, "AAPL")
        assert result is not None
        assert "close_price" in result.columns

    def test_none_input(self):
        assert DataFetcher._reshape_price_df(None, "AAPL") is None

    def test_empty_input(self):
        assert DataFetcher._reshape_price_df(pd.DataFrame(), "AAPL") is None


# ===================================================================
# _classify_missing
# ===================================================================


class TestClassifyMissing:

    @patch("modules.input.data_collector.utils.yf")
    def test_delisted(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["ABMD"])
        assert "ABMD" in result["delisted"]
        assert result["fetch_error"] == []

    @patch("modules.input.data_collector.utils.yf")
    def test_fetch_error(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._classify_missing(["AAPL"])
        assert "AAPL" in result["fetch_error"]
        assert result["delisted"] == []

    @patch("modules.input.data_collector.utils.yf")
    def test_exception(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("API error")

        result = fetcher._classify_missing(["BAD"])
        assert "BAD" in result["fetch_error"]


# ===================================================================
# _is_cached
# ===================================================================


class TestIsCached:

    def test_cached_true(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = True
        assert fetcher._is_cached("prices", "AAPL") is True

    def test_cached_false_missing_ctl(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.side_effect = [True, False]
        assert fetcher._is_cached("prices", "AAPL") is False


# ===================================================================
# fetch_prices
# ===================================================================


class TestFetchPrices:

    def test_all_cached(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = True
        cached_df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "trade_date": pd.bdate_range("2024-01-01", periods=5),
                "close_price": [150.0] * 5,
            }
        )
        mock_minio_conn.download_dataframe.return_value = cached_df

        with patch("modules.input.data_collector.prices.yf") as mock_yf:
            result = fetcher.fetch_prices(["AAPL"])
            mock_yf.download.assert_not_called()

        assert len(result) == 5

    @patch("modules.input.data_collector.prices.yf")
    def test_downloads_uncached(self, mock_yf, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        raw = pd.DataFrame(
            {
                "Date": pd.bdate_range("2024-01-01", periods=5),
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [98.0] * 5,
                "Close": [102.0] * 5,
                "Adj Close": [102.0] * 5,
                "Volume": [1e6] * 5,
            }
        ).set_index("Date")
        mock_yf.download.return_value = raw

        result = fetcher.fetch_prices(["AAPL"])
        mock_yf.download.assert_called_once()
        assert len(result) > 0
        assert "symbol" in result.columns


# ===================================================================
# fetch_fundamentals
# ===================================================================


class TestFetchFundamentals:

    def test_fetch(self, fetcher, mock_minio_conn):
        mock_minio_conn.object_exists.return_value = False

        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [3e11],
            }
        )

        with patch.object(fetcher, "_fetch_single_fundamental", return_value=fake_df):
            result = fetcher.fetch_fundamentals(["AAPL"], period="1y", source="simfin")

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"


# ===================================================================
# _fetch_single_fundamental
# ===================================================================


class TestCacheHelpers:

    def test_ctl_path(self, fetcher):
        assert fetcher._ctl_path("prices", "AAPL") == "prices/AAPL.ctl"

    def test_parquet_path(self, fetcher):
        assert fetcher._parquet_path("prices", "AAPL") == "prices/AAPL.parquet"

    def test_write_ctl(self, fetcher, mock_minio_conn):
        fetcher._write_ctl("prices", "AAPL", 100, "yfinance")
        mock_minio_conn.upload_json.assert_called_once()
        call_args = mock_minio_conn.upload_json.call_args[0]
        assert call_args[1] == "prices/AAPL.ctl"
        assert call_args[2]["rows"] == 100

    def test_cache_dataframe(self, fetcher, mock_minio_conn):
        df = pd.DataFrame({"col": [1, 2]})
        fetcher._cache_dataframe("prices", "AAPL", df, "yfinance")
        mock_minio_conn.upload_dataframe.assert_called_once()
        mock_minio_conn.upload_json.assert_called_once()

    def test_load_cached(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_dataframe.return_value = pd.DataFrame({"col": [1]})
        result = fetcher._load_cached("prices", "AAPL")
        assert len(result) == 1

    def test_mark_loaded(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = {
            "name": "AAPL",
            "loaded_to_postgres": False,
        }
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_called_once()
        uploaded = mock_minio_conn.upload_json.call_args[0][2]
        assert uploaded["loaded_to_postgres"] is True

    def test_mark_loaded_no_ctl(self, fetcher, mock_minio_conn):
        mock_minio_conn.download_json.return_value = None
        fetcher.mark_loaded("prices", "AAPL")
        mock_minio_conn.upload_json.assert_not_called()


class TestFetchSingleFundamental:

    def test_simfin_source(self, fetcher):
        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "total_debt": [1e11],
                "net_income": [2e10],
                "book_equity": [1e11],
                "shares_outstanding": [15_000_000_000],
                "eps": [1.30],
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )

        with patch.object(fetcher, "_fetch_simfin_fundamentals", return_value=fake_df):
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="simfin"
            )

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["total_assets"] == 3e11
        assert result.iloc[0]["net_income"] == 2e10

    def test_waterfall_source_delegates(self, fetcher):
        fake_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )

        with patch.object(
            fetcher, "_fetch_waterfall_fundamentals", return_value=fake_df
        ) as mock_wf:
            result = fetcher._fetch_single_fundamental(
                "AAPL", period="5y", source="waterfall"
            )
            mock_wf.assert_called_once_with("AAPL", "5y")

        assert result is not None
        assert len(result) == 1


# ===================================================================
# _normalize_fundamentals_source
# ===================================================================


class TestNormalizeFundamentalsSource:

    def test_waterfall(self):
        assert DataFetcher._normalize_fundamentals_source("waterfall") == "waterfall"

    def test_simfin(self):
        assert DataFetcher._normalize_fundamentals_source("simfin") == "simfin"

    def test_default_is_waterfall(self):
        assert DataFetcher._normalize_fundamentals_source(None) == "waterfall"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            DataFetcher._normalize_fundamentals_source("bloomberg")


# ===================================================================
# _resolve_cik
# ===================================================================


class TestEdgarGetJson:

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_retries_on_failure(self, mock_get, fetcher):
        """Should retry on request failure and succeed on third attempt."""
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("timeout")
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {"data": "ok"}
        mock_get.side_effect = [
            req_lib.RequestException("timeout"),
            req_lib.RequestException("timeout"),
            ok_resp,
        ]
        result = fetcher._edgar_get_json("http://example.com", max_retries=3)
        assert result == {"data": "ok"}
        assert mock_get.call_count == 3

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_returns_none_on_404_when_allowed(self, mock_get, fetcher):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        result = fetcher._edgar_get_json("http://example.com", allow_not_found=True)
        assert result is None

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_retries_on_429(self, mock_get, fetcher):
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "0"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {"ok": True}
        mock_get.side_effect = [rate_resp, ok_resp]
        result = fetcher._edgar_get_json("http://example.com", max_retries=3)
        assert result == {"ok": True}
        assert mock_get.call_count == 2

    @patch("modules.input.data_collector.edgar.requests.get")
    def test_returns_none_after_all_retries_fail(self, mock_get, fetcher):
        mock_get.side_effect = req_lib.RequestException("fail")
        result = fetcher._edgar_get_json("http://example.com", max_retries=2)
        assert result is None


class TestResolveCik:

    def test_resolves_known_ticker(self, fetcher):
        with patch.object(
            fetcher,
            "_edgar_get_json",
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
            },
        ):
            cik = fetcher._resolve_cik("AAPL")
        assert cik == "0000320193"

    def test_returns_none_for_unknown(self, fetcher):
        with patch.object(
            fetcher,
            "_edgar_get_json",
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            },
        ):
            assert fetcher._resolve_cik("ZZZZ") is None

    def test_returns_none_on_network_error(self, fetcher):
        with patch.object(fetcher, "_edgar_get_json", return_value=None):
            assert fetcher._resolve_cik("AAPL") is None

    def test_caches_ticker_map(self, fetcher):
        mock_json = MagicMock(
            return_value={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            }
        )
        with patch.object(fetcher, "_edgar_get_json", mock_json):
            fetcher._resolve_cik("AAPL")
            fetcher._resolve_cik("AAPL")
        # Should only fetch the ticker map once
        assert mock_json.call_count == 1


# ===================================================================
# _edgar_fetch_concept
# ===================================================================


class TestEdgarFetchConcept:

    def test_parses_concept_data(self, fetcher):
        payload = {
            "units": {
                "USD": [
                    {
                        "end": "2024-03-30",
                        "start": "2024-01-01",
                        "val": 3.5e11,
                        "filed": "2024-05-01",
                        "form": "10-Q",
                    },
                    {
                        "end": "2024-06-30",
                        "start": "2024-04-01",
                        "val": 3.6e11,
                        "filed": "2024-08-01",
                        "form": "10-Q",
                    },
                ]
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_fetch_concept("0000320193", "Assets")
        assert len(result) == 2
        assert result.iloc[0]["val"] == 3.5e11 or result.iloc[0]["val"] == 3.6e11

    def test_returns_empty_on_not_found(self, fetcher):
        with patch.object(fetcher, "_edgar_get_json", return_value=None):
            result = fetcher._edgar_fetch_concept("0000320193", "Assets")
        assert result.empty

    def test_filters_by_cutoff(self, fetcher):
        payload = {
            "units": {
                "USD": [
                    {
                        "end": "2020-03-30",
                        "val": 1e11,
                        "filed": "2020-05-01",
                        "form": "10-Q",
                    },
                    {
                        "end": "2024-03-30",
                        "val": 3.5e11,
                        "filed": "2024-05-01",
                        "form": "10-Q",
                    },
                ]
            }
        }
        with patch.object(fetcher, "_edgar_get_json", return_value=payload):
            result = fetcher._edgar_fetch_concept(
                "0000320193", "Assets", cutoff=pd.Timestamp("2023-01-01")
            )
        assert len(result) == 1
        assert result.iloc[0]["val"] == 3.5e11


# ===================================================================
# _fetch_edgar_fundamentals
# ===================================================================


class TestFetchEdgarFundamentals:

    def test_happy_path(self, fetcher):
        """Mock at method level — resolve CIK, fiscal periods, and concepts."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        concept_df = pd.DataFrame(
            {
                "end": ["2024-03-30"],
                "start": ["2024-01-01"],
                "val": [3.5e11],
                "filed": [pd.Timestamp("2024-05-01")],
            }
        )
        empty_concept = pd.DataFrame(columns=["end", "start", "val", "filed"])

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher,
            "_edgar_fetch_concept",
            side_effect=[
                concept_df.copy(),  # Assets
                concept_df.assign(val=2e10),  # NetIncomeLoss
                empty_concept,  # ProfitLoss fallback (not needed)
                concept_df.assign(val=1e11),  # Equity primary
                empty_concept,  # Equity fallback
                empty_concept,  # shares tag 1
                empty_concept,  # shares tag 2
                empty_concept,  # EPS diluted
                empty_concept,  # EPS basic
                concept_df.assign(val=5e10),  # LongTermDebt
            ],
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        assert not result.empty
        assert result.iloc[0]["total_assets"] == 3.5e11
        assert result.iloc[0]["source"] == "edgar"

    def test_no_cik(self, fetcher):
        fetcher._ticker_to_cik = {"MSFT": "0000789019"}
        result = fetcher._fetch_edgar_fundamentals("ZZZZ")
        assert result.empty

    def test_debt_fallback_to_tier2(self, fetcher):
        """LongTermDebt empty → falls back to LongTermDebtAndCapitalLeaseObligations."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30"],
                "report_date": [pd.Timestamp("2024-03-30")],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
            }
        )
        val_df = pd.DataFrame(
            {
                "end": ["2024-03-30"],
                "start": ["2024-01-01"],
                "val": [1e11],
                "filed": [pd.Timestamp("2024-05-01")],
            }
        )
        empty = pd.DataFrame(columns=["end", "start", "val", "filed"])
        debt_lease = val_df.assign(val=8e10)

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher,
            "_edgar_fetch_concept",
            side_effect=[
                val_df.copy(),  # Assets
                val_df.copy(),  # NetIncomeLoss (has data → ProfitLoss NOT called)
                val_df.copy(),  # Equity primary
                empty,  # Equity fallback
                empty,
                empty,  # shares
                empty,
                empty,  # EPS diluted, EPS basic
                empty,  # LongTermDebt (empty!)
                debt_lease,  # LongTermDebtAndCapitalLeaseObligations (tier 2)
            ],
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        assert not result.empty
        assert result.iloc[0]["total_debt"] == 8e10

    def test_ytd_to_standalone_conversion(self, fetcher):
        """net_income should be de-cumulated from YTD to standalone quarterly."""
        periods = pd.DataFrame(
            {
                "report_date_str": ["2024-03-30", "2024-06-30", "2024-09-30"],
                "report_date": pd.to_datetime(
                    ["2024-03-30", "2024-06-30", "2024-09-30"]
                ),
                "fiscal_year": [2024, 2024, 2024],
                "fiscal_quarter": [1, 2, 3],
            }
        )
        empty = pd.DataFrame(columns=["end", "start", "val", "filed"])
        # YTD net_income: Q1=10, Q1+Q2=25, Q1+Q2+Q3=40
        ni_concept = pd.DataFrame(
            {
                "end": ["2024-03-30", "2024-06-30", "2024-09-30"],
                "start": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "val": [10, 25, 40],
                "filed": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-11-01"]),
            }
        )

        with patch.object(
            fetcher, "_resolve_cik", return_value="0000320193"
        ), patch.object(
            fetcher, "_edgar_get_fiscal_periods", return_value=periods
        ), patch.object(
            fetcher,
            "_edgar_fetch_concept",
            side_effect=[
                empty,  # Assets
                ni_concept,  # NetIncomeLoss (has data → ProfitLoss NOT called)
                empty,
                empty,  # Equity primary, Equity fallback
                empty,
                empty,  # shares (2 tags)
                empty,
                empty,  # EPS diluted, EPS basic
                empty,  # LongTermDebt (empty → tier 2)
                empty,  # LongTermDebtAndCapitalLeaseObligations (empty → tier 3)
                empty,
                empty,  # LongTermDebtNoncurrent, LongTermDebtCurrent
            ],
        ):
            result = fetcher._fetch_edgar_fundamentals("AAPL")

        result = result.sort_values("fiscal_quarter")
        # Standalone: Q1=10, Q2=15, Q3=15
        assert list(result["net_income"]) == [10, 15, 15]


# ===================================================================
# Cache TTL
# ===================================================================


class TestCacheExpiry:

    def test_expired_cache_returns_false(self, fetcher):
        fetcher.cache_ttl_days = 7
        fetcher.minio.object_exists.return_value = True
        fetcher.minio.download_json.return_value = {
            "fetched_at": (pd.Timestamp.now() - pd.DateOffset(days=10)).isoformat()
        }
        assert fetcher._is_cached("prices", "AAPL") is False

    def test_fresh_cache_returns_true(self, fetcher):
        fetcher.cache_ttl_days = 7
        fetcher.minio.object_exists.return_value = True
        fetcher.minio.download_json.return_value = {
            "fetched_at": pd.Timestamp.now().isoformat()
        }
        assert fetcher._is_cached("prices", "AAPL") is True

    def test_no_ttl_always_cached(self, fetcher):
        fetcher.cache_ttl_days = None
        fetcher.minio.object_exists.return_value = True
        assert fetcher._is_cached("prices", "AAPL") is True


# ===================================================================
# _fetch_yfinance_fundamentals
# ===================================================================


class TestFetchYfinanceFundamentals:

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_happy_path(self, mock_yf, fetcher):
        dates = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]

        bs = pd.DataFrame(
            {"2024-03-31": [3e11, 1e11, 1e11], "2024-06-30": [3.2e11, 1.1e11, 1.2e11]},
            index=["Total Assets", "Total Debt", "Stockholders Equity"],
        )
        bs.columns = dates

        inc = pd.DataFrame(
            {"2024-03-31": [2e10, 1.3], "2024-06-30": [2.2e10, 1.4]},
            index=["Net Income", "Diluted EPS"],
        )
        inc.columns = dates

        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = bs
        mock_ticker.quarterly_income_stmt = inc
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert not result.empty
        assert len(result) == 2
        assert result.iloc[0]["source"] == "yfinance"
        assert "total_assets" in result.columns
        assert "net_income" in result.columns

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_empty_returns_empty(self, mock_yf, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert result.empty

    @patch("modules.input.data_collector.yfinance_fundamentals.yf")
    def test_exception_returns_empty(self, mock_yf, fetcher):
        mock_yf.Ticker.side_effect = Exception("API error")
        result = fetcher._fetch_yfinance_fundamentals("AAPL")
        assert result.empty


# ===================================================================
# _merge_waterfall
# ===================================================================


class TestMergeWaterfall:

    def test_fills_nulls_from_lower_priority(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [None],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [2.9e11],  # lower priority, should not override
                "net_income": [2e10],  # fills null from edgar
                "book_equity": [9e10],  # lower priority, should not override
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )

        result = fetcher._merge_waterfall([("edgar", edgar_df), ("simfin", simfin_df)])
        assert len(result) == 1
        row = result.iloc[0]
        assert row["total_assets"] == 3e11  # kept from edgar
        assert row["net_income"] == 2e10  # filled from simfin
        assert row["book_equity"] == 1e11  # kept from edgar
        assert row["source"] == "edgar"

    def test_adds_new_quarters_from_lower_priority(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [2],  # new quarter
                "report_date": [pd.Timestamp("2024-06-30")],
                "total_assets": [3.1e11],
                "source": ["simfin"],
            }
        )

        result = fetcher._merge_waterfall([("edgar", edgar_df), ("simfin", simfin_df)])
        assert len(result) == 2

    def test_single_source(self, fetcher):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "total_assets": [3e11],
                "source": ["edgar"],
            }
        )
        result = fetcher._merge_waterfall([("edgar", df)])
        assert len(result) == 1


# ===================================================================
# _forward_fill_fundamentals
# ===================================================================


class TestForwardFillFundamentals:

    def test_fills_from_previous_quarter(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "fiscal_year": [2024, 2024],
                "fiscal_quarter": [1, 2],
                "total_assets": [3e11, None],
                "net_income": [2e10, 2.1e10],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert result.iloc[1]["total_assets"] == 3e11  # forward filled
        assert result.iloc[1]["net_income"] == 2.1e10  # not overwritten

    def test_no_fill_across_symbols(self):
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "fiscal_year": [2024, 2024],
                "fiscal_quarter": [1, 1],
                "total_assets": [3e11, None],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert pd.isna(result[result["symbol"] == "MSFT"].iloc[0]["total_assets"])

    def test_handles_empty(self):
        assert DataFetcher._forward_fill_fundamentals(None) is None
        result = DataFetcher._forward_fill_fundamentals(pd.DataFrame())
        assert result.empty

    def test_caps_at_2_quarters(self):
        """Forward fill should stop after 2 consecutive nulls."""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 4,
                "fiscal_year": [2024, 2024, 2024, 2024],
                "fiscal_quarter": [1, 2, 3, 4],
                "total_assets": [3e11, None, None, None],
            }
        )
        result = DataFetcher._forward_fill_fundamentals(df)
        assert result.iloc[1]["total_assets"] == 3e11  # filled (1st)
        assert result.iloc[2]["total_assets"] == 3e11  # filled (2nd)
        assert pd.isna(result.iloc[3]["total_assets"])  # NOT filled (3rd, beyond limit)


# ===================================================================
# _fetch_waterfall_fundamentals (integration)
# ===================================================================


class TestFetchWaterfallFundamentals:

    def test_uses_all_sources(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [None],
                "book_equity": [None],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )
        simfin_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "net_income": [2e10],
                "book_equity": [None],
                "currency": ["USD"],
                "source": ["simfin"],
            }
        )
        yf_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["yfinance"],
            }
        )

        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=edgar_df
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=simfin_df
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=yf_df
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is not None
        row = result.iloc[0]
        assert row["total_assets"] == 3e11  # from edgar
        assert row["net_income"] == 2e10  # from simfin
        assert row["book_equity"] == 1e11  # from yfinance

    def test_all_sources_empty(self, fetcher):
        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=pd.DataFrame()
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is None

    def test_edgar_only(self, fetcher):
        edgar_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "fiscal_year": [2024],
                "fiscal_quarter": [1],
                "report_date": [pd.Timestamp("2024-03-31")],
                "total_assets": [3e11],
                "net_income": [2e10],
                "book_equity": [1e11],
                "currency": ["USD"],
                "source": ["edgar"],
            }
        )

        with patch.object(
            fetcher, "_fetch_edgar_fundamentals", return_value=edgar_df
        ), patch.object(
            fetcher, "_fetch_simfin_fundamentals", return_value=pd.DataFrame()
        ), patch.object(
            fetcher, "_fetch_yfinance_fundamentals", return_value=pd.DataFrame()
        ):
            result = fetcher._fetch_waterfall_fundamentals("AAPL", "5y")

        assert result is not None
        assert len(result) == 1
