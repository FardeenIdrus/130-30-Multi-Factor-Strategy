"""Tests for backtest_engine — Steps 3-6 of the 130/30 backtest.

All tests are pure-Python (no DB, no network). The DB-dependent
orchestrator (run_backtest) is not tested here.
"""

import pandas as pd
import pytest

from modules.backtest.backtest_engine import (
    _compute_drift_adjusted_weights,
    _compute_gross_return,
    _compute_stock_returns,
    _compute_turnover,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positions(symbols, directions, weights):
    return pd.DataFrame(
        {"symbol": symbols, "direction": directions, "final_weight": weights}
    )


def _returns(symbols, values):
    return pd.Series(values, index=symbols)


# ---------------------------------------------------------------------------
# TestComputeStockReturns
# ---------------------------------------------------------------------------


class TestComputeStockReturns:

    def test_basic_return(self):
        """(110/100) - 1 = 0.10 for a single stock."""
        pos = _positions(["A"], ["long"], [0.5])
        price_t = pd.Series({"A": 100.0})
        price_t1 = pd.Series({"A": 110.0})
        result = _compute_stock_returns(pos, price_t, price_t1)
        assert pytest.approx(result["A"], rel=1e-9) == 0.10

    def test_missing_price_excluded(self):
        """Symbols missing a price at either date are dropped."""
        pos = _positions(["A", "B"], ["long", "long"], [0.5, 0.5])
        price_t = pd.Series({"A": 100.0})  # B missing
        price_t1 = pd.Series({"A": 110.0, "B": 200.0})
        result = _compute_stock_returns(pos, price_t, price_t1)
        assert "A" in result.index
        assert "B" not in result.index

    def test_zero_price_excluded(self):
        """Symbols with price_t == 0 are dropped to avoid division by zero."""
        pos = _positions(["A"], ["long"], [0.5])
        price_t = pd.Series({"A": 0.0})
        price_t1 = pd.Series({"A": 10.0})
        result = _compute_stock_returns(pos, price_t, price_t1)
        assert result.empty

    def test_negative_return(self):
        """Price decline produces negative return."""
        pos = _positions(["A"], ["long"], [0.5])
        price_t = pd.Series({"A": 200.0})
        price_t1 = pd.Series({"A": 150.0})
        result = _compute_stock_returns(pos, price_t, price_t1)
        assert pytest.approx(result["A"], rel=1e-9) == -0.25


# ---------------------------------------------------------------------------
# TestComputeGrossReturn
# ---------------------------------------------------------------------------


class TestComputeGrossReturn:

    def test_long_only(self):
        """Gross = Σ(w_long × r). Short book absent."""
        pos = _positions(["A", "B"], ["long", "long"], [0.65, 0.65])
        rets = _returns(["A", "B"], [0.10, 0.20])
        gross, long_ret, short_ret = _compute_gross_return(pos, rets)
        assert pytest.approx(long_ret, rel=1e-9) == 0.65 * 0.10 + 0.65 * 0.20
        assert pytest.approx(short_ret, rel=1e-9) == 0.0
        assert pytest.approx(gross, rel=1e-9) == long_ret

    def test_gross_equals_long_minus_short(self):
        """gross = long_ret - short_ret (spec Step 3)."""
        pos = _positions(
            ["A", "B", "C"],
            ["long", "long", "short"],
            [0.65, 0.65, 0.30],
        )
        rets = _returns(["A", "B", "C"], [0.10, 0.05, 0.08])
        gross, long_ret, short_ret = _compute_gross_return(pos, rets)
        assert pytest.approx(gross, rel=1e-9) == long_ret - short_ret

    def test_short_rising_hurts_portfolio(self):
        """When a short position rises, gross return falls."""
        # Long flat, short rises 10% → short contribution hurts gross
        pos = _positions(["A", "B"], ["long", "short"], [1.30, 0.30])
        rets = _returns(["A", "B"], [0.0, 0.10])
        gross, long_ret, short_ret = _compute_gross_return(pos, rets)
        assert gross < 0
        assert pytest.approx(gross, rel=1e-9) == -0.30 * 0.10

    def test_short_falling_helps_portfolio(self):
        """When a short position falls, gross return is positive."""
        pos = _positions(["A", "B"], ["long", "short"], [1.30, 0.30])
        rets = _returns(["A", "B"], [0.0, -0.10])
        gross, _, _ = _compute_gross_return(pos, rets)
        assert gross > 0
        assert pytest.approx(gross, rel=1e-9) == 0.30 * 0.10

    def test_missing_return_symbol_dropped(self):
        """Positions without a return are silently excluded."""
        pos = _positions(["A", "B"], ["long", "long"], [0.65, 0.65])
        rets = _returns(["A"], [0.10])  # B has no return
        gross, long_ret, _ = _compute_gross_return(pos, rets)
        assert pytest.approx(long_ret, rel=1e-9) == 0.65 * 0.10

    def test_130_30_market_neutral_scenario(self):
        """130/30 with equal long/short returns → gross ≈ net market return."""
        # All stocks return 5%; net exposure = 1.0 → gross ≈ 5%
        pos = _positions(
            ["A", "B"],
            ["long", "short"],
            [1.30, 0.30],
        )
        rets = _returns(["A", "B"], [0.05, 0.05])
        gross, _, _ = _compute_gross_return(pos, rets)
        assert pytest.approx(gross, rel=1e-9) == 1.30 * 0.05 - 0.30 * 0.05


# ---------------------------------------------------------------------------
# TestComputeDriftAdjustedWeights
# ---------------------------------------------------------------------------


class TestComputeDriftAdjustedWeights:

    def test_weights_preserve_scale(self):
        """Drift-adjusted weights preserve the original weight scale (unnormalised)."""
        prev = _positions(["A", "B", "C"], ["long"] * 3, [0.50, 0.30, 0.20])
        rets = _returns(["A", "B", "C"], [0.0, 0.0, 0.0])
        w = _compute_drift_adjusted_weights(prev, rets)
        # With zero returns, drift weights == original weights, sum = 1.0
        assert pytest.approx(w.sum(), abs=1e-9) == 1.0

    def test_no_returns_weights_unchanged(self):
        """With zero returns, drift-adjusted weights equal original weights."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.60, 0.40])
        rets = _returns(["A", "B"], [0.0, 0.0])
        w = _compute_drift_adjusted_weights(prev, rets)
        assert pytest.approx(w["A"], rel=1e-9) == 0.60
        assert pytest.approx(w["B"], rel=1e-9) == 0.40

    def test_higher_return_increases_relative_weight(self):
        """Stock with higher return gets larger drift-adjusted weight."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.50, 0.50])
        rets = _returns(["A", "B"], [0.20, 0.0])
        w = _compute_drift_adjusted_weights(prev, rets)
        assert w["A"] > w["B"]

    def test_missing_return_treated_as_zero(self):
        """Symbol missing from returns gets r=0 (price unchanged)."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.50, 0.50])
        rets = _returns(["A"], [0.10])  # B missing → r=0
        w = _compute_drift_adjusted_weights(prev, rets)
        # A: 0.50 * 1.10 = 0.55, B: 0.50 * 1.0 = 0.50
        assert pytest.approx(w["A"], rel=1e-9) == 0.55
        assert pytest.approx(w["B"], rel=1e-9) == 0.50
        assert w["A"] > w["B"]

    def test_formula_correctness(self):
        """Drift-adjusted weight = final_weight × (1 + r) — unnormalised."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.60, 0.40])
        rets = _returns(["A", "B"], [0.10, 0.20])
        w = _compute_drift_adjusted_weights(prev, rets)
        assert pytest.approx(w["A"], rel=1e-9) == 0.60 * 1.10
        assert pytest.approx(w["B"], rel=1e-9) == 0.40 * 1.20


# ---------------------------------------------------------------------------
# TestComputeTurnover
# ---------------------------------------------------------------------------


class TestComputeTurnover:

    def test_first_period_no_previous(self):
        """First period: turnover = Σ final_weight (entering from zero)."""
        pos = _positions(["A", "B"], ["long", "short"], [1.30, 0.30])
        rets = _returns(["A", "B"], [0.05, 0.05])
        turnover = _compute_turnover(pos, None, rets)
        assert pytest.approx(turnover, rel=1e-9) == 1.60

    def test_identical_portfolio_after_drift(self):
        """If new weights exactly match drift-adjusted weights, turnover = 0."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.50, 0.50])
        rets = _returns(["A", "B"], [0.0, 0.0])  # no drift
        # new weights = same as drift-adjusted
        curr = _positions(["A", "B"], ["long"] * 2, [0.50, 0.50])
        turnover = _compute_turnover(curr, prev, rets)
        assert pytest.approx(turnover, abs=1e-9) == 0.0

    def test_full_replacement(self):
        """Completely replacing all positions generates maximum turnover."""
        prev = _positions(["A", "B"], ["long"] * 2, [0.65, 0.65])
        rets = _returns(["A", "B"], [0.0, 0.0])
        curr = _positions(["C", "D"], ["long"] * 2, [0.65, 0.65])
        turnover = _compute_turnover(curr, prev, rets)
        # drift: A=0.65, B=0.65 (zero returns → no drift)
        # exit A(0.65) + exit B(0.65) + enter C(0.65) + enter D(0.65) = 2.60
        assert pytest.approx(turnover, rel=1e-9) == 2.60

    def test_partial_rebalance(self):
        """Partial weight shift produces correct absolute difference."""
        prev = _positions(["A"], ["long"], [0.50])
        rets = _returns(["A"], [0.0])  # no drift → w'_A = 0.50
        curr = _positions(["A"], ["long"], [0.60])
        turnover = _compute_turnover(curr, prev, rets)
        assert pytest.approx(turnover, rel=1e-9) == 0.10

    def test_drift_reduces_turnover(self):
        """When a stock drifts toward new target, turnover < naive weight diff."""
        # Stock A rises 20% → drifts from 0.50 to higher; new target also higher
        prev = _positions(["A", "B"], ["long"] * 2, [0.50, 0.50])
        rets = _returns(["A", "B"], [0.20, 0.0])
        # After drift: A ~ 0.545, B ~ 0.455
        curr = _positions(["A", "B"], ["long"] * 2, [0.55, 0.45])
        turnover_with_drift = _compute_turnover(curr, prev, rets)

        # Without drift (naive): |0.55-0.50| + |0.45-0.50| = 0.10
        turnover_naive = abs(0.55 - 0.50) + abs(0.45 - 0.50)
        assert turnover_with_drift < turnover_naive

    def test_spec_example(self):
        """Spec example: Turnover=7.9%, short=30% → trading cost=0.020%."""
        # With turnover=0.079: trading_cost = 0.079 * 0.0025 = 0.0001975 ≈ 0.020%
        # short_notional=0.30: borrow_cost = 0.30 * 0.0075/12 = 0.0001875 ≈ 0.019%
        # total ≈ 0.039% = 0.00039
        trading_cost = 0.079 * 0.0025
        borrow_cost = 0.30 * 0.0075 / 12
        total = trading_cost + borrow_cost
        assert pytest.approx(trading_cost * 100, abs=0.001) == 0.020
        assert pytest.approx(borrow_cost * 100, abs=0.001) == 0.019
        assert pytest.approx(total * 100, abs=0.001) == 0.039
