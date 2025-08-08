"""Unit tests for eval.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import math
import numpy as np
import os
import pandas as pd
import pytest
import re
from datetime import datetime
from pybroker.eval import (
    EvalMetrics,
    EvaluateMixin,
    annual_total_return_percent,
    avg_profit_loss,
    bca_boot_conf,
    calmar_ratio,
    conf_profit_factor,
    conf_sharpe_ratio,
    drawdown_conf,
    iqr,
    largest_win_loss,
    max_drawdown,
    max_drawdown_percent,
    max_wins_losses,
    profit_factor,
    r_squared,
    relative_entropy,
    sharpe_ratio,
    sortino_ratio,
    total_profit_loss,
    total_return_percent,
    ulcer_index,
    upi,
    win_loss_rate,
    winning_losing_trades,
)
from typing import get_type_hints

np.random.seed(42)


@pytest.fixture(params=[0, 1, 2])
def value_type(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 10, 1000])
def rand_values(value_type, request):
    if not request.param:
        return np.empty(0)
    if value_type == 0:
        return np.zeros(request.param)
    elif value_type == 1:
        return np.ones(request.param)
    return np.random.rand(request.param)


@pytest.fixture(params=[True, False])
def calc_bootstrap(request):
    return request.param


@pytest.fixture()
def portfolio_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/portfolio_df.pkl")
    )


@pytest.fixture()
def trades_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/trades_df.pkl")
    )


def truncate(value, n):
    return math.floor(value * 10**n) / 10**n


@pytest.mark.parametrize(
    "n, n_boot, expected_msg",
    [
        (0, 100, "Bootstrap sample size must be greater than 0."),
        (-1, 100, "Bootstrap sample size must be greater than 0."),
        (10, 0, "Number of boostrap samples must be greater than 0."),
        (10, -1, "Number of boostrap samples must be greater than 0."),
    ],
)
def test_bca_boot_conf_when_invalid_params_then_error(n, n_boot, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        bca_boot_conf(np.random.rand(100), n, n_boot, profit_factor)


@pytest.mark.parametrize("n, n_boot", [(1, 100), (1, 1), (10, 100), (10, 1)])
def test_conf_profit_factor(n, n_boot, rand_values):
    intervals = conf_profit_factor(rand_values, n, n_boot)
    assert len(intervals) == 6


@pytest.mark.parametrize("n, n_boot", [(1, 100), (1, 1), (10, 100), (10, 1)])
def test_conf_sharpe_ratio(n, n_boot, rand_values):
    intervals = conf_sharpe_ratio(rand_values, n, n_boot)
    assert len(intervals) == 6


@pytest.mark.parametrize("n, n_boot", [(1, 100), (1, 1), (10, 100), (10, 1)])
def test_drawdown_conf(n, n_boot, rand_values):
    dd, dd_pct = drawdown_conf(rand_values * 1000, rand_values, n, n_boot)
    assert len(dd) == 4
    assert len(dd_pct) == 4


@pytest.mark.parametrize(
    "n, n_boot, expected_msg",
    [
        (0, 100, "Bootstrap sample size must be greater than 0."),
        (-1, 100, "Bootstrap sample size must be greater than 0."),
        (10, 0, "Number of boostrap samples must be greater than 0."),
        (10, -1, "Number of boostrap samples must be greater than 0."),
    ],
)
def test_drawdown_conf_when_invalid_params_then_error(n, n_boot, expected_msg):
    values = np.random.rand(100)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        drawdown_conf(values, values, n, n_boot)


def test_drawdown_conf_when_length_mismatch_then_error():
    with pytest.raises(
        ValueError,
        match=re.escape("Param changes length does not match returns length."),
    ):
        drawdown_conf(np.random.rand(100), np.random.rand(101), 10, 100)


@pytest.mark.parametrize(
    "values, expected_pf",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 1.499999),
        ([1, 1, 1, 1], 40000000001),
        ([1], 10000000001),
        ([-1], 0),
        ([0, 0, 0, 0], 0),
        ([], 0),
    ],
)
def test_profit_factor(values, expected_pf):
    pf = profit_factor(np.array(values))
    assert truncate(pf, 6) == truncate(expected_pf, 6)


@pytest.mark.parametrize(
    "values, obs, expected_sharpe",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.167443),
        (
            [0.1, -0.2, 0.3, 0, -0.4, 0.5],
            252,
            0.16744367165578425 * np.sqrt(252),
        ),
        ([1, 1, 1, 1], None, 0),
        ([1], None, 0),
        ([], None, 0),
    ],
)
def test_sharpe_ratio(values, obs, expected_sharpe):
    sharpe = sharpe_ratio(np.array(values), obs)
    assert truncate(sharpe, 6) == truncate(expected_sharpe, 6)


@pytest.mark.parametrize(
    "values, obs, expected_sortino",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.499999),
        (
            [0.1, -0.2, 0.3, 0, -0.4, 0.5],
            252,
            0.4999999999999999 * np.sqrt(252),
        ),
        ([1, 1, 1, 1], None, 0),
        ([1], None, 0),
        ([], None, 0),
    ],
)
def test_sortino_ratio(values, obs, expected_sortino):
    sortino = sortino_ratio(np.array(values), obs)
    assert truncate(sortino, 6) == truncate(expected_sortino, 6)


@pytest.mark.parametrize(
    "values, expected_dd",
    [
        ([0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], -0.4),
        ([0.1, -0.4], -0.4),
        ([-0.1], -0.1),
        ([1, 1, 1, 1], 0),
        ([1], 0),
        ([], 0),
    ],
)
def test_max_drawdown(values, expected_dd):
    changes = np.array(values)
    assert max_drawdown(changes) == expected_dd


@pytest.mark.parametrize(
    "values, bars_per_year, expected_calmar",
    [
        ([0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], 252, -9),
        ([0.1, -0.4], 252, -94.5),
        ([1, 1, 1, 1], 252, 0),
        ([1], 252, 0),
        ([], 252, 0),
    ],
)
def test_calmar_ratio(values, bars_per_year, expected_calmar):
    calmar = calmar_ratio(np.array(values), bars_per_year)
    assert truncate(calmar, 6) == expected_calmar


@pytest.mark.parametrize(
    "values, expected_dd, expected_index",
    [
        ([0, 0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], -36.25, 6),
        ([0, -0.2], -20, 1),
        ([-0.1], -10, 0),
        ([0, 0, 0, 0], 0, None),
        ([0], 0, None),
        ([], 0, None),
    ],
)
def test_max_drawdown_percent(values, expected_dd, expected_index):
    returns = np.array(values)
    dd, index = max_drawdown_percent(returns)
    assert round(dd, 2) == expected_dd
    if expected_index is None:
        assert index is None
    else:
        assert index == expected_index


@pytest.mark.parametrize(
    "values, expected_iqr",
    [
        ([1, 3, 5, 7, 8, 10, 11, 13], 6.5),
        ([1], 0),
        ([1, 2], 0),
        ([1, 1, 1, 1, 1], 0),
        ([], 0),
    ],
)
def test_iqr(values, expected_iqr):
    assert iqr(np.array(values)) == expected_iqr


@pytest.mark.parametrize(
    "values, expected_entropy",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.782775),
        ([1, 1, 1, 1], 0),
        ([1], 0),
        ([], 0),
    ],
)
def test_relative_entropy(values, expected_entropy):
    entropy = relative_entropy(np.array(values))
    assert truncate(entropy, 6) == expected_entropy


@pytest.mark.parametrize(
    "values, period, expected_ui",
    [
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 0.909259),
        ([100, 101, 102, 100, 99, 103, 103, 102], 1, 0),
        ([0, 0, 0, 0, 0], 2, 0),
        ([1, 1, 1, 1, 1], 2, 0),
        ([100], 14, 0),
        ([100], 1, 0),
        ([], 2, 0),
    ],
)
def test_ulcer_index(values, period, expected_ui):
    assert truncate(ulcer_index(np.array(values), period), 6) == expected_ui


@pytest.mark.parametrize(
    "values, period", [([100, 101, 102], 0), ([100, 101, 102], -1)]
)
def test_ulcer_index_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape("n needs to be >= 1.")):
        ulcer_index(np.array(values), period)


@pytest.mark.parametrize(
    "values, period, ui, expected_upi",
    [
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, None, 0.329757),
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 0, 0),
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 1, 0.299834),
        ([100, 101, 102, 100, 99, 103, 103, 102], 1, None, 0),
        ([0, 0, 0, 0, 0], 2, None, 0),
        ([1, 1, 1, 1, 1], 2, None, 0),
        ([100], 14, None, 0),
        ([100], 1, None, 0),
        ([], 2, None, 0),
        ([], 14, None, 0),
        ([], 14, 0, 0),
        ([], 14, 1.5, 0),
        ([100], 14, None, 0),
        ([100], 14, 0, 0),
        ([100], 14, 1.5, 0),
        ([100], 1, None, 0),
        ([100, 101], 14, None, 0),
        ([100, 101], 14, 0, 0),
        ([100, 101, 102], 2, 0, 0),
    ],
)
def test_upi(values, period, ui, expected_upi):
    upi_ = upi(np.array(values), period=period, ui=ui)
    assert truncate(upi_, 6) == expected_upi


@pytest.mark.parametrize(
    "values, period", [([100, 101, 102], 0), ([100, 101, 102], -1)]
)
def test_upi_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape("n needs to be >= 1.")):
        upi(np.array(values), period)


@pytest.mark.parametrize(
    "values, expected_win_rate, expected_loss_rate",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 70, 30),
        ([0.1], 100, 0),
        ([-0.1], 0, 100),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_win_loss_rate(values, expected_win_rate, expected_loss_rate):
    pnls = np.array(values)
    win_rate, loss_rate = win_loss_rate(pnls)
    assert win_rate == expected_win_rate
    assert loss_rate == expected_loss_rate


@pytest.mark.parametrize(
    "values, expected_winning_trades, expected_losing_trades",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 7, 3),
        ([0.1], 1, 0),
        ([-0.1], 0, 1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_winning_losing_trades(
    values, expected_winning_trades, expected_losing_trades
):
    pnls = np.array(values)
    winning_trades, losing_trades = winning_losing_trades(pnls)
    assert winning_trades == expected_winning_trades
    assert losing_trades == expected_losing_trades


@pytest.mark.parametrize(
    "values, expected_profit, expected_loss",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.9, -0.6),
        ([0, 0, 0, 0, 0], 0, 0),
        ([0.1], 0.1, 0),
        ([-0.1], 0, -0.1),
        ([], 0, 0),
    ],
)
def test_total_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = total_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss


@pytest.mark.parametrize(
    "values, expected_profit, expected_loss",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.3, -0.3),
        ([1, 1, 1, 1, 1], 1, 0),
        ([-1, -1, -1, -1, -1], 0, -1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_avg_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = avg_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss


@pytest.mark.parametrize(
    "values, expected_win, expected_loss",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.3, -0.4),
        ([1, 1, 1, 1, 1], 1, 0),
        ([-1, -1, -1, -1, -1], 0, -1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_largest_win_loss(values, expected_win, expected_loss):
    pnls = np.array(values)
    win, loss = largest_win_loss(pnls)
    assert win == expected_win
    assert loss == expected_loss


@pytest.mark.parametrize(
    "values, expected_wins, expected_losses",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 3, 2),
        ([1, 1, 1, 1, 1], 5, 0),
        ([-1, -1, -1, -1, -1], 0, 5),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_max_wins_losses(values, expected_wins, expected_losses):
    pnls = np.array(values)
    wins, losses = max_wins_losses(pnls)
    assert wins == expected_wins
    assert losses == expected_losses


@pytest.mark.parametrize(
    "values, expected_r2",
    [
        ([1, 3, 5, 7, 8, 10, 11, 13], 0.992907),
        ([1], 0),
        ([-1], 0),
        ([1, 1, 1, 1, 1], 0),
        ([0, 0, 0, 0, 0], 0),
        ([], 0),
    ],
)
def test_r_squared(values, expected_r2):
    r2 = r_squared(np.array(values))
    assert truncate(r2, 6) == expected_r2


@pytest.mark.parametrize(
    "initial_value, pnl, expected_return", [(100, 10, 10), (0, 10, 0)]
)
def test_total_return_percent(initial_value, pnl, expected_return):
    return_pct = total_return_percent(initial_value, pnl)
    assert truncate(return_pct, 2) == expected_return


@pytest.mark.parametrize(
    "initial_value, pnl, bars_per_year, total_bars, expected_return",
    [
        (100, 10, 252, 756, 3.22),
        (0, 10, 252, 756, 0),
        (100, 10, 252, 0, 0),
    ],
)
def test_annual_total_return_percent(
    initial_value, pnl, bars_per_year, total_bars, expected_return
):
    return_pct = annual_total_return_percent(
        initial_value, pnl, bars_per_year, total_bars
    )
    assert truncate(return_pct, 2) == expected_return


class TestEvaluateMixin:
    @pytest.mark.parametrize(
        "bars_per_year, expected_sharpe, expected_sortino",
        [
            (None, 0.026013464180574847, 0.02727734785007549),
            (
                252,
                0.026013464180574847 * np.sqrt(252),
                0.02727734785007549 * np.sqrt(252),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "bootstrap_sample_size, bootstrap_samples", [(10, 100), (100_000, 100)]
    )
    def test_evaluate(
        self,
        bootstrap_sample_size,
        bootstrap_samples,
        portfolio_df,
        trades_df,
        calc_bootstrap,
        bars_per_year,
        expected_sharpe,
        expected_sortino,
    ):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            portfolio_df,
            trades_df,
            calc_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            bootstrap_samples=bootstrap_samples,
            bars_per_year=bars_per_year,
        )
        assert result.metrics is not None
        if not calc_bootstrap:
            assert result.bootstrap is None
        else:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
            ci = result.bootstrap.conf_intervals
            assert ci.columns.tolist() == ["lower", "upper"]
            names = ci.index.get_level_values(0).unique().tolist()
            assert names == ["Profit Factor", "Sharpe Ratio"]
            for name in names:
                df = ci[ci.index.get_level_values(0) == name]
                confs = df.index.get_level_values(1).tolist()
                assert confs == ["97.5%", "95%", "90%"]
            dd = result.bootstrap.drawdown_conf
            assert dd.columns.tolist() == ["amount", "percent"]
            conf = dd.index.get_level_values(0).tolist()
            assert conf == ["99.9%", "99%", "95%", "90%"]
        metrics = result.metrics
        assert metrics.initial_market_value == 500000
        assert metrics.end_market_value == 693111.87
        assert metrics.total_pnl == 165740.2
        assert (
            metrics.unrealized_pnl
            == metrics.end_market_value
            - metrics.initial_market_value
            - metrics.total_pnl
        )
        assert metrics.total_return_pct == 33.14804
        assert metrics.total_profit == 403511.07999999996
        assert metrics.total_loss == -237770.88
        assert metrics.max_drawdown == -56721.59999999998
        assert metrics.max_drawdown_pct == -7.908428778116649
        assert metrics.max_drawdown_date == datetime(2022, 1, 25, 5, 0)
        assert metrics.win_rate == 52.57731958762887
        assert metrics.loss_rate == 47.42268041237113
        assert metrics.winning_trades == 204
        assert metrics.losing_trades == 184
        assert metrics.avg_pnl == 427.1654639175258
        assert metrics.avg_return_pct == 0.279639175257732
        assert metrics.avg_trade_bars == 2.4149484536082473
        assert metrics.avg_profit == 1977.9954901960782
        assert metrics.avg_profit_pct == 3.1687745098039217
        assert metrics.avg_winning_trade_bars == 2.465686274509804
        assert metrics.avg_loss == -1292.233043478261
        assert metrics.avg_loss_pct == -2.9235326086956523
        assert metrics.avg_losing_trade_bars == 2.358695652173913
        assert metrics.largest_win == 21069.63
        assert metrics.largest_win_pct == 14.49
        assert metrics.largest_win_bars == 3
        assert metrics.largest_loss == -11487.43
        assert metrics.largest_loss_pct == -6.49
        assert metrics.largest_loss_bars == 3
        assert metrics.max_wins == 7
        assert metrics.max_losses == 7
        assert metrics.sharpe == expected_sharpe
        assert metrics.sortino == expected_sortino
        assert metrics.profit_factor == 1.0759385033768167
        assert metrics.ulcer_index == 1.898347959437099
        assert metrics.upi == 0.01844528848501509
        assert metrics.equity_r2 == 0.8979045919638434
        assert metrics.std_error == 69646.36129687089
        assert metrics.total_fees == 0
        if bars_per_year is not None:
            assert metrics.calmar == 1.1557170701224246
            assert truncate(metrics.annual_return_pct, 6) == truncate(
                5.897743691129764, 6
            )
            assert metrics.annual_std_error == 1105601.710272446
            assert metrics.annual_volatility_pct == 21.36797425126505
        else:
            assert metrics.calmar is None
            assert metrics.annual_return_pct is None
            assert metrics.annual_std_error is None
            assert metrics.annual_volatility_pct is None

    def test_evaluate_when_portfolio_empty(self, trades_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            pd.DataFrame(columns=["market_value", "fees"]),
            trades_df,
            calc_bootstrap,
            bootstrap_sample_size=10,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in (
                "calmar",
                "annual_return_pct",
                "annual_std_error",
                "annual_volatility_pct",
                "max_drawdown_date",
            ):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_single_market_value(
        self, trades_df, calc_bootstrap
    ):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            pd.DataFrame(
                [[1000, 0]],
                columns=["market_value", "fees"],
                index=[pd.Timestamp("2023-04-12 00:00:00")],
            ),
            trades_df,
            calc_bootstrap,
            bootstrap_sample_size=10,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in (
                "calmar",
                "annual_return_pct",
                "annual_std_error",
                "annual_volatility_pct",
                "max_drawdown_date",
            ):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_trades_empty(self, portfolio_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            portfolio_df,
            pd.DataFrame(columns=["pnl", "return_pct", "bars"]),
            calc_bootstrap,
            bootstrap_sample_size=10,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        metrics = result.metrics
        assert metrics is not None
        assert metrics.total_pnl == 0
        assert metrics.total_return_pct == 0
        assert metrics.total_profit == 0
        assert metrics.total_loss == 0
        assert metrics.win_rate == 0
        assert metrics.loss_rate == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.avg_pnl == 0
        assert metrics.avg_return_pct == 0
        assert metrics.avg_trade_bars == 0
        assert metrics.avg_profit == 0
        assert metrics.avg_profit_pct == 0
        assert metrics.avg_winning_trade_bars == 0
        assert metrics.avg_loss == 0
        assert metrics.avg_loss_pct == 0
        assert metrics.avg_losing_trade_bars == 0
        assert metrics.largest_win == 0
        assert metrics.largest_win_pct == 0
        assert metrics.largest_win_bars == 0
        assert metrics.largest_loss == 0
        assert metrics.largest_loss_pct == 0
        assert metrics.largest_loss_bars == 0
        assert metrics.max_wins == 0
        assert metrics.max_losses == 0
        assert metrics.total_fees == 0
        if calc_bootstrap:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
        else:
            assert result.bootstrap is None
