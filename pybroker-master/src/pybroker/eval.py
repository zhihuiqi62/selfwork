"""Contains implementation of evaluation metrics."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.scope import StaticScope
from pybroker.vect import highv, inverse_normal_cdf, normal_cdf
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from numba import njit
from numpy.typing import NDArray
from typing import Callable, NamedTuple, Optional


class BootConfIntervals(NamedTuple):
    """Holds confidence intervals of bootstrap tests.

    Attributes:
        low_2p5: Lower bound of 97.5% confidence interval.
        high_2p5: Upper bound of 97.5% confidence interval.
        low_5: Lower bound of 95% confidence interval.
        high_5: Upper bound of 95% confidence interval.
        low_10: Lower bound of 90% confidence interval.
        high_10: Upper bound of 90% confidence interval.
    """

    low_2p5: float
    high_2p5: float
    low_5: float
    high_5: float
    low_10: float
    high_10: float


@njit
def bca_boot_conf(
    x: NDArray[np.float64],
    n: int,
    n_boot: int,
    fn: Callable[[NDArray[np.float64]], float],
) -> BootConfIntervals:
    """Computes confidence intervals for a user-defined parameter using the
    `bias corrected and accelerated (BCa) bootstrap method.
    <https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html>`_

    Args:
        x: :class:`numpy.ndarray` containing the data for the randomized
            bootstrap sampling.
        n: Number of elements in each random bootstrap sample.
        n_boot: Number of random bootstrap samples to use.
        fn: :class:`Callable` for computing the parameter used for the
            confidence intervals.

    Returns:
        :class:`.BootConfIntervals` containing the computed confidence
        intervals.
    """

    if n <= 0:
        raise ValueError("Bootstrap sample size must be greater than 0.")
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n_x = len(x)
    if not n_x:
        return BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if n_x <= n:
        n = n_x
        n_boot = 1

    def clamp(k: int):
        return min(max(k, 0), n_boot - 1)

    x_buff = np.zeros(n)
    boot = np.zeros(n_boot)
    theta_hat = fn(x[:n])
    z0_count = 0
    for i in range(n_boot):
        for j in range(n):
            k = np.random.choice(n_x)
            x_buff[j] = x[k]
        param = fn(x_buff)
        boot[i] = param
        if param < theta_hat:
            z0_count += 1
    z0_count = min(z0_count, n_boot - 1)
    z0_count = max(z0_count, 1)
    z0 = inverse_normal_cdf(z0_count / n_boot)
    theta_dot = 0.0
    for i in range(n):
        x_temp, x[i] = x[i], x[n - 1]
        param = fn(x[: n - 1])
        theta_dot += param
        x_buff[i] = param
        x[i] = x_temp
    theta_dot /= n
    numer = denom = 0
    for i in range(n):
        diff = theta_dot - x_buff[i]
        diff_sq = diff**2
        denom += diff_sq
        numer += diff_sq * diff
    denom = np.power(np.sqrt(denom), 3)
    accel = numer / (6 * denom + 1.0e-60)
    boot.sort()
    zlo = inverse_normal_cdf(0.025)
    zhi = inverse_normal_cdf(0.975)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int((alo * (n_boot + 1))) - 1
    k = clamp(k)
    low_2p5 = boot[k]
    k = int(((1 - ahi) * (n_boot + 1))) - 1
    k = clamp(k)
    high_2p5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.05)
    zhi = inverse_normal_cdf(0.95)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int((alo * (n_boot + 1))) - 1
    k = clamp(k)
    low_5 = boot[k]
    k = int(((1 - ahi) * (n_boot + 1))) - 1
    k = clamp(k)
    high_5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.1)
    zhi = inverse_normal_cdf(0.9)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = int((alo * (n_boot + 1))) - 1
    k = clamp(k)
    low_10 = boot[k]
    k = int(((1 - ahi) * (n_boot + 1))) - 1
    k = clamp(k)
    high_10 = boot[n_boot - 1 - k]
    return BootConfIntervals(low_2p5, high_2p5, low_5, high_5, low_10, high_10)


@njit
def profit_factor(
    changes: NDArray[np.float64], use_log: bool = False
) -> np.floating:
    """Computes the profit factor, which is the ratio of gross profit to gross
    loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
        use_log: Whether to log transform the profit factor. Defaults to False.
    """
    wins = changes[changes > 0]
    losses = changes[changes < 0]
    if not len(wins) and not len(losses):
        return np.float64(0)
    numer = denom = 1.0e-10
    numer += np.sum(wins)
    denom -= np.sum(losses)
    if use_log:
        return np.log(numer / denom)
    else:
        return np.divide(numer, denom)


@njit
def log_profit_factor(changes: NDArray[np.float64]) -> np.floating:
    """Computes the log transformed profit factor, which is the ratio of gross
    profit to gross loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
    """
    return profit_factor(changes, use_log=True)


@njit
def sharpe_ratio(
    returns: NDArray[np.float64],
    obs: Optional[int] = None,
    downside_only: bool = False,
) -> np.floating:
    """Computes the
    `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sharpe Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    std_changes = returns[returns < 0] if downside_only else returns
    if not len(std_changes):
        return np.float64(0)
    std = np.std(std_changes)
    if std == 0:
        return np.float64(0)
    sr = np.mean(returns) / std
    if obs is not None:
        sr *= np.sqrt(obs)
    return sr


def sortino_ratio(
    returns: NDArray[np.float64], obs: Optional[int] = None
) -> float:
    """Computes the
    `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sortino Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    return float(sharpe_ratio(returns, obs, downside_only=True))


def conf_profit_factor(
    x: NDArray[np.float64], n: int, n_boot: int
) -> BootConfIntervals:
    """Computes confidence intervals for :func:`.profit_factor`."""
    intervals = bca_boot_conf(x, n, n_boot, log_profit_factor)
    return BootConfIntervals(
        low_2p5=np.exp(intervals.low_2p5),
        high_2p5=np.exp(intervals.high_2p5),
        low_5=np.exp(intervals.low_5),
        high_5=np.exp(intervals.high_5),
        low_10=np.exp(intervals.low_10),
        high_10=np.exp(intervals.high_10),
    )


def conf_sharpe_ratio(
    x: NDArray[np.float64], n: int, n_boot: int, obs: Optional[int] = None
) -> BootConfIntervals:
    """Computes confidence intervals for :func:`.sharpe_ratio`."""
    intervals = bca_boot_conf(x, n, n_boot, sharpe_ratio)
    if obs is not None:
        factor = np.sqrt(obs)
        intervals = BootConfIntervals(
            low_2p5=intervals.low_2p5 * factor,
            high_2p5=intervals.high_2p5 * factor,
            low_5=intervals.low_5 * factor,
            high_5=intervals.high_5 * factor,
            low_10=intervals.low_10 * factor,
            high_10=intervals.high_10 * factor,
        )
    return intervals


@njit
def max_drawdown(changes: NDArray[np.float64]) -> float:
    """Computes maximum drawdown, measured in cash.

    Args:
        changes: Array of differences between each bar and the previous bar.
    """
    n = len(changes)
    if not n:
        return 0
    cumulative = 0
    max_equity = 0
    dd = 0
    for change in changes:
        cumulative += change
        if cumulative > max_equity:
            max_equity = cumulative
        else:
            loss = max_equity - cumulative
            if loss > dd:
                dd = loss
    return -dd


def calmar_ratio(returns: NDArray[np.float64], bars_per_year: int) -> float:
    """Computes the Calmar Ratio.

    Args:
        returns: Array of returns centered at 0.
        bars_per_year: Number of bars per annum.
    """
    if not len(returns):
        return 0
    max_dd = np.abs(max_drawdown(returns))
    if max_dd == 0:
        return 0
    return np.mean(returns) * bars_per_year / max_dd


@njit
def max_drawdown_percent(
    returns: NDArray[np.float64],
) -> tuple[float, Optional[int]]:
    """Computes maximum drawdown, measured in percentage loss.

    Args:
        returns: Array of returns centered at 0.

    Returns:
        - Maximum drawdown, measured in percentage loss.
        - Index of the maximum drawdown.
    """
    returns = returns + 1
    n = len(returns)
    if not n:
        return 0, None
    cumulative = 1.0
    max_equity = 1.0
    dd = 0.0
    index = None
    for i, r in enumerate(returns):
        cumulative *= r
        if cumulative > max_equity:
            max_equity = cumulative
        elif max_equity > 0:
            loss = (cumulative / max_equity - 1) * 100
            if loss < dd:
                dd = loss
                index = i
    return dd, index


@njit
def _dd_conf(q: float, boot: NDArray[np.float64]) -> float:
    k = int((q * (len(boot) + 1)) - 1)
    k = max(k, 0)
    return boot[k]


class DrawdownConfs(NamedTuple):
    """Contains upper bounds of confidence intervals for maximum drawdown.

    Attributes:
        q_001: 99.9% confidence upper bound.
        q_01: 99% confidence upper bound.
        q_05: 95% confidence upper bound.
        q_10: 90% confidence upper bound.
    """

    q_001: float
    q_01: float
    q_05: float
    q_10: float


class DrawdownMetrics(NamedTuple):
    """Contains drawdown metrics.

    Attributes:
        confs: Upper bounds of confidence intervals for maximum
            drawdown, measured in cash.
        pct_confs: Upper bounds of confidence intervals for maximum
            drawdown, measured in percentage.
    """

    confs: DrawdownConfs
    pct_confs: DrawdownConfs


@njit
def _dd_confs(boot: NDArray[np.float64]) -> DrawdownConfs:
    boot.sort()
    boot = boot[::-1]
    return DrawdownConfs(
        _dd_conf(0.999, boot),
        _dd_conf(0.99, boot),
        _dd_conf(0.95, boot),
        _dd_conf(0.9, boot),
    )


@njit
def drawdown_conf(
    changes: NDArray[np.float64],
    returns: NDArray[np.float64],
    n: int,
    n_boot: int,
) -> DrawdownMetrics:
    """Computes upper bounds of confidence intervals for maximum drawdown using
    the bootstrap method.

    Args:
        changes: Array of differences between each bar and the previous bar.
        returns: Array of returns centered at 0.
        n: Number of elements in each random bootstrap sample.
        n_boot: Number of random bootstrap samples to use.

    Returns:
        :class:`.DrawdownMetrics` containing the confidence bounds.
    """
    if n <= 0:
        raise ValueError("Bootstrap sample size must be greater than 0.")
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n_changes = len(changes)
    if n_changes != len(returns):
        raise ValueError("Param changes length does not match returns length.")
    if n_changes <= n:
        n = n_changes
        n_boot = 1
    changes_sample = np.zeros(n)
    returns_sample = np.zeros(n)
    boot_dd = np.zeros(n_boot)
    boot_dd_pct = np.zeros(n_boot)
    for i in range(n_boot):
        for j in range(n):
            k = np.random.choice(n_changes)
            changes_sample[j] = changes[k]
            returns_sample[j] = returns[k]
        boot_dd[i] = max_drawdown(changes_sample)
        boot_dd_pct[i], _ = max_drawdown_percent(returns_sample)
    return DrawdownMetrics(_dd_confs(boot_dd), _dd_confs(boot_dd_pct))


@njit
def relative_entropy(values: NDArray[np.float64]) -> float:
    """Computes the relative `entropy
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
    """
    x = values[~np.isnan(values)]
    n = len(x)
    if not n:
        return 0
    n_bins = 3
    if n >= 10000:
        n_bins = 20
    elif n >= 1000:
        n_bins = 10
    elif n >= 100:
        n_bins = 5
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    factor = (n_bins - 1.0e-10) / (max_val - min_val + 1.0e-60)
    count = np.zeros(n_bins)
    for v in x:
        k = int(factor * (v - min_val))
        count[k] += 1
    sum_ = 0
    for c in count:
        if c == 0:
            continue
        p = c / n
        sum_ += p * np.log(p)
    return -sum_ / np.log(n_bins)


def iqr(values: NDArray[np.float64]) -> float:
    """Computes the `interquartile range (IQR)
    <https://en.wikipedia.org/wiki/Interquartile_range>`_ of ``values``."""
    x = values[~np.isnan(values)]
    if not len(x):
        return 0
    percentiles: NDArray[np.float64] = np.percentile(
        x, [75, 25], method="midpoint"
    )
    q75: float = float(percentiles[0])
    q25: float = float(percentiles[1])
    return q75 - q25


@njit
def ulcer_index(values: NDArray[np.float64], period: int = 14) -> float:
    """Computes the
    `Ulcer Index <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
    """
    n = len(values)
    if n <= period:
        return 0
    start = period - 1
    dd = np.zeros(n - start)
    max_values = highv(values, period)
    for i in range(start, n):
        if max_values[i] == 0:
            dd[i - start] = 0
            continue
        dd[i - start] = (values[i] - max_values[i]) / max_values[i] * 100
    return np.sqrt(np.mean(np.square(dd)))


@njit
def upi(
    values: NDArray[np.float64], period: int = 14, ui: Optional[float] = None
) -> float:
    """Computes the `Ulcer Performance Index
    <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
    """
    if len(values) <= 1:
        return 0
    if ui is None:
        ui = ulcer_index(values, period)
    if ui == 0:
        return 0
    r = np.zeros(len(values) - 1)
    for i in range(len(r)):
        r[i] = (values[i + 1] - values[i]) / values[i] * 100
    return float(np.mean(r) / ui)


def win_loss_rate(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the win rate and loss rate as percentages.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of win rate and loss rate.
    """
    pnls = pnls[pnls != 0]
    n = len(pnls)
    if not n:
        return 0, 0
    win_rate = len(pnls[pnls > 0]) / n * 100
    loss_rate = len(pnls[pnls < 0]) / n * 100
    return win_rate, loss_rate


def winning_losing_trades(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Returns the number of winning and losing trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` containing numbers of winning and losing trades.
    """
    pnls = pnls[pnls != 0]
    if not len(pnls):
        return 0, 0
    return len(pnls[pnls > 0]), len(pnls[pnls < 0])


def total_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes total profit and loss.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of total profit and total loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        np.sum(profits) if len(profits) else 0,
        np.sum(losses) if len(losses) else 0,
    )


def avg_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the average profit and average loss per trade.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of average profit and average loss.
    """

    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        float(np.mean(profits)) if len(profits) else 0,
        float(np.mean(losses)) if len(losses) else 0,
    )


def largest_win_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the largest profit and largest loss of all trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of largest profit and largest loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        np.max(profits) if len(profits) else 0,
        np.min(losses) if len(losses) else 0,
    )


@njit
def max_wins_losses(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Computes the max consecutive wins and max consecutive losses.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` of max consecutive wins and max consecutive losses.
    """
    max_wins = max_losses = wins = losses = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1
            max_wins = max(max_wins, wins)
        else:
            wins = 0
        if pnl < 0:
            losses += 1
            max_losses = max(max_losses, losses)
        else:
            losses = 0
    return max_wins, max_losses


def total_return_percent(initial_value: float, pnl: float) -> float:
    """Computes total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
    """
    if initial_value == 0:
        return 0
    return ((pnl + initial_value) / initial_value - 1) * 100


def annual_total_return_percent(
    initial_value: float, pnl: float, bars_per_year: int, total_bars: int
) -> float:
    """Computes annualized total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
        bars_per_year: Number of bars per annum.
        total_bars: Total number of bars of the return.
    """
    if initial_value == 0 or total_bars == 0:
        return 0
    return (
        np.power(
            (pnl + initial_value) / initial_value, bars_per_year / total_bars
        )
        - 1
    ) * 100


def r_squared(values: NDArray[np.float64]) -> float:
    """Computes R-squared of ``values``."""
    n = len(values)
    if not n:
        return 0
    x = np.arange(n)
    try:
        coeffs = np.polyfit(x, values, 1)
        pred = np.poly1d(coeffs)(x)
        y_hat = np.mean(values)
        ssres = float(np.sum((values - pred) ** 2))
        sstot = float(np.sum((values - y_hat) ** 2))
        if sstot == 0:
            return 0
        return 1 - ssres / sstot
    except Exception:
        return 0


class BootstrapResult(NamedTuple):
    """Contains results of bootstrap tests.

    Attributes:
        conf_intervals: :class:`pandas.DataFrame` containing confidence
            intervals for :func:`.log_profit_factor` and :func:`.sharpe_ratio`.
        drawdown_conf: :class:`pandas.DataFrame` containing upper bounds of
            confidence intervals for maximum drawdown.
        profit_factor: Contains profit factor confidence intervals.
        sharpe: Contains Sharpe Ratio confidence intervals.
        drawdown: Contains drawdown confidence intervals.
    """

    conf_intervals: pd.DataFrame
    drawdown_conf: pd.DataFrame
    profit_factor: BootConfIntervals
    sharpe: BootConfIntervals
    drawdown: DrawdownMetrics


@dataclass(frozen=True)
class EvalMetrics:
    """Contains metrics for evaluating a :class:`pybroker.strategy.Strategy`.

    Attributes:
        trade_count: Number of trades that were filled.
        initial_market_value: Initial market value of the
            :class:`pybroker.portfolio.Portfolio`.
        end_market_value: Ending market value of the
            :class:`pybroker.portfolio.Portfolio`.
        total_pnl: Total realized profit and loss (PnL).
        unrealized_pnl: Total unrealized profit and loss (PnL).
        total_return_pct: Total realized return measured in percentage.
        annual_return_pct: Annualized total realized return measured in
            percentage.
        total_profit: Total realized profit.
        total_loss: Total realized loss.
        total_fees: Total brokerage fees. See
            :attr:`pybroker.config.StrategyConfig.fee_mode` for more info.
        max_drawdown: Maximum drawdown, measured in cash.
        max_drawdown_pct: Maximum drawdown, measured in percentage.
        max_drawdown_date: Date of maximum drawdown.
        win_rate: Win rate of trades.
        loss_rate: Loss rate of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        avg_pnl: Average profit and loss (PnL) per trade, measured in cash.
        avg_return_pct: Average return per trade, measured in percentage.
        avg_trade_bars: Average number of bars per trade.
        avg_profit: Average profit per trade, measured in cash.
        avg_profit_pct: Average profit per trade, measured in percentage.
        avg_winning_trade_bars: Average number of bars per winning trade.
        avg_loss: Average loss per trade, measured in cash.
        avg_loss_pct: Average loss per trade, measured in percentage.
        avg_losing_trade_bars: Average number of bars per losing trade.
        largest_win: Largest profit of a trade, measured in cash.
        largest_win_pct: Largest profit of a trade, measured in percentage
        largest_win_bars: Number of bars in the largest winning trade.
        largest_loss: Largest loss of a trade, measured in cash.
        largest_loss_pct: Largest loss of a trade, measured in percentage.
        largest_loss_bars: Number of bars in the largest losing trade.
        max_wins: Maximum number of consecutive winning trades.
        max_losses: Maximum number of consecutive losing trades.
        sharpe: `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_,
            computed per bar.
        sortino: `Sortino Ratio
            <https://en.wikipedia.org/wiki/Sortino_ratio>`_, computed per bar.
        calmar: Calmar Ratio, computed per bar.
        profit_factor: Ratio of gross profit to gross loss, computed per bar.
        ulcer_index: `Ulcer Index
            <https://en.wikipedia.org/wiki/Ulcer_index>`_, computed per bar.
        upi: `Ulcer Performance Index
            <https://en.wikipedia.org/wiki/Ulcer_index>`_, computed per bar.
        equity_r2: R^2 of the equity curve, computed per bar on market values
            of portfolio.
        std_error: Standard error, computed per bar on market values of
            portfolio.
        annual_std_error: Annualized standard error, computed per bar on market
            values of portfolio.
        annual_volatility_pct: Annualized volatility percentage, computed per
            bar on market values of portfolio.
    """

    trade_count: int = field(default=0)
    initial_market_value: float = field(default=0)
    end_market_value: float = field(default=0)
    total_pnl: float = field(default=0)
    unrealized_pnl: float = field(default=0)
    total_return_pct: float = field(default=0)
    annual_return_pct: Optional[float] = field(default=None)
    total_profit: float = field(default=0)
    total_loss: float = field(default=0)
    total_fees: float = field(default=0)
    max_drawdown: float = field(default=0)
    max_drawdown_pct: float = field(default=0)
    max_drawdown_date: Optional[datetime] = field(default=None)
    win_rate: float = field(default=0)
    loss_rate: float = field(default=0)
    winning_trades: int = field(default=0)
    losing_trades: int = field(default=0)
    avg_pnl: float = field(default=0)
    avg_return_pct: float = field(default=0)
    avg_trade_bars: float = field(default=0)
    avg_profit: float = field(default=0)
    avg_profit_pct: float = field(default=0)
    avg_winning_trade_bars: float = field(default=0)
    avg_loss: float = field(default=0)
    avg_loss_pct: float = field(default=0)
    avg_losing_trade_bars: float = field(default=0)
    largest_win: float = field(default=0)
    largest_win_pct: float = field(default=0)
    largest_win_bars: int = field(default=0)
    largest_loss: float = field(default=0)
    largest_loss_pct: float = field(default=0)
    largest_loss_bars: int = field(default=0)
    max_wins: int = field(default=0)
    max_losses: int = field(default=0)
    sharpe: float = field(default=0)
    sortino: float = field(default=0)
    calmar: Optional[float] = field(default=None)
    profit_factor: float = field(default=0)
    ulcer_index: float = field(default=0)
    upi: float = field(default=0)
    equity_r2: float = field(default=0)
    std_error: float = field(default=0)
    annual_std_error: Optional[float] = field(default=None)
    annual_volatility_pct: Optional[float] = field(default=None)


class ConfInterval(NamedTuple):
    """Confidence interval upper and low bounds.

    Attributes:
        name: Parameter name.
        conf: Confidence interval percentage represented as a ``str``.
        lower: Lower bound.
        upper: Upper bound.
    """

    name: str
    conf: str
    lower: float
    upper: float


class EvalResult(NamedTuple):
    """Contains evaluation result.

    Attributes:
        metrics: Evaluation metrics.
        bootstrap: Randomized bootstrap metrics.
    """

    metrics: EvalMetrics
    bootstrap: Optional[BootstrapResult]


class _ConfsResult(NamedTuple):
    df: pd.DataFrame
    profit_factor: BootConfIntervals
    sharpe: BootConfIntervals


class _DrawdownResult(NamedTuple):
    df: pd.DataFrame
    metrics: DrawdownMetrics


class EvaluateMixin:
    """Mixin for computing evaluation metrics."""

    def evaluate(
        self,
        portfolio_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        calc_bootstrap: bool,
        bootstrap_sample_size: int,
        bootstrap_samples: int,
        bars_per_year: Optional[int],
    ) -> EvalResult:
        """Computes evaluation metrics.

        Args:
            portfolio_df: :class:`pandas.DataFrame` of portfolio market values
                per bar.
            trades_df: :class:`pandas.DataFrame` of trades.
            calc_bootstrap: ``True`` to calculate randomized bootstrap metrics.
            bootstrap_sample_size: Size of each random bootstrap sample.
            bootstrap_samples: Number of random bootstrap samples to use.
            bars_per_year: Number of observations per years that will be used
                to annualize evaluation metrics. For example, a value of
                ``252`` would be used to annualize the Sharpe Ratio for daily
                returns.

        Returns:
            :class:`.EvalResult` containing evaluation metrics.
        """
        market_values = portfolio_df["market_value"].to_numpy()
        fees = portfolio_df["fees"].to_numpy()
        bar_returns = self._calc_bar_returns(portfolio_df)
        bar_return_dates = bar_returns.index.to_series().reset_index(drop=True)
        bar_returns = bar_returns.to_numpy()
        bar_changes = self._calc_bar_changes(portfolio_df)
        if (
            not len(market_values)
            or not len(bar_returns)
            or not len(bar_changes)
        ):
            return EvalResult(EvalMetrics(), None)
        pnls = trades_df["pnl"].to_numpy()
        return_pcts = trades_df["return_pct"].to_numpy()
        bars = trades_df["bars"].to_numpy()
        winning_trades = trades_df[trades_df["pnl"] > 0]
        winning_bars = winning_trades["bars"].to_numpy()
        losing_trades = trades_df[trades_df["pnl"] < 0]
        losing_bars = losing_trades["bars"].to_numpy()
        largest_win = winning_trades[
            winning_trades["pnl"] == winning_trades["pnl"].max()
        ]
        largest_win_pct = (
            0 if largest_win.empty else largest_win["return_pct"].values[0]
        )
        largest_win_bars = (
            0 if largest_win.empty else largest_win["bars"].values[0]
        )
        largest_loss = losing_trades[
            losing_trades["pnl"] == losing_trades["pnl"].min()
        ]
        largest_loss_pct = (
            0 if largest_loss.empty else largest_loss["return_pct"].values[0]
        )
        largest_loss_bars = (
            0 if largest_loss.empty else largest_loss["bars"].values[0]
        )
        metrics = self._calc_eval_metrics(
            market_values,
            bar_changes,
            bar_returns,
            bar_return_dates,
            pnls,
            return_pcts,
            bars=bars,
            winning_bars=winning_bars,
            losing_bars=losing_bars,
            largest_win_num_bars=largest_win_bars,
            largest_win_pct=largest_win_pct,
            largest_loss_num_bars=largest_loss_bars,
            largest_loss_pct=largest_loss_pct,
            fees=fees,
            bars_per_year=bars_per_year,
        )
        logger = StaticScope.instance().logger
        if not calc_bootstrap:
            return EvalResult(metrics, None)
        if len(bar_returns) <= bootstrap_sample_size:
            logger.warn_bootstrap_sample_size(
                len(bar_returns), bootstrap_sample_size
            )
        logger.calc_bootstrap_metrics_start(
            samples=bootstrap_samples, sample_size=bootstrap_sample_size
        )
        confs_result = self._calc_conf_intervals(
            changes=bar_changes,
            returns=bar_returns,
            sample_size=bootstrap_sample_size,
            samples=bootstrap_samples,
            bars_per_year=bars_per_year,
        )
        dd_result = self._calc_drawdown_conf(
            changes=bar_changes,
            returns=bar_returns,
            sample_size=bootstrap_sample_size,
            samples=bootstrap_samples,
        )
        bootstrap = BootstrapResult(
            conf_intervals=confs_result.df,
            drawdown_conf=dd_result.df,
            profit_factor=confs_result.profit_factor,
            sharpe=confs_result.sharpe,
            drawdown=dd_result.metrics,
        )
        logger.calc_bootstrap_metrics_completed()
        return EvalResult(metrics, bootstrap)

    def _calc_bar_returns(self, df: pd.DataFrame) -> pd.Series:
        prev_market_value = df["market_value"].shift(1)
        returns = (df["market_value"] - prev_market_value) / prev_market_value
        return returns.dropna()

    def _calc_bar_changes(self, df: pd.DataFrame) -> NDArray[np.float64]:
        changes = df["market_value"] - df["market_value"].shift(1)
        return changes.dropna().to_numpy()

    def _calc_eval_metrics(
        self,
        market_values: NDArray[np.float64],
        bar_changes: NDArray[np.float64],
        bar_returns: NDArray[np.float64],
        bar_return_dates: pd.Series,
        pnls: NDArray[np.float64],
        return_pcts: NDArray[np.float64],
        bars: NDArray[np.int_],
        winning_bars: NDArray[np.int_],
        losing_bars: NDArray[np.int_],
        largest_win_num_bars: int,
        largest_win_pct: float,
        largest_loss_num_bars: int,
        largest_loss_pct: float,
        fees: NDArray[np.float64],
        bars_per_year: Optional[int],
    ) -> EvalMetrics:
        total_fees = fees[-1] if len(fees) else 0
        max_dd = max_drawdown(bar_changes)
        max_dd_pct, max_dd_index = max_drawdown_percent(bar_returns)
        max_dd_date = (
            bar_return_dates.iloc[max_dd_index].to_pydatetime()
            if max_dd_index
            else None
        )
        sharpe = sharpe_ratio(bar_returns, bars_per_year)
        sortino = sortino_ratio(bar_returns, bars_per_year)
        pf = profit_factor(bar_changes)
        r2 = r_squared(market_values)
        ui = ulcer_index(market_values)
        upi_ = upi(market_values, ui=ui)
        std_error = float(np.std(market_values))
        largest_win = 0.0
        largest_loss = 0.0
        win_rate = 0.0
        loss_rate = 0.0
        winning_trades = 0
        losing_trades = 0
        avg_pnl = 0.0
        avg_return_pct = 0.0
        avg_trade_bars = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        avg_profit_pct = 0.0
        avg_loss_pct = 0.0
        avg_winning_trade_bars = 0.0
        avg_losing_trade_bars = 0.0
        total_profit = 0.0
        total_loss = 0.0
        total_pnl = 0.0
        unrealized_pnl = 0.0
        max_wins = 0
        max_losses = 0
        if len(pnls):
            largest_win, largest_loss = largest_win_loss(pnls)
            win_rate, loss_rate = win_loss_rate(pnls)
            winning_trades, losing_trades = winning_losing_trades(pnls)
            avg_profit, avg_loss = avg_profit_loss(pnls)
            avg_profit_pct, avg_loss_pct = avg_profit_loss(return_pcts)
            total_profit, total_loss = total_profit_loss(pnls)
            max_wins, max_losses = max_wins_losses(pnls)
            total_pnl = float(np.sum(pnls))
            # Check length to avoid "Mean of empty slice" warning.
            if len(pnls):
                avg_pnl = float(np.mean(pnls))
            if len(return_pcts):
                avg_return_pct = float(np.mean(return_pcts))
            if len(bars):
                avg_trade_bars = float(np.mean(bars))
            if len(winning_bars):
                avg_winning_trade_bars = float(np.mean(winning_bars))
            if len(losing_bars):
                avg_losing_trade_bars = float(np.mean(losing_bars))
        total_return_pct = total_return_percent(
            initial_value=market_values[0], pnl=total_pnl
        )
        unrealized_pnl = market_values[-1] - market_values[0] - total_pnl
        annual_return_pct = None
        annual_std_error = None
        annual_volatility_pct = None
        calmar = None
        if bars_per_year is not None:
            annual_return_pct = annual_total_return_percent(
                initial_value=market_values[0],
                pnl=total_pnl,
                bars_per_year=bars_per_year,
                total_bars=len(market_values),
            )
            annual_std_error = std_error * np.sqrt(bars_per_year)
            annual_volatility_pct = float(
                np.std(bar_returns * 100) * np.sqrt(bars_per_year)
            )
            calmar = calmar_ratio(bar_returns, bars_per_year)
        return EvalMetrics(
            trade_count=len(pnls),
            initial_market_value=market_values[0],
            end_market_value=market_values[-1],
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_date=max_dd_date,
            largest_win=largest_win,
            largest_win_pct=largest_win_pct,
            largest_win_bars=largest_win_num_bars,
            largest_loss=largest_loss,
            largest_loss_pct=largest_loss_pct,
            largest_loss_bars=largest_loss_num_bars,
            max_wins=max_wins,
            max_losses=max_losses,
            win_rate=win_rate,
            loss_rate=loss_rate,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_pnl=avg_pnl,
            avg_return_pct=avg_return_pct,
            avg_trade_bars=avg_trade_bars,
            avg_profit=avg_profit,
            avg_profit_pct=avg_profit_pct,
            avg_winning_trade_bars=avg_winning_trade_bars,
            avg_loss=avg_loss,
            avg_loss_pct=avg_loss_pct,
            avg_losing_trade_bars=avg_losing_trade_bars,
            total_profit=total_profit,
            total_loss=total_loss,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            total_return_pct=total_return_pct,
            annual_return_pct=annual_return_pct,
            total_fees=total_fees,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            profit_factor=pf,
            equity_r2=r2,
            ulcer_index=ui,
            upi=upi_,
            std_error=std_error,
            annual_std_error=annual_std_error,
            annual_volatility_pct=annual_volatility_pct,
        )

    def _calc_conf_intervals(
        self,
        changes: NDArray[np.float64],
        returns: NDArray[np.float64],
        sample_size: int,
        samples: int,
        bars_per_year: Optional[int],
    ) -> _ConfsResult:
        pf_intervals = conf_profit_factor(changes, sample_size, samples)
        pf_conf = self._to_conf_intervals("Profit Factor", pf_intervals)
        sr_intervals = conf_sharpe_ratio(
            returns, sample_size, samples, bars_per_year
        )
        sharpe_conf = self._to_conf_intervals("Sharpe Ratio", sr_intervals)
        df = pd.DataFrame.from_records(
            pf_conf + sharpe_conf, columns=ConfInterval._fields
        )
        df.set_index(["name", "conf"], inplace=True)
        return _ConfsResult(
            df=df, profit_factor=pf_intervals, sharpe=sr_intervals
        )

    def _to_conf_intervals(
        self, name: str, conf: BootConfIntervals
    ) -> deque[ConfInterval]:
        results: deque[ConfInterval] = deque()
        results.append(
            ConfInterval(name, "97.5%", conf.low_2p5, conf.high_2p5)
        )
        results.append(ConfInterval(name, "95%", conf.low_5, conf.high_5))
        results.append(ConfInterval(name, "90%", conf.low_10, conf.high_10))
        return results

    def _calc_drawdown_conf(
        self,
        changes: NDArray[np.float64],
        returns: NDArray[np.float64],
        sample_size: int,
        samples: int,
    ) -> _DrawdownResult:
        metrics = drawdown_conf(changes, returns, sample_size, samples)
        df = pd.DataFrame(
            zip(("99.9%", "99%", "95%", "90%"), *metrics),
            columns=("conf", "amount", "percent"),
        )
        df.set_index("conf", inplace=True)
        return _DrawdownResult(df=df, metrics=metrics)
