"""Unit tests for indicator.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
import re
from pybroker.cache import CacheDateFields
from .fixtures import *  # noqa: F401
from pybroker.common import BarData, DataCol, IndicatorSymbol, to_datetime
from pybroker.indicator import (
    _to_bar_data,
    Indicator,
    IndicatorsMixin,
    IndicatorSet,
    adx,
    aroon_diff,
    aroon_down,
    aroon_up,
    close_minus_ma,
    cubic_deviation,
    cubic_trend,
    delta_on_balance_volume,
    detrended_rsi,
    highest,
    indicator,
    intraday_intensity,
    laguerre_rsi,
    linear_deviation,
    linear_trend,
    lowest,
    macd,
    money_flow,
    normalized_negative_volume_index,
    normalized_on_balance_volume,
    normalized_positive_volume_index,
    price_change_oscillator,
    price_intensity,
    price_volume_fit,
    quadratic_deviation,
    quadratic_trend,
    reactivity,
    returns,
    stochastic,
    stochastic_rsi,
    volume_weighted_ma_ratio,
    volume_momentum,
)
from pybroker.vect import lowv

TF_SECONDS = 60
BETWEEN_TIME = ("10:00", "15:30")


@pytest.fixture()
def cache_date_fields(data_source_df):
    return CacheDateFields(
        start_date=to_datetime(sorted(data_source_df["date"].unique())[0]),
        end_date=to_datetime(sorted(data_source_df["date"].unique())[-1]),
        tf_seconds=TF_SECONDS,
        between_time=BETWEEN_TIME,
        days=None,
    )


@pytest.fixture(params=[True, False])
def disable_parallel(request):
    return request.param


@pytest.fixture()
def ind_syms(hhv_ind, llv_ind, sumv_ind, symbols):
    return [
        IndicatorSymbol(ind.name, sym)
        for sym in symbols
        for ind in (hhv_ind, llv_ind, sumv_ind)
    ]


@pytest.fixture()
def setup_teardown(scope):
    scope.register_custom_cols("adj_close")
    yield
    scope.unregister_custom_cols("adj_close")


@pytest.mark.usefixtures("setup_teardown")
def test_to_bar_data(scope, data_source_df):
    bar_data = _to_bar_data(data_source_df)
    for col in scope.all_data_cols:
        expected = (
            data_source_df[col].to_numpy()
            if col in data_source_df.columns
            else None
        )
        assert np.array_equal(getattr(bar_data, col), expected)


@pytest.mark.parametrize("drop_col", ["date", "open", "high", "low", "close"])
def test_to_bar_data_when_missing_cols_then_error(drop_col, data_source_df):
    with pytest.raises(
        ValueError, match=f"DataFrame is missing required column: {drop_col}"
    ):
        _to_bar_data(data_source_df.drop(columns=drop_col))


@pytest.mark.usefixtures("setup_teardown")
def test_indicator():
    ind = indicator("llv", lowv)
    assert isinstance(ind, Indicator)
    assert ind.name == "llv"


@pytest.mark.usefixtures("setup_teardown")
class TestIndicator:
    def test_call_with_kwargs(self, hhv_ind, data_source_df):
        data = hhv_ind(data_source_df)
        assert len(data) == len(data_source_df["date"])
        assert isinstance(data.index[0], pd.Timestamp)

    def test_call_when_invalid_shape_then_error(self, data_source_df):
        def invalid_shape(_data):
            return np.array([[1, 2, 3], [4, 5, 6]])

        ind_invalid_shape = indicator("invalid_shape", invalid_shape)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Indicator invalid_shape must return a one-dimensional array."
            ),
        ):
            ind_invalid_shape(data_source_df)

    def test_iqr(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.iqr(data_source_df), float)

    def test_relative_entropy(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.relative_entropy(data_source_df), float)

    def test_repr(self, hhv_ind):
        assert repr(hhv_ind) == "Indicator('hhv', {'n': 5})"


@pytest.mark.usefixtures("setup_teardown")
class TestIndicatorsMixin:
    def _assert_indicators(self, ind_data, ind_syms, data_source_df):
        assert set(ind_data.keys()) == set(ind_syms)
        for ind_sym, series in ind_data.items():
            df = data_source_df[data_source_df["symbol"] == ind_sym.symbol]
            assert len(series) == df.shape[0]

    @pytest.mark.usefixtures("setup_ind_cache")
    def test_compute_indicators(
        self, ind_syms, data_source_df, cache_date_fields, disable_parallel
    ):
        mixin = IndicatorsMixin()
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)

    @pytest.mark.usefixtures("setup_ind_cache")
    def test_compute_indicators_when_empty_data(
        self, ind_syms, cache_date_fields, disable_parallel
    ):
        mixin = IndicatorsMixin()
        ind_data = mixin.compute_indicators(
            df=pd.DataFrame(columns=[col.value for col in DataCol]),
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        assert len(ind_data) == 0

    @pytest.mark.usefixtures("setup_enabled_ind_cache")
    def test_compute_indicators_data_when_cached(
        self, ind_syms, cache_date_fields, data_source_df, disable_parallel
    ):
        mixin = IndicatorsMixin()
        mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)

    @pytest.mark.usefixtures("setup_enabled_ind_cache")
    def test_compute_indicators_when_partial_cached(
        self, ind_syms, cache_date_fields, data_source_df, disable_parallel
    ):
        mixin = IndicatorsMixin()
        mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms[:1],
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)


class TestIndicatorSet:
    def test_add_and_remove(self, hhv_ind, llv_ind, sumv_ind):
        ind_set = IndicatorSet()
        ind_set.add(hhv_ind)
        ind_set.add([llv_ind, sumv_ind], hhv_ind)
        assert ind_set._ind_names == set(["llv", "hhv", "sumv"])
        ind_set.remove(llv_ind)
        assert ind_set._ind_names == set(["hhv", "sumv"])
        ind_set.remove(hhv_ind, sumv_ind)
        assert not ind_set._ind_names

    def test_clear(self, hhv_ind, llv_ind, sumv_ind):
        ind_set = IndicatorSet()
        ind_set.add(llv_ind, sumv_ind, hhv_ind)
        assert ind_set._ind_names == set(["llv", "hhv", "sumv"])
        ind_set.clear()
        assert not ind_set._ind_names

    def test_call_when_indicators_empty_then_error(self, data_source_df):
        ind_set = IndicatorSet()
        with pytest.raises(ValueError, match="No indicators were added."):
            ind_set(data_source_df)

    @pytest.mark.parametrize(
        "df", [pd.DataFrame(), LazyFixture("data_source_df")]
    )
    def test_call(self, df, hhv_ind, llv_ind, disable_parallel, request):
        df = get_fixture(request, df)
        ind_set = IndicatorSet()
        ind_set.add([hhv_ind, llv_ind])
        result = ind_set(df, disable_parallel)
        assert len(result) == len(df)
        assert set(result.columns) == set(["date", "symbol", "hhv", "llv"])


@pytest.mark.parametrize(
    "fn, values, period, expected",
    [
        (
            highest,
            [3, 3, 4, 2, 5, 6, 1, 3],
            3,
            [np.nan, np.nan, 4, 4, 5, 6, 6, 6],
        ),
        (highest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        (highest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]),
        (highest, [1], 1, [1]),
        (
            lowest,
            [3, 3, 4, 2, 5, 6, 1, 3],
            3,
            [np.nan, np.nan, 3, 2, 2, 2, 1, 1],
        ),
        (lowest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        (lowest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]),
        (lowest, [1], 1, [1]),
        (
            returns,
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            1,
            [np.nan, 0.5, 0.13333333, -0.23529412, -0.07692308, 0.16666667],
        ),
        (
            returns,
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            2,
            [np.nan, np.nan, 0.7, -0.133333, -0.294118, 0.076923],
        ),
        (returns, [1], 1, [np.nan]),
        (returns, [], 5, []),
    ],
)
def test_wrappers(fn, values, period, expected):
    n = len(values)
    dates = pd.date_range(start="1/1/2018", end="1/1/2019").to_numpy()[:n]
    bar_data = BarData(
        date=dates,
        open=np.zeros(n),
        high=np.zeros(n),
        low=np.zeros(n),
        close=np.array(values),
        volume=None,
        vwap=None,
    )
    indicator = fn("my_indicator", "close", period)
    assert isinstance(indicator, Indicator)
    assert indicator.name == "my_indicator"
    series = indicator(bar_data)
    assert np.array_equal(series.index.to_numpy(), dates)
    assert np.array_equal(
        np.round(series.values, 6), np.round(expected, 6), equal_nan=True
    )


@pytest.mark.parametrize(
    "fn, args",
    [
        (
            detrended_rsi,
            {
                "field": "close",
                "short_length": 5,
                "long_length": 10,
                "reg_length": 20,
            },
        ),
        (macd, {"short_length": 5, "long_length": 10, "smoothing": 2.0}),
        (
            stochastic,
            {
                "lookback": 10,
                "smoothing": 2,
            },
        ),
        (
            stochastic_rsi,
            {
                "field": "close",
                "rsi_lookback": 10,
                "sto_lookback": 10,
                "smoothing": 2.0,
            },
        ),
        (
            linear_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            quadratic_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            cubic_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            adx,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_up,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_down,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_diff,
            {
                "lookback": 10,
            },
        ),
        (close_minus_ma, {"lookback": 10, "atr_length": 20, "scale": 0.5}),
        (linear_deviation, {"field": "close", "lookback": 10, "scale": 0.5}),
        (
            quadratic_deviation,
            {"field": "close", "lookback": 10, "scale": 0.5},
        ),
        (cubic_deviation, {"field": "close", "lookback": 10, "scale": 0.5}),
        (price_intensity, {"smoothing": 1.0, "scale": 0.5}),
        (
            price_change_oscillator,
            {"short_length": 5, "multiplier": 3, "scale": 0.5},
        ),
        (intraday_intensity, {"lookback": 10, "smoothing": 1.0}),
        (money_flow, {"lookback": 10, "smoothing": 1.0}),
        (reactivity, {"lookback": 10, "smoothing": 1.0, "scale": 0.5}),
        (price_volume_fit, {"lookback": 10, "scale": 0.5}),
        (volume_weighted_ma_ratio, {"lookback": 10, "scale": 0.5}),
        (normalized_on_balance_volume, {"lookback": 10, "scale": 0.5}),
        (
            delta_on_balance_volume,
            {"lookback": 10, "delta_length": 5, "scale": 0.5},
        ),
        (normalized_positive_volume_index, {"lookback": 10, "scale": 0.5}),
        (normalized_negative_volume_index, {"lookback": 10, "scale": 0.5}),
        (volume_momentum, {"short_length": 5, "multiplier": 2, "scale": 2.0}),
        (laguerre_rsi, {"fe_length": 20}),
    ],
)
def test_indicators(fn, args):
    dates = pd.date_range(start="1/1/2018", end="1/1/2019").to_numpy()
    n = len(dates)
    bar_data = BarData(
        date=dates,
        open=np.random.rand(n),
        high=np.random.rand(n),
        low=np.random.rand(n),
        close=np.random.rand(n),
        volume=np.random.rand(n),
        vwap=None,
    )
    indicator = fn(fn.__name__, **args)
    assert isinstance(indicator, Indicator)
    assert indicator.name == fn.__name__
    series = indicator(bar_data)
    assert len(series) == n
    assert np.array_equal(series.index.to_numpy(), dates)
