"""Contains common classes and utilities."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from joblib import Parallel
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Final,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

_tf_pattern: Final = re.compile(r"(\d+)([A-Za-z]+)")
_tf_abbr: Final = {
    "s": "sec",
    "m": "min",
    "h": "hour",
    "d": "day",
    "w": "week",
}
_CENTS: Final = Decimal(".01")


class IndicatorSymbol(NamedTuple):
    """:class:`pybroker.indicator.Indicator`/symbol identifier.

    Attributes:
        ind_name: Indicator name.
        symbol: Ticker symbol.
    """

    ind_name: str
    symbol: str


class ModelSymbol(NamedTuple):
    """:class:`pybroker.model.ModelSource`/symbol identifier.

    Attributes:
        model_name: Model name.
        symbol: Ticker symbol.
    """

    model_name: str
    symbol: str


class TrainedModel(NamedTuple):
    """Trained model/symbol identifier.

    Attributes:
        name: Trained model name.
        instance: Trained model instance.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        input_cols: Names of the columns to be used as input for the model when
            making predictions.
    """

    name: str
    instance: Any
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]]
    input_cols: Optional[tuple[str]]


class DataCol(Enum):
    """Default data column names."""

    DATE = "date"
    SYMBOL = "symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    VWAP = "vwap"


class Day(Enum):
    """Enumeration of days."""

    MON = 0
    TUES = 1
    WEDS = 2
    THURS = 3
    FRI = 4
    SAT = 5
    SUN = 6


class PriceType(Enum):
    """Enumeration of price types used to specify fill price with
    :class:`pybroker.context.ExecContext`.

    Attributes:
        OPEN: Open price of the current bar.
        LOW: Low price of the current bar.
        HIGH: High price of the current bar.
        CLOSE: Close price of the current bar.
        MIDDLE: Midpoint between low price and high price of the current bar.
        AVERAGE: Average of open, low, high, and close prices of the current
            bar.
    """

    OPEN = "open"
    LOW = "low"
    HIGH = "high"
    CLOSE = "close"
    MIDDLE = "middle"
    AVERAGE = "average"


class StopType(Enum):
    """Stop types.

    Attributes:
        BAR: Stop that triggers after n bars.
        LOSS: Stop loss.
        PROFIT: Take profit.
        TRAILING: Trailing stop loss.
    """

    BAR = "bar"
    LOSS = "loss"
    PROFIT = "profit"
    TRAILING = "trailing"


class FeeMode(Enum):
    """Brokerage fee mode to use for backtesting.

    Attributes:
        ORDER_PERCENT: Fee is a percentage of order amount, where order amount
            is fill_price * shares.
        PER_ORDER: Fee is a constant amount per order.
        PER_SHARE: Fee is a constant amount per share in order.
    """

    ORDER_PERCENT = "order_percent"
    PER_ORDER = "per_order"
    PER_SHARE = "per_share"


class FeeInfo(NamedTuple):
    """Contains info for custom fee calculations.

    Attributes:
        symbol: Trading symbol.
        shares: Number of shares in order.
        fill_price: Fill price of order.
        order_type: Type of order, either "buy" or "sell".
    """

    symbol: str
    shares: Decimal
    fill_price: Decimal
    order_type: Literal["buy", "sell"]


class PositionMode(Enum):
    """Position mode for backtesting.

    Attributes:
        DEFAULT: Long and short positions.
        LONG_ONLY: Long-only positions.
        SHORT_ONLY: Short-only positions.
    """

    DEFAULT = "default"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"


class BarData:
    r"""Contains data for a series of bars. Each field is a
    :class:`numpy.ndarray` that contains bar values in the series. The values
    are sorted in ascending chronological order.

    Args:
        date: Timestamps of each bar.
        open: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volumes.
        vwap: Volume-weighted average prices (VWAP).
        \**kwargs: Custom data fields.
    """

    def __init__(
        self,
        date: NDArray[np.datetime64],
        open: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        volume: Optional[NDArray[np.float64]],
        vwap: Optional[NDArray[np.float64]],
        **kwargs,
    ):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self._custom_col_data = kwargs

    def __getattr__(self, attr):
        if self._custom_col_data and attr in self._custom_col_data:
            return self._custom_col_data[attr]
        raise AttributeError(f"Attribute {attr!r} not found.")


def to_datetime(
    date: Union[str, datetime, np.datetime64, pd.Timestamp],
) -> datetime:
    """Converts ``date`` to :class:`datetime`."""
    if isinstance(date, pd.Timestamp):
        return date.to_pydatetime()  # type: ignore[union-attr]
    elif isinstance(date, datetime):
        return date  # type: ignore[return-value]
    elif isinstance(date, str):
        return pd.to_datetime(date).to_pydatetime()
    elif isinstance(date, np.datetime64):
        return pd.Timestamp(date).to_pydatetime()
    else:
        raise TypeError(f"Unsupported date type: {type(date)}")


def to_decimal(value: Union[int, float, Decimal]) -> Decimal:
    """Converts ``value`` to :class:`Decimal`."""
    value_type = type(value)
    if value_type == Decimal:
        return value  # type: ignore[return-value]
    elif value_type is int:
        return Decimal(value)
    return Decimal(str(value))


def parse_timeframe(timeframe: str) -> list[tuple[int, str]]:
    """Parses timeframe string with the following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        ``list`` of ``tuple[int, str]``, where each tuple contains an ``int``
        value and ``str`` unit of one of the following: ``sec``, ``min``,
        ``hour``, ``day``, ``week``.
    """
    parts = _tf_pattern.findall(timeframe)
    if not parts or len(parts) != len(timeframe.split()):
        raise ValueError("Invalid timeframe format.")
    result = []
    units = frozenset(_tf_abbr.values())
    seen_units = set()
    for part in parts:
        unit = part[1].lower()
        if unit in _tf_abbr:
            unit = _tf_abbr[unit]
        if unit not in units:
            raise ValueError("Invalid timeframe format.")
        if unit in seen_units:
            raise ValueError("Invalid timeframe format.")
        result.append((int(part[0]), unit))
        seen_units.add(unit)
    return result


def to_seconds(timeframe: Optional[str]) -> int:
    """Converts a timeframe string to seconds, where ``timeframe`` supports the
    following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        The converted number of seconds.
    """
    if not timeframe:
        return 0
    seconds = {
        "sec": 1,
        "min": 60,
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
    }
    return sum(
        part[0] * seconds[part[1]] for part in parse_timeframe(timeframe)
    )


def quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series:
    """Quantizes a :class:`pandas.DataFrame` column by rounding values to the
    nearest cent.

    Returns:
        The quantized column converted to ``float`` values.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found in DataFrame.")
    df = df[~df[col].isna()]
    values = df[col]
    if round:
        values = values.apply(lambda d: d.quantize(_CENTS, ROUND_HALF_UP))
    return values.astype(float)


def verify_data_source_columns(df: pd.DataFrame):
    """Verifies that a :class:`pandas.DataFrame` contains all of the
    columns required by a :class:`pybroker.data.DataSource`.
    """
    required_cols = (
        DataCol.SYMBOL,
        DataCol.DATE,
        DataCol.OPEN,
        DataCol.HIGH,
        DataCol.LOW,
        DataCol.CLOSE,
    )
    missing = []
    for col in required_cols:
        if col.value not in df.columns:
            missing.append(col.value)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing!r}")


def verify_date_range(start_date: datetime, end_date: datetime):
    """Verifies date range bounds."""
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be on or before end_date "
            f"({end_date})."
        )


def default_parallel() -> Parallel:
    """Returns a :class:`joblib.Parallel` instance with ``n_jobs`` equal to
    the number of CPUs on the host machine.
    """
    return Parallel(n_jobs=os.cpu_count(), prefer="processes", backend="loky")


def get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]:
    """Returns sorted unique values from a DataFrame column of dates.
    Guarantees compatability between Pandas 1 and 2.
    """
    result = col.unique()
    # TODO: Remove after Pandas 1.0 is no longer supported.
    if hasattr(result, "to_numpy"):
        result = result.to_numpy()
    result.sort()
    return result
