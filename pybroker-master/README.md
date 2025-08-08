<img src="https://github.com/edtechre/pybroker/blob/master/docs/_static/pybroker-logo.png?raw=true" alt="PyBroker">

[![python](https://img.shields.io/badge/python-v3-brightgreen.svg)](https://www.python.org/)
[![Apache 2.0 with Commons Clause](https://img.shields.io/badge/license-Apache%202.0%20Clause-green)](https://www.pybroker.com/en/latest/license.html)
[![Documentation Status](https://readthedocs.org/projects/pybroker/badge/?version=latest)](https://www.pybroker.com/en/latest/?badge=latest)
[![Package status](https://github.com/edtechre/pybroker/actions/workflows/main.yml/badge.svg?event=push)](https://github.com/edtechre/pybroker/actions)
[![Downloads](https://static.pepy.tech/badge/lib-pybroker)](https://pepy.tech/project/lib-pybroker)
[![Github stars](https://img.shields.io/github/stars/edtechre/pybroker?style=social)](https://github.com/edtechre/pybroker/)
[![Twitter](https://img.shields.io/twitter/follow/libpybroker?style=social)](https://twitter.com/intent/follow?screen_name=libpybroker)

## Algorithmic Trading in Python with Machine Learning

Are you looking to enhance your trading strategies with the power of Python and
machine learning? Then you need to check out **PyBroker**! This Python framework
is designed for developing algorithmic trading strategies, with a focus on
strategies that use machine learning. With PyBroker, you can easily create and
fine-tune trading rules, build powerful models, and gain valuable insights into
your strategy’s performance.

## Key Features

- A super-fast backtesting engine built in [NumPy](https://numpy.org/) and accelerated with [Numba](https://numba.pydata.org/).
- The ability to create and execute trading rules and models across multiple instruments with ease.
- Access to historical data from [Alpaca](https://alpaca.markets/), [Yahoo Finance](https://finance.yahoo.com/), [AKShare](https://github.com/akfamily/akshare), or from [your own data provider](https://www.pybroker.com/en/latest/notebooks/7.%20Creating%20a%20Custom%20Data%20Source.html).
- The option to train and backtest models using [Walkforward Analysis](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis), which simulates how the strategy would perform during actual trading.
- More reliable trading metrics that use randomized [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) to provide more accurate results.
- Caching of downloaded data, indicators, and models to speed up your development process.
- Parallelized computations that enable faster performance.

With PyBroker, you'll have all the tools you need to create winning trading
strategies backed by data and machine learning. Start using PyBroker today and
take your trading to the next level!

## Installation

PyBroker supports Python 3.9+ on Windows, Mac, and Linux. You can install
PyBroker using ``pip``:

```bash
   pip install -U lib-pybroker
```

Or you can clone the Git repository with:

```bash
   git clone https://github.com/edtechre/pybroker
```

## A Quick Example

Get a glimpse of what backtesting with PyBroker looks like with these code
snippets:

**Rule-based Strategy**:

```python
   from pybroker import Strategy, YFinance, highest

   def exec_fn(ctx):
      # Get the rolling 10 day high.
      high_10d = ctx.indicator('high_10d')
      # Buy on a new 10 day high.
      if not ctx.long_pos() and high_10d[-1] > high_10d[-2]:
         ctx.buy_shares = 100
         # Hold the position for 5 days.
         ctx.hold_bars = 5
         # Set a stop loss of 2%.
         ctx.stop_loss_pct = 2

   strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='7/1/2022')
   strategy.add_execution(
      exec_fn, ['AAPL', 'MSFT'], indicators=highest('high_10d', 'close', period=10))
   # Run the backtest after 20 days have passed.
   result = strategy.backtest(warmup=20)
```

**Model-based Strategy**:

```python
   import pybroker
   from pybroker import Alpaca, Strategy

   def train_fn(train_data, test_data, ticker):
      # Train the model using indicators stored in train_data.
      ...
      return trained_model

   # Register the model and its training function with PyBroker.
   my_model = pybroker.model('my_model', train_fn, indicators=[...])

   def exec_fn(ctx):
      preds = ctx.preds('my_model')
      # Open a long position given my_model's latest prediction.
      if not ctx.long_pos() and preds[-1] > buy_threshold:
         ctx.buy_shares = 100
      # Close the long position given my_model's latest prediction.
      elif ctx.long_pos() and preds[-1] < sell_threshold:
         ctx.sell_all_shares()

   alpaca = Alpaca(api_key=..., api_secret=...)
   strategy = Strategy(alpaca, start_date='1/1/2022', end_date='7/1/2022')
   strategy.add_execution(exec_fn, ['AAPL', 'MSFT'], models=my_model)
   # Run Walkforward Analysis on 1 minute data using 5 windows with 50/50 train/test data.
   result = strategy.walkforward(timeframe='1m', windows=5, train_size=0.5)
```

## User Guide

- [Getting Started with Data Sources](https://www.pybroker.com/en/latest/notebooks/1.%20Getting%20Started%20with%20Data%20Sources.html)
- [Backtesting a Strategy](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html)
- [Evaluating with Bootstrap Metrics](https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html)
- [Ranking and Position Sizing](https://www.pybroker.com/en/latest/notebooks/4.%20Ranking%20and%20Position%20Sizing.html)
- [Writing Indicators](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html)
- [Training a Model](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html)
- [Creating a Custom Data Source](https://www.pybroker.com/en/latest/notebooks/7.%20Creating%20a%20Custom%20Data%20Source.html)
- [Applying Stops](https://www.pybroker.com/en/latest/notebooks/8.%20Applying%20Stops.html)
- [Rebalancing Positions](https://www.pybroker.com/en/latest/notebooks/9.%20Rebalancing%20Positions.html)
- [Rotational Trading](https://www.pybroker.com/en/latest/notebooks/10.%20Rotational%20Trading.html)
- [FAQs](https://www.pybroker.com/en/latest/notebooks/FAQs.html)

## Online Documentation

[The full reference documentation is hosted at **www.pybroker.com**.](https://www.pybroker.com)

(For Chinese users: [中文文档](https://www.pybroker.com/zh_CN/latest/), courtesy of [Albert King](https://github.com/albertandking).)

## Contact

<img src="https://github.com/edtechre/pybroker/blob/master/docs/_static/email-image.png?raw=true">

## AI-Powered Stock News

Stay informed with AI-powered news on top trending stocks. See [www.trendninja.ai](https://www.trendninja.ai) for the latest updates!
