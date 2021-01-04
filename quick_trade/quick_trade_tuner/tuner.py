import itertools
from collections import defaultdict
from typing import Iterable, Dict, Any, List

import numpy as np
from quick_trade.brokers import TradingClient
from . import core


class QuickTradeTuner(object):
    def __init__(self,
                 client: TradingClient,
                 tickers: Iterable,
                 intervals: Iterable,
                 starts: Iterable,
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None):
        """

        :param client: trading client
        :param tickers: ticker
        :param intervals: list of intervals -> ['1m', '4h'...]
        :param starts: starts for client.get_data_historical (['2 Dec 2020', '3 Sep 1970'])
        :param strategies_kwargs: kwargs for strategies: {'strategy_supertrend': [{'multiplier': 10}]}, you can use Choice, Linspace, Arange as argument's value
        """
        strategies_kwargs = core.transform_all_tunable_values(strategies_kwargs)
        strategies = list(strategies_kwargs.keys())
        self.strategies_and_kwargs: List[str] = []
        self._strategies = []
        self._frames_data: tuple = tuple(itertools.product(tickers, intervals, starts))
        self.client = client
        for strategy in strategies:
            for kwargs in strategies_kwargs[strategy]:
                self._strategies.append([strategy, kwargs])

    def tune(self, your_trading_class, backtest_kwargs=dict(plot=False, print_out=False)) -> dict:
        def best():
            return defaultdict(best)

        self.result_tunes = best()
        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            start = data[2]

            df = self.client.get_data_historical(ticker=ticker,
                                                 interval=interval,
                                                 start=start)
            for strategy, kwargs in self._strategies:
                trader = your_trading_class(ticker=ticker, df=df, interval=interval)
                trader._get_attr(strategy)(**kwargs)
                trader.backtest(**backtest_kwargs)

                __ = str(kwargs).replace(": ", "=").replace("'", "").strip("{").strip("}")
                strat_kw = f'{strategy}({__})'
                self.strategies_and_kwargs.append(strat_kw)
                self.result_tunes[ticker][interval][start][strat_kw]['winrate'] = trader.winrate
                self.result_tunes[ticker][interval][start][strat_kw]['trades'] = trader.trades
                self.result_tunes[ticker][interval][start][strat_kw]['losses'] = trader.losses
                self.result_tunes[ticker][interval][start][strat_kw]['profits'] = trader.profits
                self.result_tunes[ticker][interval][start][strat_kw]['percentage year profit'] = trader.year_profit

        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            start = data[2]
            self.result_tunes = dict(self.result_tunes)
            self.result_tunes[ticker] = dict(self.result_tunes[ticker])
            self.result_tunes[ticker][interval] = dict(self.result_tunes[ticker][interval])
            self.result_tunes[ticker][interval][start] = dict(self.result_tunes[ticker][interval][start])
            for strategy in self.strategies_and_kwargs:
                self.result_tunes[ticker][interval][start][strategy] = dict(
                    self.result_tunes[ticker][interval][start][strategy])

        return self.result_tunes

    def filter_tunes(self, sort_by: str = 'percentage year profit') -> dict:
        filtered = {}
        for ticker, tname in zip(self.result_tunes.values(), self.result_tunes):
            for interval, iname in zip(ticker.values(), ticker):
                for start, sname in zip(interval.values(), interval):
                    for strategy, stratname in zip(start.values(), start):
                        filtered[
                            f'ticker: {tname}, interval: {iname}, start(period): {sname} :: {stratname}'] = strategy
        return {k: v for k, v in sorted(filtered.items(), key=lambda x: -x[1][sort_by])}


class Choise(core.TunableValue):
    def __init__(self, values: Iterable[Any]):
        self.values = values


class Arange(core.TunableValue):
    def __init__(self, min_value, max_value, step):
        self.values = np.arange(min_value, max_value + step, step)


class Linspace(core.TunableValue):
    def __init__(self, start, stop, num):
        self.values = np.linspace(start=start, stop=stop, num=num)
