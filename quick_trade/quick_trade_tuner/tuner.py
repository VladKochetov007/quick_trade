from collections import defaultdict
from itertools import product
from json import dump
from json import load
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

from numpy import arange
from numpy import isnan
from numpy import linspace
from pandas import DataFrame
from tqdm import tqdm

from .core import TunableValue
from .core import transform_all_tunable_values
from .. import utils
from ..brokers import TradingClient


class QuickTradeTuner(object):
    def __init__(self,
                 client: TradingClient,
                 tickers: Iterable[str],
                 intervals: Iterable[str],
                 limits: Iterable[int],
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None,
                 multi_backtest: bool = True):
        """

        :param client: trading client
        :param tickers: tickers
        :param intervals: list of intervals -> ['1m', '4h'...]
        :param limits: limits for client.get_data_historical ([1000, 700...])
        :param strategies_kwargs: kwargs for strategies: {'strategy_supertrend': [{'multiplier': 10}]}, you can use Choice, Linspace, Arange as argument's value and recourse it. You can also set rules for arranging arguments for each strategy by using _RULES_ and kwargs to access the values of the arguments.

        """
        strategies_kwargs = transform_all_tunable_values(strategies_kwargs)
        strategies = list(strategies_kwargs.keys())
        self.strategies_and_kwargs: List[str] = []
        self._strategies = []
        self.tickers = tickers
        self.multi_test: bool = multi_backtest
        if multi_backtest:
            tickers = [tickers]
        self._frames_data: tuple = tuple(product(tickers, intervals, limits))
        self.client = client
        for strategy in strategies:
            for kwargs in strategies_kwargs[strategy]:
                if eval(kwargs.get('_RULES_', 'True')):
                    without_rules = kwargs.copy()
                    if '_RULES_' in kwargs.keys():
                        without_rules.pop('_RULES_')
                    self._strategies.append([strategy, without_rules])

    def tune(
            self,
            your_trading_class,
            use_tqdm: bool = True,
            update_json: bool = True,
            update_json_path: str = 'returns.json',
            **backtest_kwargs
    ) -> dict:
        backtest_kwargs['plot'] = False
        backtest_kwargs['show'] = False
        backtest_kwargs['print_out'] = False
        if use_tqdm:
            bar: tqdm = tqdm(
                total=len(self._strategies) * len(self._frames_data)
            )

        def get_dict():
            return defaultdict(get_dict)

        self.result_tunes = get_dict()
        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            limit = data[2]
            if not self.multi_test:
                df = self.client.get_data_historical(ticker=ticker,
                                                     interval=interval,
                                                     limit=limit)
            else:
                df = DataFrame()
            for strategy, kwargs in self._strategies:
                trader = your_trading_class(ticker='ALL/ALL' if self.multi_test else ticker, df=df, interval=interval)
                trader.set_client(self.client)

                if self.multi_test:
                    backtest_kwargs['limit'] = limit
                    kwargs_m = {}
                    for ticker_ in ticker:
                        kwargs_m[ticker_] = [{strategy: kwargs}]
                    trader.multi_backtest(test_config=kwargs_m,
                                          **backtest_kwargs)
                else:
                    trader._get_attr(strategy)(**kwargs)
                    trader.backtest(**backtest_kwargs)

                arguments = str(kwargs).replace(": ", "=").replace("'", "").strip("{").strip("}")
                strat_kw = f'{strategy}({arguments})'
                self.strategies_and_kwargs.append(strat_kw)
                if self.multi_test:
                    old_tick = ticker
                    ticker = 'ALL'
                utils.logger.debug('testing %s ... :', strat_kw)
                self.result_tunes[ticker][interval][limit][strat_kw]['winrate'] = trader.winrate
                self.result_tunes[ticker][interval][limit][strat_kw]['trades'] = trader.trades
                self.result_tunes[ticker][interval][limit][strat_kw]['losses'] = trader.losses
                self.result_tunes[ticker][interval][limit][strat_kw]['profits'] = trader.profits
                self.result_tunes[ticker][interval][limit][strat_kw]['percentage year profit'] = trader.year_profit
                self.result_tunes[ticker][interval][limit][strat_kw]['mean deviation'] = trader.mean_deviation
                if self.multi_test:
                    ticker = old_tick
                if use_tqdm:
                    bar.update(1)
                if update_json:
                    self.save_tunes(path=update_json_path)

        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            limit = data[2]
            self.result_tunes = dict(self.result_tunes)
            if self.multi_test:
                old_tick = ticker
                ticker = 'ALL'
            self.result_tunes[ticker] = dict(self.result_tunes[ticker])
            self.result_tunes[ticker][interval] = dict(self.result_tunes[ticker][interval])
            self.result_tunes[ticker][interval][limit] = dict(self.result_tunes[ticker][interval][limit])
            for strategy in self.strategies_and_kwargs:
                self.result_tunes[ticker][interval][limit][strategy] = dict(
                    self.result_tunes[ticker][interval][limit][strategy])
            if self.multi_test:
                ticker = old_tick

        return self.result_tunes

    def sort_tunes(self, sort_by: str = 'percentage year profit', drop_na: bool = True) -> dict:
        utils.logger.debug('sorting tunes')
        not_filt = self.result_tunes
        self.result_tunes = dict()
        for ticker, tname in zip(not_filt.values(), not_filt):
            for interval, iname in zip(ticker.values(), ticker):
                for start, sname in zip(interval.values(), interval):
                    for strategy, stratname in zip(start.values(), start):
                        self.result_tunes[f'ticker: {tname}, interval: {iname}, limit: {sname} :: {stratname}'] = strategy
        return self.resorting(sort_by=sort_by, drop_na=drop_na)

    def resorting(self, sort_by: str = 'percentage year profit', drop_na: bool = True):
        if drop_na:
            for key, data in self.result_tunes.copy().items():
                if isnan(data[sort_by]):
                    del self.result_tunes[key]
        self.result_tunes = {k: v for k, v in sorted(self.result_tunes.items(), key=lambda x: -x[1][sort_by])}
        utils.logger.debug('tunes are sorted')
        return self.result_tunes

    def save_tunes(self, path: str = 'returns.json'):
        utils.logger.debug('saving tunes in "%s"', path)
        with open(path, 'w') as file:
            dump(self.result_tunes, file)

    def load_tunes(self, path: str = 'returns.json'):
        utils.logger.debug('loading tunes from "%s"', path)
        with open(path, 'r') as file:
            self.result_tunes = load(file)


class Choise(TunableValue):
    pass


class Arange(TunableValue):
    def __init__(self, min_value, max_value, step):
        self.values = arange(min_value, max_value + step, step).astype('int').tolist()


class Linspace(TunableValue):
    def __init__(self, start, stop, num):
        self.values = linspace(start=start, stop=stop, num=num).astype('float').tolist()
