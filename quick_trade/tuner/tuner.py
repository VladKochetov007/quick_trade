from itertools import product
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from numpy import arange
from numpy import linspace
from pandas import DataFrame
from tqdm import tqdm

from .core import TunableValue
from .core import transform_all_tunable_values, resort_tunes
from .. import utils
from ..brokers import TradingClient
from .. import _saving
from .._code_inspect import format_arguments


class QuickTradeTuner(object):
    _frames_data: tuple
    def __init__(self,
                 client: TradingClient,
                 tickers: Iterable[str] = None,
                 intervals: Iterable[str] = None,
                 limits: Iterable = None,
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None,
                 multi_backtest: bool = True):
        """

        :param client: trading client
        :param tickers: tickers
        :param intervals: list of intervals -> ['1m', '4h'...]
        :param limits: limits for client.get_data_historical ([1000, 700...])
        :param strategies_kwargs: kwargs for strategies: {'strategy_supertrend': [{'multiplier': 10}]}, you can use Choice, Linspace, Arange as argument's value and recourse it. You can also set rules for arranging arguments for each strategy by using _RULES_ and kwargs to access the values of the arguments.

        """
        self.strategies_and_kwargs: List[str] = []
        self._strategies = []
        self.tickers = tickers
        self.multi_test: bool = multi_backtest
        if multi_backtest:
            tickers = [tickers]
        self.client = client
        if strategies_kwargs is None:
            strategies_kwargs = dict()
        if intervals is None:
            intervals = ['1h']
        if limits is None:
            limits = [1000]
        if tickers is None:
            tickers = []
        self._frames_data = tuple(product(tickers, intervals, limits))
        strategies_kwargs = transform_all_tunable_values(strategies_kwargs)
        strategies = list(strategies_kwargs.keys())
        for strategy in strategies:
            for kwargs in strategies_kwargs[strategy]:
                if eval(kwargs.get('_RULES_', 'True')):
                    without_rules = kwargs.copy()
                    if '_RULES_' in kwargs.keys():
                        without_rules.pop('_RULES_')
                    self._strategies.append([strategy, without_rules])

    def _get_df(self, ticker: str, interval: str, limit):
        return self.client.get_data_historical(ticker=ticker,
                                               interval=interval,
                                               limit=limit)

    def tune(
            self,
            trading_class,
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

        self.result_tunes = utils.recursive_dict()
        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            limit = data[2]
            if not self.multi_test:
                df = self._get_df(ticker=ticker,
                                  interval=interval,
                                  limit=limit)
            else:
                df = DataFrame()
                frames = {t: self._get_df(ticker=t, interval=interval, limit=limit) for t in ticker}
            for strategy, kwargs in self._strategies:
                trader = trading_class(ticker='ALL/ALL' if self.multi_test else ticker, df=df, interval=interval)
                trader.set_client(self.client)

                if self.multi_test:
                    backtest_kwargs['limit'] = limit
                    kwargs_m = {}
                    for ticker_ in ticker:
                        kwargs_m[ticker_] = [{strategy: kwargs}]
                    trader.multi_backtest(test_config=kwargs_m,
                                          _dataframes=frames,
                                          **backtest_kwargs)
                else:
                    trader._get_attr(strategy)(**kwargs)
                    trader.backtest(**backtest_kwargs)

                if self.multi_test:
                    old_tick = ticker
                    ticker = ' '.join(self.tickers)
                    strat_kw = format_arguments(strategy, kwargs=kwargs)
                else:
                    strat_kw = trader._registered_strategy
                self.strategies_and_kwargs.append(strat_kw)

                utils.logger.debug('testing %s ... :', strat_kw)

                for filter_name, filter_attr in utils.TUNER_CODECONF.items():
                    self.result_tunes[ticker][interval][limit][strat_kw][filter_name] = trader._get_attr(filter_attr)

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
                ticker = ' '.join(self.tickers)
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
        self.result_tunes = resort_tunes(tunes=self.result_tunes, sort_by=sort_by, drop_na=drop_na)
        return self.result_tunes

    def save_tunes(self, path: str = 'returns.json'):
        utils.logger.debug('saving tunes in "%s"', path)
        _saving.write_json(data=self.result_tunes, path=path, indent=utils.TUNER_INDENT)

    def load_tunes(self, path: str = 'returns.json', data: dict = {}):
        not_empty_dict = len(data.items())
        utils.logger.debug('loading tunes from "%s"', path if not_empty_dict else 'dict')
        if not_empty_dict:
            self.result_tunes = data
        else:
            self.result_tunes = _saving.read_json(path=path)

    def get_best(self, num: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
        return list(self.result_tunes.items())[:num]

    def get_worst(self, num: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
        return list(self.result_tunes.items())[-num:]


class Choise(TunableValue):
    pass


class Arange(TunableValue):
    def __init__(self, start, stop, step):
        self.values = arange(start, stop + step, step).astype('int').tolist()


class Linspace(TunableValue):
    def __init__(self, start, stop, num):
        self.values = linspace(start=start, stop=stop, num=num).astype('float').tolist()


class GeometricProgression(TunableValue):
    def __init__(self, start, stop, multiplier):
        val = start
        self.values = []
        while val <= stop:
            self.values.append(val)
            val *= multiplier
