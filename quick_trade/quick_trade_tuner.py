import itertools
from collections import defaultdict
from typing import Iterable

from quick_trade.brokers import TradingClient
from quick_trade.trading_sys import Trader


class QuickTradeTuner(object):
    def __init__(self,
                 client: TradingClient,
                 tickers: Iterable,
                 intervals: Iterable,
                 starts: Iterable,
                 strategies: Iterable[str]):
        self._frames_data: tuple = tuple(itertools.product(tickers, intervals, starts))
        self._strategies = strategies
        self.client = client

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
            for strategy in self._strategies:
                trader: Trader = your_trading_class(ticker=ticker, df=df, interval=interval)
                trader._get_attr(strategy)()
                trader.backtest(**backtest_kwargs)

                self.result_tunes[ticker][interval][start][strategy]['winrate'] = trader.winrate
                self.result_tunes[ticker][interval][start][strategy]['trades'] = trader.trades
                self.result_tunes[ticker][interval][start][strategy]['losses'] = trader.losses
                self.result_tunes[ticker][interval][start][strategy]['profits'] = trader.profits
                self.result_tunes[ticker][interval][start][strategy]['percentage year profit'] = trader.year_profit

        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            start = data[2]
            self.result_tunes = dict(self.result_tunes)
            self.result_tunes[ticker] = dict(self.result_tunes[ticker])
            self.result_tunes[ticker][interval] = dict(self.result_tunes[ticker][interval])
            self.result_tunes[ticker][interval][start] = dict(self.result_tunes[ticker][interval][start])
            for strategy in self._strategies:
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
