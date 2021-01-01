import itertools
from collections import defaultdict
from typing import Iterable

from quick_trade.trading_sys import Trader
from quick_trade.brokers import TradingClient


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

    def tune(self, your_trading_class):
        def best():
            return defaultdict(best)

        self.best_tunes = best()
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
                trader.backtest(plot=False, print_out=False)

                self.best_tunes[ticker][interval][start][strategy]['winrate'] = trader.winrate
                self.best_tunes[ticker][interval][start][strategy]['trades'] = trader.trades
                self.best_tunes[ticker][interval][start][strategy]['losses'] = trader.losses
                self.best_tunes[ticker][interval][start][strategy]['profits'] = trader.profits
                self.best_tunes[ticker][interval][start][strategy]['percentage year profit'] = trader.year_profit

        for data in self._frames_data:
            ticker = data[0]
            interval = data[1]
            start = data[2]
            self.best_tunes = dict(self.best_tunes)
            self.best_tunes[ticker] = dict(self.best_tunes[ticker])
            self.best_tunes[ticker][interval] = dict(self.best_tunes[ticker][interval])
            self.best_tunes[ticker][interval][start] = dict(self.best_tunes[ticker][interval][start])
            for strategy in self._strategies:
                self.best_tunes[ticker][interval][start][strategy] = dict(
                    self.best_tunes[ticker][interval][start][strategy])

        return self.best_tunes
