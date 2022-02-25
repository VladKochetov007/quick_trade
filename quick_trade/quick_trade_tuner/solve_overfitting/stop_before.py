import pandas as pd

class Analyzer(object):
    def __init__(self, train: dict, val: dict, sorted_by: str = 'percentage year profit'):
        self.train_data = train
        self.profit_keys = list(train.keys())[::-1]
        self.validation_data = val
        self.sorted_by = sorted_by

    def generate_frame(self):
        self.frame = pd.DataFrame(index=self.profit_keys)
        self.frame['train'] = [self.train_data[key][self.sorted_by] for key in self.profit_keys]
        self.frame['validation'] = [self.validation_data[key][self.sorted_by] for key in self.profit_keys]

if __name__ == '__main__':
    # I know, it's very bad code
    import matplotlib.pyplot as plt
    def part_1(train=True):
        import ccxt
        import time
        import quick_trade.quick_trade.trading_sys as qtr
        import pandas as pd
        import datetime
        import typing
        from typing import Dict
        from quick_trade.quick_trade import brokers
        from binance.client import Client
        from quick_trade.quick_trade import utils
        from quick_trade.quick_trade.plots import QuickTradeGraph, make_figure
        import numpy as np
        from quick_trade.quick_trade import strategy

        class BinanceTradingClient(Client, brokers.TradingClient):
            @utils.wait_success
            def get_data_historical(self,
                                    ticker: str = None,
                                    limit: int = 1000,
                                    interval: str = '1m',
                                    start_type: str = '%d %b %Y'):
                try:
                    # print(ticker)
                    if not USENEW:
                        with open('../../../../dataframes/' + ticker.replace('/', '') + f'{interval}.csv') as file:
                            df = pd.read_csv(file)
                            # print('in dir')
                    else:
                        # print('downloading')
                        raise ValueError
                except Exception as e:
                    ticker = ticker.replace('/', '')
                    start_date = datetime.datetime.strptime('5 Aug 2000', start_type)

                    today = datetime.datetime.now()
                    frames = self.get_historical_klines(ticker,
                                                        interval,
                                                        start_date.strftime("%d %b %Y %H:%M:%S"),
                                                        today.strftime("%d %b %Y %H:%M:%S"),
                                                        1000)
                    data = pd.DataFrame(frames,
                                        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av',
                                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                    df = data.astype(float)
                    df.to_csv('../../../../dataframes/' + ticker.replace('/', '_') + f'{interval}.csv')
                if train:
                    return df[:(len(df)*2//3)]
                return df[(len(df)*2//3):]

        USENEW = False

        from quick_trade.quick_trade.quick_trade_tuner.tuner import QuickTradeTuner, Arange, Linspace
        import datetime

        params = {
            'strategy_bollinger_breakout':
                [
                    dict(
                        window=Arange(5, 300, 5),
                        window_dev=Linspace(0.1, 2, 10),
                        plot=False
                    )
                ],
        }
        tuner = QuickTradeTuner(BinanceTradingClient(),
                                tickers=['BTC/USDT'],
                                intervals=['1h'],
                                limits=[1000],
                                strategies_kwargs=params,
                                multi_backtest=False)
        tuner.tune(qtr.ExampleStrategies, update_json=False)
        tuner.sort_tunes('profit/deviation ratio')
        if train:
            tuner.save_tunes('../../../../stopbeforeoverfitting2/train.json')
        else:
            tuner.save_tunes('../../../../stopbeforeoverfitting2/validation.json')

    part_1(False)
    part_1(True)
    from quick_trade.quick_trade._saving import read_json
    a = Analyzer(read_json('../../../../stopbeforeoverfitting2/train.json'),
                 read_json('../../../../stopbeforeoverfitting2/validation.json'),
                 'profit/deviation ratio')
    a.generate_frame()
    print(a.frame)
    a.frame.plot()
    plt.show()
