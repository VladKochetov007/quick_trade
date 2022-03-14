from quick_trade.quick_trade_tuner.avoid_overfitting.validation_analysis import ValidationTuner, Analyzer
from quick_trade.quick_trade_tuner.tuner import Arange
import pandas as pd
import datetime
from quick_trade import brokers
from binance.client import Client  # pip3 install python-binance
from quick_trade import utils
from quick_trade.trading_sys import ExampleStrategies
from quick_trade import strategy
from quick_trade.plots import ValidationAnalysisGraph


class MyTrader(ExampleStrategies):
    @strategy
    def strategy_bollinger_breakout(self,
                                    to_mid: bool = False,
                                    to_opposite: bool = False,
                                    window=80,
                                    window_dev=1):
        super().strategy_bollinger_breakout(plot=False,
                                            to_opposite=to_opposite,
                                            to_mid=to_mid,
                                            window=window,
                                            window_dev=window_dev)
        return self.returns


class BinanceTradingClient(Client, brokers.TradingClient):
    @utils.wait_success
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1m',
                            start_type: str = '%d %b %Y'):
        try:
            with open('dataframes/' + ticker.replace('/', '') + f'{interval}.csv') as file:
                df = pd.read_csv(file)
        except:
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
            df.to_csv('dataframes/' + ticker.replace('/', '_') + f'{interval}.csv')
        return df


params = {
    'strategy_bollinger_breakout':
        [
            dict(
                window=Arange(1, 10, 1),
                window_dev=1
            )
        ],
}

t = ValidationTuner(BinanceTradingClient(),
                    tickers=['BTC/USDT', 'ETH/USDT', 'ETC/USDT', 'LTC/USDT'],
                    intervals=['1h'],
                    limits=[1000],
                    strategies_kwargs=params,
                    multi_backtest=True,
                    validation_split=1/3)
t.tune(MyTrader, commission=0.075, val_json_path='val.json', train_json_path='train.json')
validator = Analyzer(train='train.json',
                     val='val.json',
                     sort_by='calmar ratio')
fig = ValidationAnalysisGraph()
fig.connect_analyzer(validator)
validator.generate_frame()
validator.plot_frame()
validator.fig.show()
