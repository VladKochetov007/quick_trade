from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner.avoid_overfitting.volatility import Tuner, split_tickers_volatility
from custom_client import BinanceTradingClient
from quick_trade.tuner import Arange, Choise, GeometricProgression
from quick_trade.tuner import bests_to_config
from quick_trade.plots import BasePlotlyGraph, make_figure

class TrainClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '30m',
                            start_type: str = '%d %b %Y'):
        df = super().get_data_historical(ticker=ticker, interval=interval)
        return df[len(df)//3*2:]

class ValidationClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '30m',
                            start_type: str = '%d %b %Y'):
        df = super().get_data_historical(ticker=ticker, interval=interval)
        return df[:len(df)//3*2]


tickers = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'ADA/USDT', 'BNB/USDT', 'XRP/USDT', 'XMR/USDT']
params = {
    'strategy_bollinger_breakout':
        [
            {
                'window': Arange(10, 200, 10),
                'window_dev': Choise([0.5, 1, 1.5]),
                'plot': False
            }
        ],
}
tuner = Tuner(TrainClient(),
              split_tickers_volatility(tickers=tickers,
                                       client=TrainClient()),
              intervals=['30m'],
              strategies_kwargs=params)
tuner.tune(ExampleStrategies,
           update_json_path='volatility_validation3/returns-{}.json',
           commission=0.075)
tuner.sort_tunes('profit/deviation ratio')

config = bests_to_config(tuner.get_best(2))
fig = BasePlotlyGraph(make_figure(rows=2))

train_trader = ExampleStrategies(interval='30m')
train_trader.set_client(TrainClient())

val_trader = ExampleStrategies(interval='30m')
val_trader.set_client(ValidationClient())

for backtester in [train_trader, val_trader]:
    backtester.connect_graph()

    backtester.multi_backtest(test_config=config,
                              commission=0.075,
                              show=False)

fig.plot_line(line=train_trader.deposit_history,
             name='best train',
             color='red')

fig.plot_line(line=val_trader.deposit_history,
             name='best train validation',
             color='green',
             _row=2)
fig.log_y(1, 1)
fig.log_y(2, 1)
fig.show()

