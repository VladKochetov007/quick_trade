from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner.avoid_overfitting.volatility import Tuner, split_tickers_volatility
from custom_client import BinanceTradingClient
from quick_trade.tuner import Arange, Choise, GeometricProgression
from quick_trade.tuner import bests_to_config

class TrainClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1h',
                            start_type: str = '%d %b %Y'):
        df = super().get_data_historical(ticker=ticker, interval=interval)
        return df[:len(df)//2]

class ValidationClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1m',
                            start_type: str = '%d %b %Y'):
        df = super().get_data_historical(ticker=ticker, interval=interval)
        return df[len(df)//2:]


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
    'strategy_macd':
        [
            {
                'slow': Arange(10, 200, 10),
                'fast': Arange(10, 200, 10),
                '_RULES_': 'kwargs["slow"] > kwargs["fast"]'
            }
        ],
    'strategy_parabolic_SAR':
        [
            {
                'plot': False,
                'step': GeometricProgression(0.001, 1, 1.1),
                'max_step': GeometricProgression(0.01, 10, 1.1),
                '_RULES_': 'kwargs["step"] <= kwargs["max_step"]'
            }
        ],
    'strategy_supertrend':
        [
            {
                'plot': False,
                'multiplier': GeometricProgression(1, 15, 1.1),
                'length': Arange(1, 100, 5),
            }
        ],
}
tuner = Tuner(TrainClient(),
              split_tickers_volatility(tickers=tickers),
              intervals=['1h'],
              strategies_kwargs=params)
tuner.tune(ExampleStrategies,
           update_json_path='volatility_validation2/returns-{}.json',
           commission=0.075)
tuner.sort_tunes('profit/deviation ratio')

config = bests_to_config(tuner.get_best(10))

test_trader = ExampleStrategies(interval='1h')
test_trader.set_client(ValidationClient())
test_trader.connect_graph()

test_trader.multi_backtest(test_config=config,
                           commission=0.075)
