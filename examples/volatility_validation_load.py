from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner.avoid_overfitting.volatility import Tuner
from custom_client import BinanceTradingClient
from quick_trade.tuner import bests_to_config
from quick_trade.plots import TraderGraph, make_trader_figure

class TrainClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1h',
                            start_type: str = '%d %b %Y'):
        client = BinanceTradingClient()
        df = client.get_data_historical(ticker=ticker, interval=interval)
        return df[:-10000]

class ValidationClient(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1m',
                            start_type: str = '%d %b %Y'):

        client = BinanceTradingClient()
        df = client.get_data_historical(ticker=ticker, interval=interval)[-10000:]
        return df


tuner = Tuner(BinanceTradingClient())
tuner.load_tunes(path='volatility_validation/returns-{}.json')
tuner.resorting()

config = bests_to_config(tuner.get_best(5))

test_trader = ExampleStrategies(interval='1h')
test_trader.set_client(ValidationClient())
test_trader.connect_graph(TraderGraph(make_trader_figure(width=1400, height=1000, row_heights=[1, 20, 1])))

test_trader.log_deposit()
test_trader.multi_backtest(test_config=config,
                           commission=0.075)
