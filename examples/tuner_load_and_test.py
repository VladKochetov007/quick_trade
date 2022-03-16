from quick_trade.tuner import bests_to_config
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner.avoid_overfitting.volatility import Tuner
from custom_client import BinanceTradingClient
from quick_trade.plots import TraderGraph, make_trader_figure

client = BinanceTradingClient()

tuner = Tuner(client=client,
              initialize_tuners=False)
tuner.load_tunes("volatility_tuner_all_binance_history/returns-{}.json")
tuner.resorting('Sortino ratio')

trader = ExampleStrategies(interval='1h')

trader.set_client(client)
trader.connect_graph(TraderGraph(make_trader_figure(700, 1400, row_heights=[1, 20, 1])))

bests = tuner.get_best(5)
config = bests_to_config(bests)
trader.log_deposit()
trader.multi_backtest(test_config=config,
                      commission=0.075)
