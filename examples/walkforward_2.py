from quick_trade.tuner.avoid_overfitting import WalkForward
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner import Arange
from custom_client import BinanceTradingClient

from quick_trade.plots import BasePlotlyGraph, make_figure

config = {
    'strategy_bollinger_breakout':
        [
            {
                'plot': False,
                'window': Arange(10, 200, 20),
                'window_dev': 1
            }
        ]
}

graph = BasePlotlyGraph(make_figure(700, 1400))

client = BinanceTradingClient()
walkforward_optimizer = WalkForward(client=client)

walkforward_optimizer.run_analysis('BTC/USDT',
                                   '2h',
                                   config=config,
                                   trader_instance=ExampleStrategies,
                                   sort_by='profit/deviation ratio',
                                   commission=0.075)


graph.plot_line(line=walkforward_optimizer.equity(),
                name='walk-forward analysis',
                width=2.5,
                color='white')
graph.log_y()
graph.show()
