from quick_trade.tuner.avoid_overfitting import WalkForward
from custom_client import BinanceTradingClient
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner import GeometricProgression

from quick_trade.plots import BasePlotlyGraph, make_figure

config = {
    'strategy_parabolic_SAR':
        [
            {
                'plot': False,
                'step': GeometricProgression(0.005, 0.5, 1.3),
                'max_step': GeometricProgression(0.01, 2, 1.3),
                '_RULES_': 'kwargs["step"] <= kwargs["max_step"] and kwargs["step"]*3 <= kwargs["max_step"]'
            }
        ],
}

graph = BasePlotlyGraph(make_figure(700, 1400))

client = BinanceTradingClient()
walkforward_optimizer = WalkForward(client=client)

walkforward_optimizer.run_analysis('ETH/BTC',
                                   '30m',
                                   config=config,
                                   trader_instance=ExampleStrategies,
                                   sort_by='profit/deviation ratio',
                                   commission=0.075)


graph.plot_line(line=walkforward_optimizer.equity(),
                name='walk-forward optimized strategy profit',
                width=2.5,
                color='white')
graph.log_y()
graph.show()
