from quick_trade.tuner.avoid_overfitting import WalkForward
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner import Arange, Choise
from custom_client import BinanceTradingClient

from quick_trade.plots import BasePlotlyGraph, make_figure

tickers = ['BTC/USDT',
           'MANA/USDT',
           'ETH/USDT',
           'LTC/USDT',
           'LUNA/USDT',
           'GALA/USDT',
           'BNB/USDT',
           'XRP/USDT',
           'ADA/USDT',
           'SOL/USDT',
           'AVAX/USDT',
           'BCH/USDT',
           'XMR/USDT',
           'SLP/USDT',
           'CELO/USDT',
           'BAT/USDT',
           'FTM/USDT']
timeframes = ['1h', '30m', '2h', '15m']
configs = [
    {
        'strategy_bollinger_breakout':
            [
                {
                    'plot': False,
                    'window': Arange(10, 300, 20),
                    'window_dev': Choise([0.2, 0.5, 0.7, 1, 1.3, 1.5, 2])
                }
            ]
    },
    {
        'strategy_bollinger_breakout':
            [
                {
                    'plot': False,
                    'window': Arange(10, 300, 10),
                    'window_dev': Choise([0.2, 0.5, 0.7, 1, 1.3, 1.5, 2])
                }
            ]
    },
    {
        'strategy_bollinger_breakout':
            [
                {
                    'plot': False,
                    'window': Arange(10, 300, 30),
                    'window_dev': Choise([0.2, 1, 1.3])
                }
            ]
    },
    {
        'strategy_bollinger_breakout':
            [
                {
                    'plot': False,
                    'window': Arange(10, 300, 40),
                    'window_dev': Choise([0.2, 0.5, 0.7, 1, 1.3, 1.5, 2])
                }
            ]
    },
]

graph = BasePlotlyGraph(make_figure(700, 1400))

client = BinanceTradingClient()
walkforward_optimizer = WalkForward(client=client)

for ticker, timeframe, config in zip(
        tickers * len(configs) * len(timeframes),
        timeframes * len(tickers) * len(configs),
        configs * len(tickers) * len(timeframes),
):
    walkforward_optimizer.run_analysis(ticker,
                                       timeframe,
                                       config=config,
                                       trader_instance=ExampleStrategies,
                                       sort_by='profit/deviation ratio',
                                       commission=0.075,
                                       use_tqdm=False)

    print(ticker, timeframe, config, walkforward_optimizer.info(), end='\n\n\n\n\n\n\n')
