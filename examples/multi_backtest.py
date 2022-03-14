from custom_client import BinanceTradingClient
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.plots import QuickTradeGraph, make_trader_figure


client = BinanceTradingClient()
graph = QuickTradeGraph(make_trader_figure(700, 1400, row_heights=[1, 20, 2]))

trader = ExampleStrategies(interval='1h')
trader.set_client(client)
trader.connect_graph(graph)
trader.log_deposit()

# volatility tuner results
group_1_cnfig = [
    {'strategy_bollinger_breakout': dict(plot=False, window=150, window_dev=0.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=105, window_dev=1.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=165, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=145, window_dev=0.5)},
]
group_2_config = [
    {'strategy_bollinger_breakout': dict(plot=False, window=195, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=165, window_dev=1.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=190, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=185, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=170, window_dev=1.5)},
]
group_3_config = [
    {'strategy_bollinger_breakout': dict(plot=False, window=200, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=170, window_dev=1.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=175, window_dev=1.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=90, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=165, window_dev=1.5)},
]
group_4_config = [
    {'strategy_bollinger_breakout': dict(plot=False, window=195, window_dev=0.5)},
    {'strategy_bollinger_breakout': dict(plot=False, window=190, window_dev=0.5)},
    {'strategy_bollinger_breakout': dict(window=165, window_dev=1)},
    {'strategy_bollinger_breakout': dict(plot=False, window=170, window_dev=1.5)},
]
config = {
    'ETH/USDT': group_1_cnfig,
    'LTC/USDT': group_1_cnfig,
    'BNB/USDT': group_1_cnfig,
    'BCH/USDT': group_1_cnfig,
    'XMR/USDT': group_1_cnfig,

    'MANA/USDT': group_2_config,
    'AVAX/USDT': group_2_config,

    'XRP/USDT': group_3_config,
    'ADA/USDT': group_3_config,
    'BAT/USDT': group_3_config,

}
trader.multi_backtest(test_config=config,
                      commission=0.075,
                      deposit=400)
