from quick_trade.trading_sys import ExampleStrategies
from quick_trade.plots import make_trader_figure, TraderGraph
from custom_client import BinanceTradingClient

class Portfolio(ExampleStrategies):
    def DP2(self, sl=400, tp=3000):
        self.DP_2_strategy(tp=sl,
                           sl=tp)
        self.inverse_strategy()

    def DP2I(self, sl=400, tp=3000):
        self.DP_2_strategy(tp=tp,
                           sl=sl)

    def BB(self, window=80, dev=1):
        self.strategy_bollinger_breakout(window=window,
                                         window_dev=dev)

    def SMA3(self, s=1500, m=400, f=60):
        self.strategy_3_sma(fast=f,
                            slow=s,
                            mid=m)


timeframe = '15m'

figure = make_trader_figure(height=700, width=1400, row_heights=[1, 20, 1])
graph = TraderGraph(figure=figure)
client = BinanceTradingClient()

trader = Portfolio(interval=timeframe)

trader.connect_graph(graph)
trader.log_deposit()
trader.log_data()
trader.set_client(client)

config = {
    'BTC/USDT': [
        {'DP2': {}},
    ],
    'ETH/USDT': [
        {'DP2': {}},
    ],
    'SOL/USDT': [
        {'DP2': dict(sl=1000)},
        {'DP2': dict(sl=700)},
    ]
}
trader.multi_backtest(test_config=config, deposit=300, commission=0.075)
