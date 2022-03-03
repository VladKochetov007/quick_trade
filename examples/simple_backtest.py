from quick_trade.quick_trade.trading_sys import Trader
from quick_trade.quick_trade.brokers import TradingClient
from quick_trade.quick_trade.plots import make_trader_figure, QuickTradeGraph
from quick_trade.quick_trade.utils import BUY, SELL
from ta.trend import EMAIndicator, SMAIndicator
from quick_trade.quick_trade import strategy
from ccxt import binance


class MyTrader(Trader):
    @strategy
    def strategy_sma_ema_cross(self, plot=True, ema_len=20, sma_len=30):
        sma = SMAIndicator(self.df['Close'], sma_len).sma_indicator()  # pandas.Series
        ema = EMAIndicator(self.df['Close'], sma_len).ema_indicator()  # pandas.Series

        if plot:
            self.fig.plot_line(sma.values, name='SMA line', color='red')
            self.fig.plot_line(ema.values, name='EMA line', color='green')

        for simple, exponential in zip(sma.values, ema.values):
            if exponential > simple:
                self.returns.append(BUY)
            else:
                self.returns.append(SELL)
