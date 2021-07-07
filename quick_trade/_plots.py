from typing import Union, List, Iterable, Sequence

import utils
from plotly.graph_objs import Figure, Scatter
from plotly.subplots import make_subplots


@utils.assert_logger
def make_figure(height: Union[int, float] = 900,
                width: Union[int, float] = 1300,
                template: str = 'plotly_dark',
                row_heights: List[Union[int, float]] = [10, 16, 7]) -> Figure:
    assert isinstance(height, (int, float)), 'height must be of type <int> or <float>'
    assert isinstance(width, (int, float)), 'width must be of type <int> or <float>'
    assert isinstance(template, str), 'template must be of type <str>'
    assert isinstance(row_heights, list), 'row_heights must be of type <List[int, float]>'
    for el in row_heights:
        assert isinstance(el, (int, float)), 'row_heights must be of type <List[int, float]>'

    fig = make_subplots(3, 1, row_heights=row_heights, )
    fig.update_layout(
        height=height,
        width=width,
        template=template,
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(
        title_text=utils.TIME_TITLE, row=3, col=1, color=utils.TEXT_COLOR)
    fig.update_yaxes(
        title_text=utils.DEPOSIT_TITLE, row=2, col=1, color=utils.TEXT_COLOR)
    fig.update_yaxes(
        title_text=utils.RETURNS_TITLE, row=3, col=1, color=utils.TEXT_COLOR)
    fig.update_yaxes(
        title_text=utils.DATA_TITLE, row=1, col=1, color=utils.TEXT_COLOR)
    return fig


class QuickTradeGraph(object):
    figure: Figure
    data_row: int = 1
    data_col: int = 1
    deposit_row: int = 2
    deposit_col: int = 1
    returns_row: int = 3
    returns_col: int = 1

    def __init__(self, trader, figure: Figure):
        self.figure = figure
        self.trader = trader

    def show(self, **kwargs):
        self.figure.show(**kwargs)

    def plot_line(self,
                  line: Iterable=None,
                  width: float = 1.0,
                  opacity: float = 1.0,
                  color: str = None,
                  name: str = None,
                  _row: int = 1,
                  _col: int = 1,
                  mode: str = 'lines'):
        self.figure.add_trace(
            row=_row,
            col=_col,
            trace=Scatter(
                y=line,
                name=name,
                mode=mode,
                opacity=opacity,
                line=dict(
                    color=color,
                    width=width
                )
            )
        )

    def plot_candlestick(self):
        self.figure.add_candlestick(
            open=self.trader.df['Open'],
            high=self.trader.df['High'],
            low=self.trader.df['Low'],
            close=self.trader.df['Close'],
            row=self.data_row,
            col=self.data_col,
            increasing_line_color=utils.DATA_UP_COLOR,
            decreasing_line_color=utils.DATA_DOWN_COLOR,
            name=utils.DATA_NAME.format(self.trader.ticker, self.trader.interval),
            opacity=utils.DATA_ALPHA
        )

    def plot_deposit(self,
                     deposit_history: Sequence):
        deposit_start = deposit_history[0]
        self.plot_line(line=deposit_history,
                       width=utils.DEPOSIT_WIDTH,
                       opacity=utils.DEPOSIT_ALPHA,
                       color=utils.DEPOSIT_COLOR,
                       name=utils.DEPOSIT_NAME.format(deposit_start))
        average_growth = self.trader.average_growth

    def plot_returns(self, returns: Sequence):
        self.plot_line(line=returns,
                       width=utils.RETURNS_WIDTH,
                       opacity=utils.RETURNS_ALPHA,
                       color=utils.RETURNS_COLOR,
                       name=utils.RETURNS_NAME,
                       _row=self.returns_row,
                       _col=self.returns_col)

if __name__ == "__main__":
    from quick_trade.brokers import TradingClient
    from trading_sys import Trader
    import ccxt

    client = TradingClient(ccxt.binance())
    t = Trader(df=client.get_data_historical('BTC/USDT'))
    g = QuickTradeGraph(trader=t,
                        figure=make_figure())
    g.plot_candlestick()
    g.plot_line([1, 3, 2, 4, 2, 4, 3],
                color='#fff',
                width=10,
                name='hm, quick-trade is cool',
                _row=2,
                opacity=0.3)
    g.plot_line([1, 23, 45, 68, 9, 86, 53, 34, 56, 78, 9, 8, 76, 5, 4, 3, 4, 57],
                color='#fff',
                width=3,
                name='really cool',
                _row=3)
    g.figure.show()
