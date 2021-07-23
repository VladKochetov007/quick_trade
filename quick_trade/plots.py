from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

from plotly.graph_objs import Figure
from plotly.graph_objs import Scatter
from plotly.subplots import make_subplots

from . import utils


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

    fig = make_subplots(3, 1, row_heights=row_heights)
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


class BaseGraph(object):
    _figure: Figure
    data_row: int = 1
    data_col: int = 1
    deposit_row: int = 2
    deposit_col: int = 1
    returns_row: int = 3
    returns_col: int = 1

    def __init__(self, figure: Figure):
        self._figure = figure

    def connect_trader(self, trader):
        self.trader = trader
        self.trader.fig = self
        utils.logger.info('new %s graph', self.trader)

    def show(self, **kwargs):
        self._figure.show(**kwargs)


class BasePlotlyGraph(BaseGraph):
    def plot_line(self,
                  line: Iterable = None,
                  index: Iterable = None,
                  width: float = 1.0,
                  opacity: float = 1.0,
                  color: str = None,
                  name: str = None,
                  _row: int = 1,
                  _col: int = 1,
                  mode: str = 'lines',
                  marker: str = None,
                  fill: str = None,
                  fill_color: str = None):
        self._figure.add_trace(
            row=_row,
            col=_col,
            trace=Scatter(
                x=index,
                y=line,
                name=name,
                mode=mode,
                opacity=opacity,
                fill=fill,
                fillcolor=fill_color,
                line=dict(
                    color=color,
                    width=width,
                ),
                marker=dict(
                    color=color,
                    size=width,
                    opacity=opacity,
                    symbol=marker
                )
            )
        )

    def plot_candlestick(self):
        self._figure.add_candlestick(
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

    def plot_area(self,
                  fast: Iterable = None,
                  slow: Iterable = None,
                  name_fast: str = None,
                  name_slow: str = None):
        self.plot_line(line=fast,
                       color=utils.SENKOU_SPAN_A_COLOR,
                       _row=self.data_row,
                       _col=self.data_col,
                       name=name_fast,
                       opacity=utils.SENKOU_SPAN_A_ALPHA)

        self.plot_line(line=slow,
                       fill='tonexty',
                       color=utils.SENKOU_SPAN_B_COLOR,
                       _row=self.data_row,
                       _col=self.data_col,
                       name=name_slow,
                       fill_color=utils.ICHIMOKU_CLOUD_COLOR,
                       opacity=utils.SENKOU_SPAN_B_ALPHA)


class QuickTradeGraph(BasePlotlyGraph):

    def plot_deposit(self):
        deposit_start = self.trader.deposit_history[0]
        self.plot_line(line=self.trader.deposit_history,
                       width=utils.DEPOSIT_WIDTH,
                       opacity=utils.DEPOSIT_ALPHA,
                       color=utils.DEPOSIT_COLOR,
                       name=utils.DEPOSIT_NAME.format(deposit_start),
                       _row=self.deposit_row,
                       _col=self.deposit_col)

        self.plot_line(line=self.trader.average_growth,
                       width=utils.AVERAGE_GROWTH_WIDTH,
                       opacity=utils.AVERAGE_GROWTH_ALPHA,
                       color=utils.AVERAGE_GROWTH_COLOR,
                       name=utils.AVERAGE_GROWTH_NAME,
                       _row=self.deposit_row,
                       _col=self.deposit_col)

    def plot_returns(self):
        self.plot_line(line=self.trader.returns_strategy_diff,
                       width=utils.RETURNS_WIDTH,
                       opacity=utils.RETURNS_ALPHA,
                       color=utils.RETURNS_COLOR,
                       name=utils.RETURNS_NAME,
                       _row=self.returns_row,
                       _col=self.returns_col)

    def plot_SL_TP_OPN(self):
        self.plot_line(line=self.trader._stop_losses,
                       width=utils.STOP_LOSS_WIDTH,
                       opacity=utils.STOP_LOSS_ALPHA,
                       color=utils.STOP_LOSS_COLOR,
                       name=utils.STOP_LOSS_NAME,
                       _row=self.data_row,
                       _col=self.data_col)

        self.plot_line(line=self.trader._take_profits,
                       width=utils.TAKE_PROFIT_WIDTH,
                       opacity=utils.TAKE_PROFIT_ALPHA,
                       color=utils.TAKE_PROFIT_COLOR,
                       name=utils.TAKE_PROFIT_NAME,
                       _row=self.data_row,
                       _col=self.data_col)

        self.plot_line(line=self.trader._open_lot_prices,
                       width=utils.OPEN_TRADE_WIDTH,
                       opacity=utils.OPEN_TRADE_ALPHA,
                       color=utils.OPEN_TRADE_COLOR,
                       name=utils.OPEN_TRADE_NAME,
                       _row=self.data_row,
                       _col=self.data_col)

    def plot_trade_triangles(self):
        loc = self.trader.df['Close']
        preds: Dict[str, List[Union[int, float]]] = {
            'sellind': [],
            'exitind': [],
            'buyind': [],
            'bprice': [],
            'sprice': [],
            'eprice': []
        }
        for e, (pred, conv, crlev) in enumerate(zip(self.trader.returns,
                                                    self.trader._converted,
                                                    utils.convert(self.trader._credit_leverages))):
            if e != 0:
                credlev_up = self.trader._credit_leverages[e - 1] < self.trader._credit_leverages[e]
                credlev_down = self.trader._credit_leverages[e - 1] > self.trader._credit_leverages[e]
                sell = (credlev_down and pred == utils.BUY) or (credlev_up and pred == utils.SELL)
                buy = (credlev_down and pred == utils.SELL) or (credlev_up and pred == utils.BUY)
            else:
                sell = buy = False

            if conv == utils.EXIT or crlev == 0:
                preds['exitind'].append(e)
                preds['eprice'].append(loc[e])
            elif conv == utils.SELL or sell:
                preds['sellind'].append(e)
                preds['sprice'].append(loc[e])
            elif conv == utils.BUY or buy:
                preds['buyind'].append(e)
                preds['bprice'].append(loc[e])
        name: str
        index: List[Union[int, float]]
        price: List[Union[int, float]]
        width: float
        alpha: float
        for name, index, price, triangle_type, color, width, alpha in zip(
                [utils.TRADE_MARKER_BUY_NAME,
                 utils.TRADE_MARKER_SELL_NAME,
                 utils.TRADE_MARKER_EXIT_NAME],

                [preds['buyind'], preds['sellind'], preds['exitind']],
                [preds['bprice'], preds['sprice'], preds['eprice']],

                [utils.TRADE_MARKER_BUY_TYPE,
                 utils.TRADE_MARKER_SELL_TYPE,
                 utils.TRADE_MARKER_EXIT_TYPE],

                [utils.TRADE_MARKER_BUY_COLOR,
                 utils.TRADE_MARKER_SELL_COLOR,
                 utils.TRADE_MARKER_EXIT_COLOR],

                [utils.TRADE_MARKER_BUY_WIDTH,
                 utils.TRADE_MARKER_SELL_WIDTH,
                 utils.TRADE_MARKER_EXIT_WIDTH],

                [utils.TRADE_MARKER_BUY_ALPHA,
                 utils.TRADE_MARKER_SELL_ALPHA,
                 utils.TRADE_MARKER_EXIT_ALPHA]
        ):
            self.plot_line(
                mode='markers',
                name=name,
                line=price,
                index=index,
                _row=self.data_row,
                _col=self.data_col,
                color=color,
                marker=triangle_type,
                width=width,
                opacity=alpha)

    def log_y(self,
              _row: int = 1,
              _col: int = 1):
        self._figure.update_yaxes(row=_row,
                                  col=_col,
                                  type='log')
