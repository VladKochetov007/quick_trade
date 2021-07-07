from plotly.graph_objs import Figure, Scatter
from plotly.subplots import make_subplots
from typing import Union, List
import utils


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
        title_text=utils.MONEYS_TITLE, row=2, col=1, color=utils.TEXT_COLOR)
    fig.update_yaxes(
        title_text=utils.RETURNS_TITLE, row=3, col=1, color=utils.TEXT_COLOR)
    fig.update_yaxes(
        title_text=utils.DATA_TITLE, row=1, col=1, color=utils.TEXT_COLOR)
    return fig


class QuickTradeGraph(object):
    figure: Figure

    def __init__(self, figure: Figure):
        self.figure = figure

    def show(self, **kwargs):
        return self.figure.show(**kwargs)

    def plot_line(self,
                  line=None,
                  width: float = 1.0,
                  opacity: float = 1.0,
                  color: str = None,
                  name: str = None,
                  _row: int = 1,
                  _col: int = 1):
        return self.figure.add_trace(
            row=_row,
            col=_col,
            trace=Scatter(
                y=line,
                text=name,
                mode='lines',
                opacity=opacity,
                line=dict(
                    color=color,
                    width=width
                )
            )
        )


if __name__ == "__main__":
    g = QuickTradeGraph(figure=make_figure())
    g.plot_line([1,3,2,4,2,4,3], color='#fff', width=10, name='trrbgbb bgrbg 4 b')
    g.figure.show()
