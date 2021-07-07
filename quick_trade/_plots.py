from plotly.graph_objs import Figure, Layout
from typing import Union, List
import utils


@utils.assert_logger
def make_layout(height: Union[int, float] = 900,
                width: Union[int, float] = 1300,
                template: str = 'plotly_dark',
                row_heights: List[Union[int, float]] = [10, 16, 7]) -> Layout:
    assert isinstance(height, (int, float)), 'height must be of type <int> or <float>'
    assert isinstance(width, (int, float)), 'width must be of type <int> or <float>'
    assert isinstance(template, str), 'template must be of type <str>'
    assert isinstance(row_heights, list), 'row_heights must be of type <List[int, float]>'
    for el in row_heights:
        assert isinstance(el, (int, float)), 'row_heights must be of type <List[int, float]>'

    layout = Layout()
    layout.grid.rows = 3
    layout.grid.columns = 1
    layout.update(
        dict(
            height=height,
            width=width,
            template=template,
            xaxis_rangeslider_visible=False
        )
    )
    layout.xaxis.update(
        dict(
            title_text=utils.TIME_TITLE,
            row=3,
            col=1,
            color=utils.TEXT_COLOR
        )
    )  # fuck
    layout.yaxis.update(
        dict(
            title_text=utils.MONEYS_TITLE,
            row=2,
            col=1,
            color=utils.TEXT_COLOR
        )
    )
    layout.yaxis.update(
        dict(
            title_text=utils.RETURNS_TITLE,
            row=3,
            col=1,
            color=utils.TEXT_COLOR
        )
    )
    layout.yaxis.update(
        dict(
            title_text=utils.DATA_TITLE,
            row=1,
            col=1,
            color=utils.TEXT_COLOR
        )
    )

    return layout


class QuickTradeGraph(object):
    figure: Figure

    def __init__(self, layout: Layout):
        self.figure = Figure(layout=layout)

if __name__ == "__main__":
    g = QuickTradeGraph(layout=make_layout())
    g.figure.show()

