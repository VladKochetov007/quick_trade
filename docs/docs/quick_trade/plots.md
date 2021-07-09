# plots

File for creating quick-trade charts.

## make_figure

Method for creating standard figure for plotly.

| param  | type | description |
| :---: | :---: | :---: |
| height | Union\[int, float] | Plotly plot height |
| width | Union\[int, float] | Plotly plot width |
| template | str | template from https://plotly.com/python/templates/ |
| row_heights | list | The ratio of the heights of the symbol data, deposit and the deposit change. |
|returns|`plotly.graph_objs.Figure`| standard plotly figure for quick-trade |

!> The number of columns must be 1, and the number of rows must be 3.

## QuickTradeGraph

### connect_trader

### show

### plot_line

### plot_candlestick

### plot_deposit

### plot_returns

### plot_SL_TP_OPN

### plot_trade_triangles

### log_y

### plot_area
