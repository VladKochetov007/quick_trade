# _pots
File for creating quick-trade charts.

## make_layout
Method for creating layouts for plotly.

| param  | type | description |
| :---: | :---: | :---: |
| height | Union\[int, float] | Plotly plot height |
| width | Union\[int, float] | Plotly plot width |
| template | str | template from https://plotly.com/python/templates/ |
| row_heights | list | The ratio of the heights of the symbol data, deposit and the deposit change. |
|returns|`plotly.graph_objs.Layout`| standard plotly layout for quick-trade |

## QuickTradeGraph