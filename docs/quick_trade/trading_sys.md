# trading_sys:
The main file of quick_trade.

## Trader
A class in which you can trade and test strategies.

| param | type | description |
|:---:|:---:|:---:|
|ticker|str|symbol for CCXT (with "/")|
|df|pd.DataFrame|dataframe with columns: Close, Open, High, Low, Volume|
|interval|str| timeframe:1m 2m 3m 5m 15m 30m 45m 1h 90m 2h 3h 4h 12h 1d 3d 1w 1M 3M 6M|


### _get_attr
getattr from self

| param | type | description |
|:---:|:---:|:---:|
|attr|str|name of attribute|
|returns|attribute|attribute|

### _get_this_instance

| param | type | description |
|:---:|:---:|:---:|
|*args|arguments|argument for trading class \_\_init__|
|*kwargs|named arguments|named argument for trading class \_\_init__|
|returns|Trader (or child class)|new trading object|

### __get_stop_take
The method converts stop loss and take profit
in points to stop loss and take profit in 
specific prices using self._stop_loss and 
self._take_profit as values and self._open_price 
as the initial price

| param | type | description |
|:---:|:---:|:---:|
|sig|PREDICT_TYPE|signal buy/sell/exit|
|returns|Dict|{'stop': S/L value, 'take': T/P value}|


### sl_tp_adder
A method for adding a number of points to
the current S/L and T/P values.

| param  | type | description |
| :---: | :---: | :---: |
| add_stop_loss | float | points for addition to sl |
| add_take_profit | float | points for addition to tp |
| returns | Tuple\[stop_losses, take_profits] | sl, tp |

### strategy_diff
The strategy issues its verdict based on the last change to the dataframe.
If you give the entry the closing price of candles, then if the candle is green - LONG, if red - SHORT

| param  | type | description |
| :---: | :---: | :---: |
| frame_to_diff | pd.Series | series of dataframe |
| returns | PREDICT_TYPE_LIST | returns |

### strategy_rsi
When the RSI is greater than the ```maximum```, and the current value is less than the previous one, short. It's the same with long (but with ```minimum```).

If the value has crossed the border with ```mid``` - exit

| param  | type | description |
| :---: | :---: | :---: |
| minimum | float | min level of RSI |
| maximum | float | max level of RSI |
| max_mid | float | max/mid level of RSI (exit from short) |
| min_mid | float | min/mid level of RSI (exit from long) |
| returns | PREDICT_TYPE_LIST | returns |

example:
![image](https://github.com/VladKochetov007/quick_trade/blob/master/docs/rsi_strat_example.jpg?raw=true)

### strategy_parabolic_SAR
parabolic SAR strategy

| param  | type | description |
| :---: | :---: | :---: |
| plot | bool | plotting of SAR indicator |
| **sar_kwargs | named arguments | named arguments for ```ta.trend.PSARIndicator``` |
| returns | PREDICT_TYPE_LIST | returns |

### strategy_macd_histogram_diff
```strategy_diff``` with MACD's histogram data
example:

![image](https://github.com/VladKochetov007/quick_trade/blob/master/docs/macd_diff_example.jpg?raw=true)

| param  | type | description |
| :---: | :---: | :---: |
| slow | int | slow MA of MACD |
| fast | int | fast MA of MACD |
| **macd_kwargs | named arguments | named arguments for ```ta.trend.MACD``` |
| returns | PREDICT_TYPE_LIST | returns |
