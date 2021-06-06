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
