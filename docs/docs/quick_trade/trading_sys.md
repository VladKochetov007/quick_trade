# trading_sys:

The main file of quick_trade.

## Trader

A class in which you can trade and test strategies.

| param | type | description |
|:---:|:---:|:---:|
|ticker|str|symbol for CCXT (with "/")|
|df|`pd.DataFrame`|dataframe with columns: Close, Open, High, Low, Volume|
|interval|str| timeframe:1m 2m 3m 5m 15m 30m 45m 1h 90m 2h 3h 4h 12h 1d 3d 1w 1M 3M 6M|

```python
from quick_trade.trading_sys import Trader
from quick_trade.brokers import TradingClient
import ccxt


client = TradingClient(ccxt.binance())
trader = Trader('BTC/USDT', 
                client.get_data_historical('BTC/USDT', '2h'),
                '2h')

```

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

The method converts stop loss and take profit in points to stop loss and take profit in specific prices using `self._
stop_loss` and `self._take_profit` as values and `self._open_price` as the initial price

| param | type | description |
|:---:|:---:|:---:|
|sig|`utils.PREDICT_TYPE`|signal buy/sell/exit|
|returns|Dict|{'stop': S/L value, 'take': T/P value}|

### sl_tp_adder

A method for adding a number of points to the current S/L and T/P values.

| param  | type | description |
| :---: | :---: | :---: |
| add_stop_loss | float | points for addition to sl |
| add_take_profit | float | points for addition to tp |
| returns | Tuple\[stop_losses, take_profits] | sl, tp |

```python
trader.strategy_parabolic_SAR()
trader.sl_tp_adder(add_stop_loss=50)  # The stop loss moved 50 pips away from the opening price.
```

### strategy_diff

The strategy issues its verdict based on the last change to the dataframe. If you give the entry the closing price of
candles, then if the candle is green - long, if red - short

| param  | type | description |
| :---: | :---: | :---: |
| frame_to_diff | pd.Series | series of dataframe |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

### strategy_rsi

When the RSI is greater than the `maximum`, and the current value is less than the previous one, short. It's the same
with long (but with `minimum`).

If the value has crossed the border with mid` - exit

| param  | type | description |
| :---: | :---: | :---: |
| minimum | float | min level of RSI |
| maximum | float | max level of RSI |
| max_mid | float | max/mid level of RSI (exit from short) |
| min_mid | float | min/mid level of RSI (exit from long) |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

example:
![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/rsi_strat_example.jpg?raw=true)
![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/rsi_2.jpg?raw=true)

### strategy_parabolic_SAR

parabolic SAR strategy

| param  | type | description |
| :---: | :---: | :---: |
| plot | bool | plotting of SAR indicator |
| **sar_kwargs | named arguments | named arguments for `ta.trend.PSARIndicator` |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

### strategy_macd_histogram_diff

`strategy_diff` with MACD's histogram data.

| param  | type | description |
| :---: | :---: | :---: |
| slow | int | slow MA of MACD |
| fast | int | fast MA of MACD |
| **macd_kwargs | named arguments | named arguments for `ta.trend.MACD` |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

example:

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/macd_diff_example.jpg?raw=true)

### strategy_supertrend

supertrend strategy. S/L - ST indicator

| param  | type | description |
| :---: | :---: | :---: |
| plot | bool | plotting of SAR indicator |
| **st_args | arguments | arguments for `utils.SuperTrendIndicator` |
| **st_kwargs | named arguments | named arguments for `utils.SuperTrendIndicator` |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/supertrend.png?raw=true)

### strategy_bollinger

Bollinger bands strategy (not breakout)

| param  | type | description |
| :---: | :---: | :---: |
| plot | bool | plotting of bollinger bands |
| to_mid | bool | exit on mid line of `ta.volatility.BollingerBands` |
| **bollinger_args | arguments | arguments for `ta.volatility.BollingerBands` |
| **bollinger_kwargs | named arguments | named arguments for `ta.volatility.BollingerBands` |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

### get_heikin_ashi

Heikin Ashi candles

| param  | type | description |
| :---: | :---: | :---: |
| df | `pd.DataFrame` | dataframe with Open, High, Low, Close data |
| returns | `pd.DataFrame` | HA ohlc |

### crossover

| param  | type | description |
| :---: | :---: | :---: |
| fast | Iterable | When the element of this parameter is less than the element of the `slow` parameter - short. When more - long. |
| slow | Iterable | When the element of this parameter is greater than the element of the `fast` parameter - short. When less, it takes a long time. |
| returns | `utils.PREDICT_TYPE_LIST` | crossover  strategy |

```python
from ta.trend import sma_indicator

slow = sma_indicator(trader.df['Close'], 50)
fast = sma_indicator(trader.df['Close'], 20)
trader.crossover(fast=fast, slow=slow)  # crossover strategy
```

### inverse_strategy

| param  | type | description |
| :---: | :---: | :---: |
| fast | Iterable | When the element of this parameter is less than the element of the `slow` parameter - short. When more - long. |
| slow | Iterable | When the element of this parameter is greater than the element of the `fast` parameter - short. When less, it takes a long time. |
| returns | `utils.PREDICT_TYPE_LIST` | crossover  strategy |

### backtest

A method with the functionality of testing a strategy on historical data. For it to work, you need to use a strategy
that will assign values to `self.returns`,` self._stop_losses`, `self._take_profits` and `self._credit_leverages`.

| param  | type | description |
| :---: | :---: | :---: |
| deposit | float | Initial deposit for testing the strategy that you used before the test |
| bet | float | The amount of money in one deal. If you want to enter the deal on the entire deposit, enter the value `np.inf` |
| commission | float | Commission for opening a deal in percentage. If you need to exit the previous one to enter a trade, the commission is deducted 2 times. |
| plot | bool | Plotting data about candles, deposits and trades on the chart. |
| print_out | bool | Displaying data on the number of profitable and unprofitable trades and annual income to the console. |
| column | str | The parameter shows which series of the dataframe should be used to test the strategy. |
| show | bool | Show testing schedule. Includes candles, deposit, `.diff()` of deposit and other.|
| returns | `pd.DataFrame` | Dataframe with information about the deposit, strategy signals, `.diff()` of deposit, stop loss, take profit, opening prices, pseudo-line deposit, and dataframe series. |

?> The commission does not reduce the trade itself, but decreases the deposit, but if the deposit becomes less than the
desired trade, deal is immediately reduced to the level of the deposit.

```python
trader.set_pyplot()
trader.backtest(deposit=1000)
```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/plot.png?raw=true)

### multi_backtest

A method for testing a strategy on several symbols.

| param  | type | description |
| :---: | :---: | :---: |
| tickers | Sized and Iterable \[str] | Strategy testing symbols. |
| strategy_name | str | strategy name for `Trader._get_attr` |
| strategy_kwargs | Dict\[str, Any] | named arguments for `Trader._get_attr(strategy_name)` |
| deposit | float | Initial deposit for testing the strategy that you used before the test |
| bet | float | The amount of money in one deal. If you want to enter the deal on the entire deposit, enter the value `np.inf` |
| commission | float | Commission for opening a deal in percentage. If you need to exit the previous one to enter a trade, the commission is deducted 2 times. |
| plot | bool | Plotting data about candles, deposits and trades on the chart. |
| print_out | bool | Displaying data on the number of profitable and unprofitable trades and annual income to the console. |
| column | str | The parameter shows which series of the dataframe should be used to test the strategy. |
| show | bool | Show testing schedule. Includes candles, deposit, `.diff()` of deposit and other.|
| returns | `pd.DataFrame` | Dataframe with information about the deposit, pseudo-line deposit and `.diff()` of deposit. |

!> Each pair is tested separately and then the results are summarized. Because of this, the strategies do not use the
total deposit in such a test.

```python
import numpy as np


trader.set_client(client)
trader.multi_backtest(['BTC/USDT',
                       'ETH/USDT',
                       'LTC/USDT'], 
                      strategy_name='strategy_supertrend', 
                      strategy_kwargs=dict(multiplier=2, length=1), 
                      deposit=1700, 
                      commission=0.075, 
                      bet=np.inf, 
                      limit=1000)
```

output:
```
losses: 90
trades: 88
profits: 88
mean year percentage profit: 436.3006580276444%
winrate: 49.9921984709003%
```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/multi_backtest.png?raw=true)

### set_pyplot

The method sets the plotly figure for graphs

| param  | type | description |
| :---: | :---: | :---: |
| height | int | Plotly plot height |
| width | int | Plotly plot width |
| template | str | template from https://plotly.com/python/templates/ |
| row_heights | list | The ratio of the heights of the symbol data, deposit and the deposit change. |
| subplot_kwargs | named arguments | named arguments for [`plotly.subplots.make_subplots`](https://github.com/plotly/plotly.py/blob/master/packages/python/plotly/plotly/subplots.py#L45) |

### strategy_collider

This method allows you to combine two strategies into one; it takes the lists of strategy predictions, and the combining
mode as input values.

- Available modes:
    - `minimalist`: If both predictions are short, then the result of the collider will be short. If both are long, then
      long. If the predictions do not match - exit

    - `maximalist`: If both predictions are short, then the result of the collider will be short. If both are long, then
      long. If the predictions do not match - result of the last merge.

    - `super`: If an element of one of the predictions has changed, but the other not, the result is equal to the one
      that has changed. If they changed at the same time - exit.

| param  | type | description |
| :---: | :---: | :---: |
| first_returns | `utils.PREDICT_TYPE_LIST` |  result of using the strategy |
| second_returns | `utils.PREDICT_TYPE_LIST` |  result of using the strategy |
| mode | str | Colliding strategy mode |
| returns | `utils.PREDICT_TYPE_LIST` | result of combining strategies |

```python
trader.strategy_collider(trader.strategy_2_sma(50, 20),
                         trader.strategy_2_sma(20, 10),
                         'minimalist')  # crossover of 3 sma
```

### multi_strategy_collider
`strategy_collider` for multiple strategies

?> First, the first two strategies are combined, and then the result of the previous join is combined with subsequent strategies.

```python
trader.strategy_collider(trader.strategy_2_sma(50, 20),
                         trader.strategy_2_sma(20, 10),
                         trader.strategy_parabolic_SAR(),
                         trader.strategy_macd()
                         mode='maximalist')
```

### get_trading_predict
This method is needed to get the result of the strategy and open an order on the exchange.
