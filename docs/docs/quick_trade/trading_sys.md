# trading_sys:

The main file of quick_trade.

## Trader

A class in which you can trade and test strategies.

| param | type | description |
|:---:|:---:|:---:|
| ticker | str | symbol for CCXT (with "/" between base and quote) |
| df | `pd.DataFrame` |dataframe with columns: Close, Open, High, Low, Volume |
| interval | str | timeframe:1m 2m 3m 5m 15m 30m 45m 1h 90m 2h 3h 4h 12h 1d 3d 1w 1M 3M 6M |

```python
from quick_trade.trading_sys import Trader, ExampleStrategies
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

The method converts stop loss and take profit in points to stop loss and take profit in specific prices using
`self._stop_loss` and `self._take_profit` as values and `self._open_price` as the initial price

| param | type | description |
|:---:|:---:|:---:|
|sig|`utils.PREDICT_TYPE`|signal buy/sell/exit|
|returns|Dict|{'stop': S/L value, 'take': T/P value}|

### sl_tp_adder

A method for adding a number of points to the current S/L and T/P values.

| param  | type | description |
| :---: | :---: | :---: |
| add_stop_loss | float, int | points for addition to sl |
| add_take_profit | float, int | points for addition to tp |
| returns | Tuple\[stop_losses, take_profits] | sl, tp |

```python
trader.strategy_parabolic_SAR()
trader.sl_tp_adder(add_stop_loss=50)  # The stop loss moved 50 pips away from the opening price.
```

### multi_trades

This method is needed to process strategies with the ability to use several trades at once.

?> The method translates predictions that look like
[converted](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/utils?id=convert)
data into [unconverted](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/utils?id=anti_convert)
predictions and leverages.

```python
from quick_trade.utils import EXIT, BUY, SELL
import numpy as np


class MultiTrader(Trader):
    def multi_trade_strategy(self):
        ...
        for i in range(len(self.df)):
            ...
            if a:
                self.returns.append(BUY)
            elif b:
                self.returns.append(SELL)
            else:
                self.returns.append(np.nan)  # <------
        self.multi_trades()  # <----------------------
        self.set_open_stop_and_take()
        self.setcredit_leverages()
```

!> Using [`setcredit_leverages`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/trading_sys?id=setcredit_leverages)
after [`multi_trades`](#multi_trades) is not advisable!

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
| swap_stop_take | bool | Change stop-loss and take-profit values among themselves. If values for SL/TP have not been generated yet - False.|
| returns | `utils.PREDICT_TYPE_LIST` | Result of a strategy reversal |

!> SL / TP values are not recalculated; instead, the object fields are swapped.

```python
trader.some_user_strategy()
trader.inverse_strategy()
```

### backtest

A method with the functionality of testing a strategy on historical data. For it to work, you need to use a strategy that will assign values to `self.returns`,` self.stop_losses`, `self.take_profits`
and `self.credit_leverages`.

| param  | type | description |
| :---: | :---: | :---: |
| deposit | float, int | Initial deposit for testing the strategy that you used before the test |
| bet | float, int | The amount of money in one deal. If you want to enter the deal on the entire deposit, enter the value `np.inf` |
| commission | float, int | Commission for opening a deal in percentage. If you need to exit the previous one to enter a trade, the commission is deducted 2 times. |
| plot | bool | Plotting data about candles, deposits and trades on the chart. |
| print_out | bool | Displaying data on the number of profitable and unprofitable trades and annual income to the console. |
| show | bool | Show testing schedule. Includes candles, deposit, `.diff()` of deposit and other.|
| returns | `pd.DataFrame` | Dataframe with information about the deposit, strategy signals, `.diff()` of deposit, stop loss, take profit, opening prices, average growth of deposit, and dataframe series. |

?> The commission does not reduce the trade itself, but decreases the deposit, but if the deposit becomes less than the desired trade, deal is immediately reduced to the level of the deposit.

```python
from quick_trade.plots import *

fig = make_figure()
graph = QuickTradeGraph(figure=fig)
trader.connect_graph(graph)

# At this point, you need to use the strategy

trader.backtest(deposit=1000)
```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/plot.png?raw=true)

### multi_backtest

A method for testing a strategy on several symbols.

| param  | type | description |
| :---: | :---: | :---: |
| test_config | Dict[str, Dict[str, Dict[str, Any]]] | Data for testing strategies. |
| deposit | float | Initial deposit for testing the strategy that you used before the test |
| bet | float | The amount of money in one deal. If you want to enter the deal on the entire deposit, enter the value `np.inf` |
| commission | float | Commission for opening a deal in percentage. If you need to exit the previous one to enter a trade, the commission is deducted 2 times. |
| plot | bool | Plotting data about candles, deposits and trades on the chart. |
| print_out | bool | Displaying data on the number of profitable and unprofitable trades and annual income to the console. |
| show | bool | Show testing schedule. Includes candles, deposit, `.diff()` of deposit and other.|
| returns | `pd.DataFrame` | Dataframe with information about the deposit, average growth of deposit, `.diff()` of deposit and returns |

```python
import numpy as np

client = TradingClient(ccxt.binance())
trader = ExampleStrategies(ticker='ETH/BTC',
                           interval='5m')

fig = make_figure()
graph = QuickTradeGraph(figure=fig)
trader.connect_graph(graph)
trader.set_client(client)

strategy = {
    'strategy_supertrend':
        dict(
            multiplier=2,
            length=1,
            plot=False
        )
}

trader.multi_backtest(test_config={'BTC/USDT': [strategy],
                                   'LTC/USDT': [strategy],
                                   'ETH/USDT': [strategy]},
                      deposit=1700,
                      commission=0.075,
                      bet=np.inf,
                      limit=1000)
```

output:

```commandline
losses: 131
trades: 187
profits: 56
mean year percentage profit: -99.98292595327236%
winrate: 30.352119656186698%
mean deviation: 1.3564893692018003%
Sharpe ratio: -11.951556929509964
Sortino ratio: -18.561188937509858
calmar ratio: -8.002808500501935
max drawdown: 12.493479751140047%
```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/multi_backtest.png?raw=true)

### connect_graph

The method sets the plotly figure for graphs

| param  | type | description |
| :---: | :---: | :---: |
| graph | `plots.QuickTradeGraph` | QuickTradeGraph figure |

```python
from quick_trade.plots import QuickTradeGraph, make_figure

figure = make_figure()
graph = QuickTradeGraph(figure=figure)
```

```python
trader.connect_graph(graph)
```

or

```python
graph.connect_trader(trader=trader)
```

### strategy_collider

This method allows you to combine two strategies into one; it takes the lists of strategy predictions, and the combining mode as input values.

Available modes:

- `minimalist`: If both predictions are short, then the result of the collider will be short. If both are long, then long. If the predictions do not match - exit

- `maximalist`: If both predictions are short, then the result of the collider will be short. If both are long, then long. If the predictions do not match - result of the last merge.

- `super`: If an element of one of the predictions has changed, but the other not, the result is equal to the one that has changed. If they changed at the same time - exit.

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

[`strategy_collider`](#strategy_collider) for multiple strategies

?> First, the first two strategies are combined, and then the result of the previous join is combined with subsequent strategies.

```python
trader.strategy_collider(trader.strategy_2_sma(50, 20),
                         trader.strategy_2_sma(20, 10),
                         trader.strategy_parabolic_SAR(),
                         trader.strategy_macd(),
                         mode='maximalist')
```

### get_trading_predict

This method is needed to get the result of the strategy and open an order on the exchange.

| param  | type | description |
| :---: | :---: | :---: |
| bet_for_trading_on_client | float, int | The amount of money in the possible trade. Indicated in the quoted currency. |
| returns | Dict[str, Union[str, float]] | Open trade data. |

### realtime_trading
A method for running one strategy at real-time on the exchange. You will not be able to monitor the terminal by monitoring trades.

| param  | type | description |
| :---: | :---: | :---: |
| entry_start_trade | bool | Entering a trade at the first new candlestick. If False - enter when a new signal appears. |
| strategy | method(function) | The method of an instance of the trader class, it will be called to getting a trading signal. |
| start_time | `datetime.datetime` |  |
| ticker | str |  |
| print_out | bool |  |
| bet_for_trading_on_client | float, int |  |
| wait_sl_tp_checking | float, int |  |
| limit | int |  |
| strategy_in_sleep | bool |  |
| strategy_args | arguments |  |
| strategy_kwargs | named arguments |  |

```python

```

### multi_realtime_trading

### log_data

### log_deposit

### log_returns

### set_client

### convert_signal

### set_open_stop_and_take

### setcredit_leverages

### get_support_resistance

### strategy_diff

The strategy issues its verdict based on the last change to the dataframe. If you give the entry the closing price of candles, then if the candle is green - long, if red - short

| param  | type | description |
| :---: | :---: | :---: |
| frame_to_diff | pd.Series | series of dataframe |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

## ExampleStrategies

Class with examples of strategies, inherited from the [`Trader`](#Trader) class

### \_window_

### strategy_rsi

When the RSI is greater than the `maximum`, and the current value is less than the previous one, short. It's the same with long (but with `minimum`).

If the value has crossed the border with mid` - exit

| param  | type | description |
| :---: | :---: | :---: |
| minimum | Union\[float, int] | min level of RSI |
| maximum | Union\[float, int] | max level of RSI |
| max_mid | Union\[float, int] | max/mid level of RSI (exit from short) |
| min_mid | Union\[float, int] | min/mid level of RSI (exit from long) |
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
| multiplier | float, int | multiplier parameterf for [`utils.SuperTrendIndicator`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/utils?id=supertrendindicator) |
| length | int | length parameter for [`utils.SuperTrendIndicator`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/utils?id=supertrendindicator) |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/supertrend.png?raw=true)

### strategy_bollinger

Bollinger bands strategy (not breakout)

| param  | type | description |
| :---: | :---: | :---: |
| plot | bool | plotting of bollinger bands |
| to_mid | bool | stop loss on mid line of `ta.volatility.BollingerBands` |
| **bollinger_args | arguments | arguments for `ta.volatility.BollingerBands` |
| **bollinger_kwargs | named arguments | named arguments for `ta.volatility.BollingerBands` |
| returns | `utils.PREDICT_TYPE_LIST` | returns |

...
