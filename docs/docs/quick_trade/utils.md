# utils:

## convert

This function performs an operation on the data

| param  | type | description |
| :---: | :---: | :---: |
| data    | `utils.PREDICT_TYPE_LIST` | Data for processing |
| returns   | `utils.CONVERTED_TYPE_LIST` | data in which repeating elements are replaced `np.nan` |

if the element is equal to the previous one, then it becomes np.nan.

```
[0,    [0
0,      nan
0,      nan
0,      nan
0,      nan
0,      nan
0,      nan
0,      nan
1,      1
1,      nan
1,      nan
1,      nan
1,      nan
1,      nan
-1,     -1
-1,     nan
1,      1
-1,     -1
1,      1
-1,     -1
-1,     nan
-1,     nan
0,      0
0,  ->  nan
0,      nan
0,      nan
0,      nan
0,      nan
-1,     -1
-1,     nan
-1,     nan
-1,     nan
-1,     nan
-1,     nan
1,      1
1,      nan
1,      nan
1,      nan
0,      0
0,      nan
0,      nan
0,      nan
0,      nan
0,      nan
-1,     -1
-1,     nan
-1]     nan]
```

## anti_convert

Reverse from [`convert`](#convert)

| param  | type | description |
| :---: | :---: | :---: |
| converted | `utils.CONVERTED_TYPE_LIST` | returns of [`converted`](#convert) |
| _nan_num | The value to replace the `np.nan`| Without this, the replacement may be incorrect. |
| returns   | `utils.PREDICT_TYPE_LIST` | list without `np.nan` |

```commandline
In[8]: anti_convert([1, np.nan, np.nan, -1, np.nan, np.nan, np.nan, 1, 0, np.nan])
Out[8]: [1, 1, 1, -1, -1, -1, -1, 1, 0, 0]
```

## get_window

A function for getting a list of lists, each of which has n elements. These lists will be "data windows" with an offset of 1.

| param  | type | description |
| :---: | :---: | :---: |
| values | Union\[Sequence, Sized] | data for getting "windows" |
| window_length | int | length of "windows" |
| returns | List\[Iterable\[Any]] | list of "windows" |

```commandline
In[10]: get_window([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], window_length=3)
Out[10]: 
[[1, 2, 3],
 [2, 3, 4],
 [3, 4, 5],
 [4, 5, 6],
 [5, 6, 7],
 [6, 7, 8],
 [7, 8, 9],
 [8, 9, 10]]
```

## convert_signal_str

Convert signals in signal format to string.

| param  | type | description |
| :---: | :---: | :---: |
| predict | `utils.PREDICT_TYPE` | predict |
| returns | str | string predict |

```commandline
In[11]: convert_signal_str(BUY)
Out[11]: 'Buy'

In[12]: convert_signal_str(SELL)
Out[12]: 'Sell'

In[13]: convert_signal_str(EXIT)
Out[13]: 'Exit'
```

## get_exponential_growth

Function for calculating [compound interest](https://investmentu.com/simple-interest-vs-compound-interest/)

| param  | type | description |
| :---: | :---: | :---: |
| dataset | Sequence\[float] | data for transformation |
| returns | `np.ndarray` | exp growth data |

## get_coef_sec

Function for converting timeframe to profit ratio and sleep time for [`realtime_trading`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/trading_sys?id=realtime_trading)

| param  | type | description |
| :---: | :---: | :---: |
| dataset | str | timeframe |
| returns | Tuple\[float, int] | profit coef and seconds to wait |

```commandline
In[14]: get_coef_sec('1d')
Out[14]: (365, 86400)

In[15]: get_coef_sec('1m')
Out[15]: (525600, 60)

In[16]: get_coef_sec('5m')
Out[16]: (105120, 300)
```

## wait_success

Decorator. If a traceback was received during the execution of the function, then the action is repeated after `utils.WAIT_SUCCESS_SLEEP` seconds.

The main purpose is to avoid ConnectionError when trading in real time.
[see this page](https://stackoverflow.com/questions/27333671/how-to-solve-the-10054-error)

## profit_factor

Function for calculating the coefficient of exponential growth of a deposit when testing a strategy.

| param  | type | description |
| :---: | :---: | :---: |
| deposit_list | Sequence[float] | exponential growth history |
| returns | float | Growth rate with every step |

## assert_logger

Decorator. If AssertionError was called, then first the error is written to the log with the "critical" level, and then the program stops.

## get_diff

Function for getting the price movement from the current one to TP or SL.

| param  | type | description |
| :---: | :---: | :---: |
| price | float | The closing price at which the breakout was detected at this or the next moment. |
| low | float | The lower price of the candlestick that turned out to be outside the limits of the acceptable value for SL / TP. |
| high | float | The higher price of the candlestick that turned out to be outside the limits of the acceptable value for SL / TP. |
| stop_loss | float | Directly permissible value of SL. |
| take_profit | float | Directly permissible value of TP. |
| signal | `utils.PREDICT_TYPE` | trading prediction at current candle. |
| returns | float | difference of SL/TP price and current price. |

## make_multi_trade_returns

## get_multipliers

## mean_deviation

## year_profit

## map_dict

## recursive_dict
