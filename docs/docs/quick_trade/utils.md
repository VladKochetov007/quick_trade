# utils:

## convert

This function performs an operation on the data

| param  | type | description |
| :---: | :---: | :---: |
| data    | List | Data for processing |
| returns   | `utils.CONVERTED_TYPE_LIST` | data in which repeating elements are replaced `np.nan` |

if the element is equal to the previous one, then it becomes np.nan.

```
[2,    [2
2,      nan
2,      nan
2,      nan
2,      nan
2,      nan
2,      nan
2,      nan
1,      1
1,      nan
1,      nan
1,      nan
1,      nan
1,      nan
0,      0
0,      nan
1,      1
0,      0
1,      1
0,      0
0,      nan
0,      nan
2,      2
2,  ->  nan
2,      nan
2,      nan
2,      nan
2,      nan
0,      0
0,      nan
0,      nan
0,      nan
0,      nan
0,      nan
1,      1
1,      nan
1,      nan
1,      nan
2,      2
2,      nan
2,      nan
2,      nan
2,      nan
2,      nan
0,      0
0,      nan
0]      nan]
```

## anti_convert

Reverse from [`convert`](#convert)

| param  | type | description |
| :---: | :---: | :---: |
| converted | List | returns of [`converted`](#convert) |
| _nan_num | The value to replace the `np.nan`| Without this, the replacement may be incorrect. |
| returns   | list | list without `np.nan` |

```commandline
In[8]: anti_convert([1, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 1, 2, np.nan])
Out[8]: [1, 1, 1, 0, 0, 0, 0, 1, 2, 2]
```

## get_window

A function for getting a list of lists, each of which has n elements. These lists will be "data windows" with an offset
of 1.

| param  | type | description |
| :---: | :---: | :---: |
| values | Iterable | data for getting "windows" |
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

## ta_lib_to_returns

replaces ta-lib values with quick-trade predictions

| param  | type | description |
| :---: | :---: | :---: |
| talib_returns | `pd.Series` | ta-lib returns series |
| exit_ | str | value for talib 0 value (exit predict) |
| returns | list | ta-lib pattern strategy predicts |

## ta_lib_collider_all

[`ta_lib_to_returns`](#ta_lib_to_returns), but `exit_=np.nan`

## SuperTrendIndicator

supertrend indicator class

| param  | type | description |
| :---: | :---: | :---: |
| close | `pd.Series` | close data|
| high | `pd.Series` | high data|
| low |`pd.Series`| low daa|
| multiplier |float| ATR multiplier |
| length |int| ATR length (period) |

### get_supertrend

get `pd.Series` with supertrend indicator's data

### get_supertrend_upper

get `pd.Series` with supertrend upper indicator's data

### get_supertrend_lower

get `pd.Series` with supertrend lower indicator's data

### get_supertrend_strategy_returns

get `pd.Series` with supertrend predictions

### get_all_ST

get all supertrend data as `pd.DataFrame`:

    - Supertrend
    - strategy predictions
    - lower
    - upper

## get_linear

pseudo-linear numpy array

| param  | type | description |
| :---: | :---: | :---: |
| dataset | Iterable | data for linear transformation |
| returns | `np.ndarray` | linear data |

## get_coef_sec

Function for converting timeframe to profit ratio and sleep time
for [`realtime_trading`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/trading_sys?id=realtime_trading)

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

Decorator. If a traceback was received during the execution of the function, then the action is repeated
after `utils.WAIT_SUCCESS_SLEEP` seconds.

The main purpose is to avoid ConnectionError when trading in real time.
[see this page](https://stackoverflow.com/questions/27333671/how-to-solve-the-10054-error)
