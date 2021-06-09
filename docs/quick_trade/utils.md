# utils:

## set_

This function performs an operation on the data

| param  | type | description |
| :---: | :---: | :---: |
| data    | List | Data for processing |
| returns   | utils.SETED_TYPE_LIST | data in which repeating elements are replaced np.nan |

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

## anti_set_

Reverse from set_

| param  | type | description |
| :---: | :---: | :---: |
| seted | List | returns of set_ |
| _nan_num | The value to replace the np.nan| Without this, the replacement may be incorrect. |
| returns   | list | list without np.nan |

## get_window

A function for getting a list of lists, each of which has n elements. These lists will be "data windows" with an offset
of 1.

| param  | type | description |
| :---: | :---: | :---: |
| values | List | data for getting "windows" |
| window_length | int | length of "windows" |
| returns | list of lists | list of "windows" |

## convert_signal_str

Convert signals in signal format to string.

| param  | type | description |
| :---: | :---: | :---: |
| predict | utils.PREDICT_TYPE | predict |
| returns | str | string predict |

## get_binance_data

Function for getting the last Binance candles

| param  | type | description |
| :---: | :---: | :---: |
| ticker | str | symbol |
| interval | str | interval |
| date_index | bool | dataframe index as date |
|limit|int|candles limit|
| returns | pd.DataFrame | dataframe with data |

## ta_lib_to_returns

replaces ta-lib values with quick-trade predictions

| param  | type | description |
| :---: | :---: | :---: |
| talib_returns | pd.Series | ta-lib returns series |
| exit_ | str | value for talib 0 value (exit predict) |
| returns | list | ta-lib pattern strategy predicts |

## ta_lib_collider_all

ta_lib_to_returns, but exit_=np.nan

## SuperTrendIndicator

supertrend indicator class

| param  | type | description |
| :---: | :---: | :---: |
| close | pd.Series | close data|
| high | pd.Series | high data|
| low |pd.Series| low daa|
| multiplier |float| ATR multiplier |
| length |int| ATR length (period) |

### get_supertrend

get pd.Series with supertrend indicator's data

### get_supertrend_upper

get pd.Series with supertrend upper indicator's data

### get_supertrend_lower

get pd.Series with supertrend lower indicator's data

### get_supertrend_strategy_returns

get pd.Series with supertrend predictions

### get_all_ST

get all supertrend data:

    - Supertrend
    - strategy predictions
    - lower
    - upper

## get_linear

pseudo-linear numpy array

| param  | type | description |
| :---: | :---: | :---: |
| dataset | Iterable | data for linear transformation |
| returns | np.ndarray | linear data |
