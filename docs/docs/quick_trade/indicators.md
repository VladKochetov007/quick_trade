# indicators

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
