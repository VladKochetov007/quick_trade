import datetime as dt
import itertools
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from iexfinance.stocks import get_historical_intraday

PREDICT_TYPE = int

R: str = '#ff0000'
G: str = '#55ff00'
B: str = '#0015ff'
C: str = 'cyan'
INPUTS: int = 100
BUY: PREDICT_TYPE = 1
SELL: PREDICT_TYPE = 0
EXIT: PREDICT_TYPE = 2

IEX_TOKEN: str = 'Tpk_a4bc3e95d4c94810a3b2d4138dc81c5d'

os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = IEX_TOKEN
__author__: str = 'Vlad Kochetov'
__credits__: list[str] = ["Hemerson Tacon -- Stack overflow",
                          "hpaulj -- Stack overflow",
                          "Войтенко Николай Поликарпович (Vojtenko Nikolaj Polikarpovich) -- helped me test the "
                          "system of interaction with the binance crypto exchange with 50 dollars."]
__version__: str = "3.0"

TICKER: str = '^DJI'
SCATTER_SIZE: float = 12.0
SCATTER_ALPHA: float = 1.0
TAKE_STOP_OPN_WIDTH: float = 1.0
TEXT_COLOR: str = 'white'
SUB_LINES_WIDTH: float = 3.0
STOP_TAKE_OPN_ALPHA: float = 1.0
COLOR_DEPOSIT: str = 'white'
SCATTER_DEPO_ALPHA: float = SCATTER_ALPHA
DEPO_COLOR_UP: str = 'green'
DEPO_COLOR_DOWN: str = 'red'
FILE_LOG_NAME: str = 'trading.log'
logger = logging.getLogger()
logger.setLevel(50)
logging.basicConfig(level=20, filename=FILE_LOG_NAME)


def expansion_with_shear(values, ins=EXIT) -> list[PREDICT_TYPE]:
    ret: list[PREDICT_TYPE] = []
    for value in values:
        for column in range(4):
            ret.append(value)
    for i in range(3):
        ret.insert(0, ins)
    return ret[:-3]


def set_(data: Any) -> list[Any]:
    ret: list[Any] = list(data.copy())
    e: int
    for e, i in enumerate(data[1:]):
        if i == data[e]:
            ret[e + 1] = np.nan
    return ret


def to_4_col_df(data: Any, *columns) -> pd.DataFrame:
    """
    data:  |  array-like  |  data to converting

    returns:
    pd.DataFrame with 4 your columns

    """
    predict: list[list[float]] = [[] for _ in range(4)]
    for it in range(4):
        for i in range(it, len(data), 4):
            predict[it].append(data[i])
    return pd.DataFrame(predict, index=columns).T


def inverse_4_col_df(df: pd.DataFrame, columns: list[Any]) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(df)
    ret: np.ndarray = df.values
    rets: list[pd.DataFrame] = []
    for column in columns:
        rets.append(
            pd.DataFrame(itertools.chain.from_iterable(ret), columns=[column]))
    return pd.concat(rets, axis=1)


def get_data(ticker: str, undo_days: int) -> pd.DataFrame:
    tuples: list[tuple[int]] = []
    returns: pd.DataFrame
    end: dt.datetime = dt.datetime.today()
    date_list: list[dt.datetime] = [end - dt.timedelta(days=d) for d in range(undo_days)]
    for i in range(len(date_list)):
        ret = str(np.array(date_list[::-1])[i])[:10]
        ret = tuple(map(int, ret.split('-')))
        tuples.append(ret)

    dataframes: list[pd.DataFrame] = []
    df_: pd.DataFrame
    dat: dt.datetime
    for date_ in tuples:
        dat = dt.datetime(*date_)
        df_ = get_historical_intraday(ticker, date=dat, output_format='pandas')
        dataframes.append(df_)
    returns = pd.concat(dataframes, axis=0)
    returns = pd.DataFrame({
        'High': returns['high'].values,
        'Low': returns['low'].values,
        'Open': returns['open'].values,
        'Close': returns['close'].values,
        'Volume': returns['volume'].values
    })
    return returns.dropna()


def anti_set_(seted: list[Any]) -> list[Any]:
    ret: list[Any] = [seted[0]]
    flag = seted[0]
    e: int
    for e, i in enumerate(seted[1:]):
        if i is np.nan:
            ret.append(flag)
        else:
            ret.append(i)
            flag = i
    return ret


def digit(data) -> list[PREDICT_TYPE]:
    ret: list[PREDICT_TYPE] = []
    for element in list(data):
        if element == 0:
            ret.append(EXIT)
        elif element > 0:
            ret.append(BUY)
        else:
            ret.append(SELL)
    return ret


def get_window(values, window_length: int) -> list[Any]:
    ret: list[Any] = []
    for e, i in enumerate(values[:len(values) - window_length + 1]):
        ret.append(values[e:e + window_length])
    return ret


def nothing(ret: Any) -> Any:
    return ret


def get_binance_data(ticker: str = "BNBBTC", interval: str = "1m", date_index: bool = False):
    url: str = f"https://api.binance.com/api/v1/klines?symbol={ticker}&interval={interval}"
    data: list[list[Any]] = json.loads(requests.get(url).text)
    df: pd.DataFrame = pd.DataFrame(data)
    df.columns = ["open_time",
                  "Open", "High", "Low", 'Close', 'Volume',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    for column in ["Open", "High", "Low", 'Close', 'Volume']:
        df[column] = df[column].astype(float)
    if date_index:
        df.index = [dt.datetime.fromtimestamp(i / 1000) for i in df.close_time]
    return df
