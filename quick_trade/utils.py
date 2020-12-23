import datetime as dt
import itertools
import json
import logging
import os
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import requests
from iexfinance.stocks import get_historical_intraday
from ta.volatility import AverageTrueRange

PREDICT_TYPE: type = int
PREDICT_TYPE_LIST: type = List[PREDICT_TYPE]
SETED_TYPE: type = Union[PREDICT_TYPE, float]
SETED_TYPE_LIST: type = List[SETED_TYPE]

R: str = '#ff0000'
G: str = '#55ff00'
B: str = '#0015ff'
C: str = 'cyan'
REGRESSION_INPUTS: int = 100
BUY: PREDICT_TYPE = 1
SELL: PREDICT_TYPE = 0
EXIT: PREDICT_TYPE = 2

IEX_TOKEN: str = 'Tpk_a4bc3e95d4c94810a3b2d4138dc81c5d'

os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = IEX_TOKEN
__author__: str = 'Vlad Kochetov'
__credits__: List[str] = ["Hemerson Tacon  -- Stack overflow",
                          "hpaulj          -- Stack overflow",
                          "furas           -- Stack overflow",
                          "Войтенко Николай Поликарпович (Vojtenko Nikolaj Polikarpovich) -- helped me test the "
                          "system of interaction with the binance crypto exchange with 50 dollars.",
                          "https://fxgears.com/index.php?threads/how-to-acquire-free-historical-tick-and-bar-data-for"
                          "-algo-trading-and-backtesting-in-2020-stocks-forex-and-crypto-currency.1229/#post-19305"]
__version__: str = "3.8.5"

BASE_TICKER: str = '^DJI'
SCATTER_SIZE: float = 12.0
SCATTER_ALPHA: float = 1.0
TAKE_STOP_OPN_WIDTH: float = 1.0
ICHIMOKU_LINES_WIDTH: float = 2.0
ICHIMOKU_CLOUD_COLOR: str = 'rgb(0,250,250)'
ICHIMOKU_CLOUD_ALPHA: float = 0.4
TEXT_COLOR: str = 'white'
SUB_LINES_WIDTH: float = 3.0
STOP_TAKE_OPN_ALPHA: float = 0.8
COLOR_DEPOSIT: str = 'white'
DEPO_COLOR_UP: str = 'green'
DEPO_COLOR_DOWN: str = 'red'

FILE_LOG_NAME: str = 'trading.log'
logger = logging.getLogger()
logger.setLevel(30)
logging.basicConfig(level=20, filename=FILE_LOG_NAME)


class SuperTrendIndicator(object):
    """

    Supertrend (ST)
    """
    close: pd.Series
    high: pd.Series
    low: pd.Series

    def __init__(self,
                 close: pd.Series,
                 high: pd.Series,
                 low: pd.Series,
                 multiplier: float = 3.0,
                 length: int = 10):
        self.close = close
        self.high = high
        self.low = low
        self.multiplier: float = multiplier
        self.length = length
        self._all = self._get_all_ST()

    def get_supertrend(self) -> pd.Series:
        return self._all['ST']

    def get_supertrend_upper(self) -> pd.Series:
        return self._all['ST_upper']

    def get_supertrend_lower(self) -> pd.Series:
        return self._all['ST_lower']

    def get_supertrend_strategy_returns(self) -> pd.Series:
        """

        :return: pd.Series with 1 or -1 (buy, sell)
        """
        return self._all['ST_strategy']

    def get_all_ST(self) -> pd.DataFrame:
        return self._all

    def _get_all_ST(self) -> pd.DataFrame:
        """

        ST Indicator, trading predictions, ST high/low
        """
        m = self.close.size
        dir_, trend = [1] * m, [0] * m
        long, short = [np.NaN] * m, [np.NaN] * m
        ATR = AverageTrueRange(high=self.high, low=self.low, close=self.close, window=self.length)

        hl2_ = (self.high + self.low) / 2
        matr = ATR.average_true_range() * self.multiplier
        upperband = hl2_ + matr
        lowerband = hl2_ - matr

        for i in range(1, m):
            if self.close.iloc[i] > upperband.iloc[i - 1]:
                dir_[i] = BUY
            elif self.close.iloc[i] < lowerband.iloc[i - 1]:
                dir_[i] = SELL
            else:
                dir_[i] = dir_[i - 1]
                if dir_[i] == BUY and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                    lowerband.iloc[i] = lowerband.iloc[i - 1]
                if dir_[i] == SELL and upperband.iloc[i] > upperband.iloc[i - 1]:
                    upperband.iloc[i] = upperband.iloc[i - 1]

            if dir_[i] > 0:
                trend[i] = long[i] = lowerband.iloc[i]
            else:
                trend[i] = short[i] = upperband.iloc[i]

        # Prepare DataFrame to return
        df = pd.DataFrame(
            {
                f"ST": trend,
                f"ST_strategy": dir_,
                f"ST_lower": long,
                f"ST_upper": short,
            },
            index=self.close.index
        )

        return df


def expansion_with_shear(values, ins=EXIT) -> PREDICT_TYPE_LIST:
    ret: PREDICT_TYPE_LIST = []
    for value in values:
        for column in range(4):
            ret.append(value)
    for i in range(3):
        ret.insert(0, ins)
    return ret[:-3]


def set_(data: Any) -> SETED_TYPE_LIST:
    ret: List[Any] = list(data.copy())
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
    predict: List[List[float]] = [[] for _ in range(4)]
    for it in range(4):
        for i in range(it, len(data), 4):
            predict[it].append(data[i])
    return pd.DataFrame(predict, index=columns).T


def inverse_4_col_df(df: pd.DataFrame, columns: List[Any]) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(df)
    ret: np.ndarray = df.values
    rets: List[pd.DataFrame] = []
    for column in columns:
        rets.append(
            pd.DataFrame(itertools.chain.from_iterable(ret), columns=[column]))
    return pd.concat(rets, axis=1)


def get_IEX_minutely(ticker: str, undo_days: int) -> pd.DataFrame:
    tuples: List[Tuple[int]] = []
    returns: pd.DataFrame
    end: dt.datetime = dt.datetime.today()
    date_list: List[dt.datetime] = [end - dt.timedelta(days=d) for d in range(undo_days)]
    for i in range(len(date_list)):
        ret = str(np.array(date_list[::-1])[i])[:10]
        ret = tuple(map(int, ret.split('-')))
        tuples.append(ret)

    dataframes: List[pd.DataFrame] = []
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


def anti_set_(seted: List[Any]) -> List[Any]:
    ret: List[Any] = [seted[0]]
    flag = seted[0]
    e: int
    for e, i in enumerate(seted[1:]):
        if i is np.nan:
            ret.append(flag)
        else:
            ret.append(i)
            flag = i
    return ret


def digit(data) -> PREDICT_TYPE_LIST:
    ret: PREDICT_TYPE_LIST = []
    for element in list(data):
        if element == 0:
            ret.append(EXIT)
        elif element > 0:
            ret.append(BUY)
        else:
            ret.append(SELL)
    return ret


def get_window(values, window_length: int) -> List[Any]:
    ret: List[Any] = []
    for e, i in enumerate(values[:len(values) - window_length + 1]):
        ret.append(values[e:e + window_length])
    return ret


def get_binance_data(ticker: str = "BNBBTC", interval: str = "1m", date_index: bool = False):
    url: str = f"https://api.binance.com/api/v1/klines?symbol={ticker}&interval={interval}"
    data: List[List[Any]] = json.loads(requests.get(url).text)
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


def min_admit(r: int) -> float:
    """

    to math.floor analogue.
    """
    return round(float('0.' + '0' * (r - 1) + '1'), r)


def convert_signal_str(predict: PREDICT_TYPE) -> str:
    if predict == BUY:
        return 'Buy'
    elif predict == SELL:
        return 'Sell'
    elif predict == EXIT:
        return 'Exit'
