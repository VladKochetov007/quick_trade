from functools import wraps
from logging import basicConfig, getLogger
from time import sleep
from typing import Any, List, Union, Tuple, Sequence, Sized

import numpy as np
from pandas import DataFrame, Series
from ta.volatility import AverageTrueRange

PREDICT_TYPE: type = int
PREDICT_TYPE_LIST: type = List[PREDICT_TYPE]
CONVERTED_TYPE: type = Union[PREDICT_TYPE, float]
CONVERTED_TYPE_LIST: type = List[CONVERTED_TYPE]

RED: str = '#ff0000'
GREEN: str = '#55ff00'
BLUE: str = '#0015ff'
CYAN: str = 'cyan'
BUY: PREDICT_TYPE = 1
SELL: PREDICT_TYPE = 0
EXIT: PREDICT_TYPE = 2

__author__: str = 'Vlad Kochetov'
__credits__: List[str] = ["Hemerson Tacon -- Stack overflow",
                          "hpaulj -- Stack overflow",
                          "furas -- Stack overflow",
                          "Devin Jeanpierre (edit: wjandrea) -- Stack overflow",
                          "Войтенко Николай Поликарпович (Vojtenko Nikolay Polikarpovich) -- helped me test the "
                          "system of interaction with the binance crypto exchange with 50 dollars.",

                          "https://fxgears.com/index.php?threads/how-to-acquire-free-historical-tick-and-bar-data-for"
                          "-algo-trading-and-backtesting-in-2020-stocks-forex-and-crypto-currency.1229/#post-19305"
                          " -- binance get historical data method",
                          "https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/ and "
                          "https://teletype.in/@cozy_codespace/Hk70-Ntl4 -- heroku and threading problems",
                          "https://stackoverflow.com/questions/57838939/handling-exceptions-with-bulk-api-requests --"
                          "IEX token",
                          "Igor Kroitor -- donate 0.5 ETH (~1320$)",
                          "Igor Kroitor -- Helped to solve the problem with exception ConnectionError(10054).",
                          "https://stackoverflow.com/questions/27333671/how-to-solve-the-10054-error"]
__version__: str = "5.0.0"

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
WAIT_SUCCESS_SLEEP = 15
WAIT_SUCCESS_PRINT = True
USE_WAIT_SUCCESS = True

logger = getLogger()
logger.setLevel(0)
basicConfig(level=20, filename='trading.log', format='%(name)s::%(asctime)s::[%(levelname)s] %(message)s')


class SuperTrendIndicator(object):
    """

    Supertrend (ST)
    """
    close: Series
    high: Series
    low: Series

    def __init__(self,
                 close: Series,
                 high: Series,
                 low: Series,
                 multiplier: float = 3.0,
                 length: int = 10):
        self.close = close
        self.high = high
        self.low = low
        self.multiplier: float = multiplier
        self.length = length
        self._all = self._get_all_ST()

    def get_supertrend(self) -> Series:
        return self._all['ST']

    def get_supertrend_upper(self) -> Series:
        return self._all['ST_upper']

    def get_supertrend_lower(self) -> Series:
        return self._all['ST_lower']

    def get_supertrend_strategy_returns(self) -> Series:
        """

        :return: pd.Series with 1 or -1 (buy, sell)
        """
        return self._all['ST_strategy']

    def get_all_ST(self) -> DataFrame:
        return self._all

    def _get_all_ST(self) -> DataFrame:
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
        df = DataFrame(
            {
                f"ST": trend,
                f"ST_strategy": dir_,
                f"ST_lower": long,
                f"ST_upper": short,
            },
            index=self.close.index
        )

        return df


def convert(data: PREDICT_TYPE_LIST) -> CONVERTED_TYPE_LIST:
    ret: List[Any] = list(data.copy())
    e: int
    for e, i in enumerate(data[1:]):
        if i == data[e]:
            ret[e + 1] = np.nan
    return ret


def anti_convert(converted: CONVERTED_TYPE_LIST, _nan_num: float = 18699.9) -> PREDICT_TYPE_LIST:
    converted = np.nan_to_num(converted, nan=_nan_num)
    ret: List[Any] = [converted[0]]
    flag = converted[0]
    e: int
    for i in converted[1:]:
        if i == _nan_num:
            ret.append(flag)
        else:
            ret.append(i)
            flag = i
    return ret


def get_window(values: Union[Sequence, Sized], window_length: int) -> List[Any]:
    ret: List[Any] = []
    for e, i in enumerate(values[:len(values) - window_length + 1]):
        ret.append(values[e:e + window_length])
    return ret


def convert_signal_str(predict: PREDICT_TYPE) -> str:
    if predict == BUY:
        return 'Buy'
    elif predict == SELL:
        return 'Sell'
    elif predict == EXIT:
        return 'Exit'

def get_exponential_growth(dataset: Sequence[float]) -> np.ndarray:
    return_list: List[float] = []
    coef = profit_factor(dataset)
    curr = dataset[0]
    for i in range(len(dataset)):
        return_list.append(curr)
        curr *= coef
    return np.array(return_list)


def get_coef_sec(timeframe: str = '1d') -> Tuple[float, int]:
    profit_calculate_coef: float
    sec_interval: int
    if timeframe == '1m':
        profit_calculate_coef = (60 * 24 * 365)
        sec_interval = 60
    elif timeframe == '2m':
        profit_calculate_coef = (30 * 24 * 365)
        sec_interval = 120
    elif timeframe == '3m':
        profit_calculate_coef = (20 * 24 * 365)
        sec_interval = 180
    elif timeframe == '5m':
        profit_calculate_coef = (12 * 24 * 365)
        sec_interval = 300
    elif timeframe == '15m':
        profit_calculate_coef = (4 * 24 * 365)
        sec_interval = 15 * 60
    elif timeframe == '30m':
        profit_calculate_coef = (2 * 24 * 365)
        sec_interval = 60 * 30
    elif timeframe == '45m':
        profit_calculate_coef = (32 * 365)
        sec_interval = 60 * 45
    elif timeframe == '1h':
        profit_calculate_coef = (24 * 365)
        sec_interval = 60 * 60
    elif timeframe == '90m':
        profit_calculate_coef = (18 * 365)
        sec_interval = 60 * 90
    elif timeframe == '2h':
        profit_calculate_coef = (12 * 365)
        sec_interval = 60 * 60 * 2
    elif timeframe == '3h':
        profit_calculate_coef = (8 * 365)
        sec_interval = 60 * 60 * 3
    elif timeframe == '4h':
        profit_calculate_coef = (6 * 365)
        sec_interval = 60 * 60 * 4
    elif timeframe == '12h':
        profit_calculate_coef = (2 * 365)
        sec_interval = 60 * 60 * 12
    elif timeframe == '1d':
        profit_calculate_coef = 365
        sec_interval = 60 * 60 * 24
    elif timeframe == '3d':
        profit_calculate_coef = (365 / 3)
        sec_interval = 86400 * 3
    elif timeframe == '1w':
        profit_calculate_coef = 52
        sec_interval = 86400 * 7
    elif timeframe == '1M':
        profit_calculate_coef = 12
        sec_interval = 86400 * 30
    elif timeframe == '3M':
        profit_calculate_coef = 4
        sec_interval = 86400 * 90
    elif timeframe == '6M':
        profit_calculate_coef = 2
        sec_interval = 86400 * 180
    else:
        raise ValueError(f'incorrect interval; {timeframe}')
    return profit_calculate_coef, sec_interval


def wait_success(func):
    @wraps(func)
    def checker(*args, **kwargs):
        if USE_WAIT_SUCCESS:
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not isinstance(e, KeyboardInterrupt):
                        if WAIT_SUCCESS_PRINT:
                            print(f'An error occurred: {e}')
                        logger.error(f'An error occurred: {e}', exc_info=True)
                        sleep(WAIT_SUCCESS_SLEEP)
                        continue
                    else:
                        raise e
        else:
            return func(*args, **kwargs)

    return checker

def root(x: float, pwr: float = 2) -> float:
    return x ** (1/pwr)

def profit_factor(deposit_list: Sequence) -> float:
    return root(deposit_list[-1]/deposit_list[0], len(deposit_list)-1)
