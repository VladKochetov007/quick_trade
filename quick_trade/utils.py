from functools import wraps
from logging import basicConfig
from logging import getLogger
from time import sleep
from typing import Any
from typing import List
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Union
from warnings import warn

import pandas as pd
from numpy import arange
from numpy import array
from numpy import exp
from numpy import isnan
from numpy import log
from numpy import nan
from numpy import nan_to_num
from numpy import ndarray
from numpy import polyfit
from pandas import DataFrame
from pandas import Series
from ta.volatility import AverageTrueRange

PREDICT_TYPE: type = int
PREDICT_TYPE_LIST: type = List[PREDICT_TYPE]
CONVERTED_TYPE: type = Union[PREDICT_TYPE, float]
CONVERTED_TYPE_LIST: type = List[CONVERTED_TYPE]

BUY: PREDICT_TYPE = 1
SELL: PREDICT_TYPE = -1
EXIT: PREDICT_TYPE = 0

TEXT_COLOR: str = 'white'

TIME_TITLE: str = 'T I M E'

DEPOSIT_TITLE: str = 'M O N E Y S'
DEPOSIT_NAME: str = 'deposit (start: {0})'  # .format(Trader.deposit_history[0])
DEPOSIT_COLOR: str = 'white'
DEPOSIT_WIDTH: float = 2.0
DEPOSIT_ALPHA: float = 1.0

RETURNS_TITLE: str = 'R E T U R N S'
RETURNS_NAME: str = 'returns'
RETURNS_COLOR: str = DEPOSIT_COLOR
RETURNS_WIDTH: float = 2.0
RETURNS_ALPHA: float = 1.0

DATA_TITLE: str = 'D A T A'
DATA_NAME: str = '{0} {1}'  # '{} {}'.format(Trader.ticker, Trader.interval)
DATA_UP_COLOR = 'green'
DATA_DOWN_COLOR = 'red'
DATA_ALPHA: float = 1.0

AVERAGE_GROWTH_NAME: str = 'average growth'
AVERAGE_GROWTH_COLOR: str = '#F1A5FB'
AVERAGE_GROWTH_WIDTH: float = 1.0
AVERAGE_GROWTH_ALPHA: float = 1.0

STOP_LOSS_NAME: str = 'stop loss'
STOP_LOSS_COLOR: str = '#ff0000'
STOP_LOSS_WIDTH: float = 1.0
STOP_LOSS_ALPHA: float = 0.8

TAKE_PROFIT_NAME: str = 'take profit'
TAKE_PROFIT_COLOR: str = '#55ff00'
TAKE_PROFIT_WIDTH: float = 1.0
TAKE_PROFIT_ALPHA: float = 0.8

OPEN_TRADE_NAME: str = 'open trade'
OPEN_TRADE_COLOR: str = '#0015ff'
OPEN_TRADE_WIDTH: float = 1.0
OPEN_TRADE_ALPHA: float = 0.8

TRADE_MARKER_BUY_NAME: str = 'Buy'
TRADE_MARKER_BUY_TYPE: str = 'triangle-up'
TRADE_MARKER_BUY_COLOR: str = '#55ff00'
TRADE_MARKER_BUY_WIDTH: float = 12.0
TRADE_MARKER_BUY_ALPHA: float = 1.0

TRADE_MARKER_SELL_NAME: str = 'Sell'
TRADE_MARKER_SELL_TYPE: str = 'triangle-down'
TRADE_MARKER_SELL_COLOR: str = '#ff0000'
TRADE_MARKER_SELL_WIDTH: float = 12.0
TRADE_MARKER_SELL_ALPHA: float = 1.0

TRADE_MARKER_EXIT_NAME: str = 'Exit'
TRADE_MARKER_EXIT_TYPE: str = 'triangle-left'
TRADE_MARKER_EXIT_COLOR: str = '#0015ff'
TRADE_MARKER_EXIT_WIDTH: float = 12.0
TRADE_MARKER_EXIT_ALPHA: float = 1.0

TICKER_PATTERN: str = r'[A-Z0-9]+/[A-Z0-9]+'

WAIT_SUCCESS_SLEEP: float = 1.0
WAIT_SUCCESS_PRINT: bool = True
WAIT_SUCCESS_USE: bool = True

MA_FAST_NAME: str = 'SMA{}'  # .format(<SMA length>)
MA_FAST_COLOR: str = '#55ff00'
MA_FAST_WIDTH: float = 1.5
MA_FAST_ALPHA: float = 1.0

MA_MID_NAME: str = 'SMA{}'  # .format(<SMA length>)
MA_MID_COLOR: str = '#0015ff'
MA_MID_WIDTH: float = 3.0
MA_MID_ALPHA: float = 1.0

MA_SLOW_NAME: str = 'SMA{}'  # .format(<SMA length>)
MA_SLOW_COLOR: str = '#ff0000'
MA_SLOW_WIDTH: float = 4.5
MA_SLOW_ALPHA: float = 1.0

ICHIMOKU_LINES_WIDTH: float = 2.0

ICHIMOKU_CLOUD_COLOR: str = 'rgb(200,250,10)'

SENKOU_SPAN_B_NAME: str = 'senkou span b'
SENKOU_SPAN_B_COLOR: str = '#55ff00'
SENKOU_SPAN_B_ALPHA: float = 0.4

SENKOU_SPAN_A_NAME: str = 'senkou span a'
SENKOU_SPAN_A_COLOR: str = '#ff0000'
SENKOU_SPAN_A_ALPHA: float = 0.4

SAR_UP_NAME: str = 'SAR up'
SAR_UP_COLOR: str = '#ffff00'
SAR_UP_WIDTH: float = 2
SAR_UP_ALPHA: float = 1.0

SAR_DOWN_NAME: str = 'SAR down'
SAR_DOWN_COLOR: str = '#ffff00'
SAR_DOWN_WIDTH: float = 2
SAR_DOWN_ALPHA: float = 1.0

ST_UP_NAME: str = 'SuperTrend up'
ST_UP_COLOR: str = '#ff0000'
ST_UP_WIDTH: float = 2
ST_UP_ALPHA: float = 1.0

ST_DOWN_NAME: str = 'SuperTrend down'
ST_DOWN_COLOR: str = '#55ff00'
ST_DOWN_WIDTH: float = 2
ST_DOWN_ALPHA: float = 1.0

UPPER_BB_NAME: str = 'upper band'
UPPER_BB_COLOR: str = '#00ff7b'
UPPER_BB_WIDTH: float = 2
UPPER_BB_ALPHA: float = 1.0

MID_BB_NAME: str = 'mid band'
MID_BB_COLOR: str = '#fff200'
MID_BB_WIDTH: float = 2
MID_BB_ALPHA: float = 1.0

LOWER_BB_NAME: str = 'lower band'
LOWER_BB_COLOR: str = '#ff0000'
LOWER_BB_WIDTH: float = 2
LOWER_BB_ALPHA: float = 1.0

INFO_TEXT: str = """losses: {}
trades: {}
profits: {}
mean year percentage profit: {}%
winrate: {}%
mean deviation: {}%"""  # .format(Trader.losses, Trader.trades, Trader.profits, Trader.year_profit, Trader.winrate, Trader.mean_deviation)

__version__: str = "6.7.0"
__author__: str = 'Vlad Kochetov'
__credits__: List[str] = [
    "Hemerson Tacon -- Stack overflow",
    "hpaulj -- Stack overflow",
    "furas -- Stack overflow",
    "Devin Jeanpierre (edit: wjandrea) -- Stack overflow",
    "Войтенко Николай Поликарпович (Vojtenko Nikolay Polikarpovich) -- helped me test the system of interaction with the binance crypto exchange with 50 dollars.",
    "https://fxgears.com/index.php?threads/how-to-acquire-free-historical-tick-and-bar-data-for-algo-trading-and-backtesting-in-2020-stocks-forex-and-crypto-currency.1229/#post-19305 -- binance get historical data method",
    "https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/ and https://teletype.in/@cozy_codespace/Hk70-Ntl4 -- heroku and threading problems",
    "https://stackoverflow.com/questions/57838939/handling-exceptions-with-bulk-api-requests -- IEX token",
    "Igor Kroitor -- donate 0.5 ETH (~1320$)",
    "Igor Kroitor -- Helped to solve the problem with exception ConnectionError(10054).",
    "https://stackoverflow.com/questions/27333671/how-to-solve-the-10054-error",
    "Pavel Fedotov (https://github.com/Pfed-prog) -- pull request https://github.com/VladKochetov007/quick_trade/pull/60"
]

logger = getLogger(__name__)
getLogger('ccxt').setLevel(30)
getLogger('urllib3').setLevel(30)
logger.setLevel(10)

basicConfig(level=0,
            filename='trading.log',
            format='%(asctime)s [%(levelname)s]\n%(message)s\n'
                   f'[QUICK_TRADE VERSION: {__version__}] [FUNCTION: %(funcName)s] [FILE "%(module)s", '
                   'LINE %(lineno)d] %(name)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] '
                   '[FILEPATH: %(pathname)s]\n')


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
        long, short = [nan] * m, [nan] * m
        ATR = AverageTrueRange(high=self.high, low=self.low, close=self.close,
                               window=self.length)

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
            ret[e + 1] = nan
    return ret


def anti_convert(converted: CONVERTED_TYPE_LIST,
                 _nan_num: float = 18699.9) -> PREDICT_TYPE_LIST:
    converted = nan_to_num(converted, nan=_nan_num)
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


def get_exponential_growth(dataset: Sequence[float]) -> ndarray:
    x = arange(1, len(dataset) + 1, 1)
    b, a = polyfit(x, log(array(dataset)), 1)
    regression = exp(a + b * x)
    return array(regression)


def get_coef_sec(timeframe: str = '1d') -> Tuple[float, int]:
    profit_calculate_coef: Union[float, int]
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
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not WAIT_SUCCESS_USE:
                    raise e
                if not isinstance(e, KeyboardInterrupt):
                    if WAIT_SUCCESS_PRINT:
                        print(f'An error occurred: {e}, repeat request')
                    logger.error(f'An error occurred: {e}', exc_info=True)
                    sleep(WAIT_SUCCESS_SLEEP)
                    continue
                else:
                    raise e

    return checker


def profit_factor(deposit_list: Sequence[float]) -> float:
    return deposit_list[1] / deposit_list[0]


def assert_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AssertionError as AE:
            logger.critical(AE)
            raise AE

    return wrapper


def get_diff(price: float,
             low: float,
             high: float,
             stop_loss: float,
             take_profit: float,
             signal: PREDICT_TYPE) -> float:
    if signal == EXIT:
        return 0.0

    if signal == BUY and low <= stop_loss:
        return stop_loss - price

    elif signal == SELL and high >= stop_loss:
        return stop_loss - price

    elif signal == BUY and high >= take_profit:
        return take_profit - price

    elif signal == SELL and low <= take_profit:
        return take_profit - price


def make_multi_trade_returns(converted_returns: CONVERTED_TYPE_LIST) -> Tuple[PREDICT_TYPE_LIST, List[int]]:
    if EXIT in converted_returns:
        warn('The use of utils.EXIT is deprecated in this type of strategy. If utils.EXIT is the first item in the sequence, you can replace it with np.nan.')
    result_credlev: List[int] = []
    result_returns: PREDICT_TYPE_LIST = [BUY] * len(converted_returns)
    flag_lev: int = 0
    if isnan(converted_returns[0]):
        converted_returns[0] = EXIT
    ret: CONVERTED_TYPE
    for ret in converted_returns:
        if not isnan(ret):
            if ret is BUY:
                flag_lev += 1
            elif ret is SELL:
                flag_lev -= 1
        result_credlev.append(flag_lev)
    e: int
    lev: int
    for e, lev in enumerate(result_credlev):
        if lev < 0:
            result_credlev[e] = -lev
            result_returns[e] = SELL
        elif lev == 0:
            result_credlev[e] = 1
            result_returns[e] = EXIT
    return result_returns, result_credlev


def get_multipliers(df: pd.Series) -> pd.Series:
    df = df.reset_index(drop=True)
    ret: pd.Series = df / df.shift(1)
    ret[0] = 1
    return ret


def mean_deviation(frame: Series, avg_grwth: ndarray) -> float:
    relative_diff = abs(frame.values - avg_grwth) / avg_grwth
    return relative_diff.mean()
