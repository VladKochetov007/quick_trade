import numpy as np
import pandas as pd
import ta.volatility
from numpy import nan
from pandas import DataFrame
from pandas import Series
from ta.volatility import AverageTrueRange
from .utils import BUY, SELL
from typing import Union

class Indicator:
    pass

class SuperTrendIndicator(Indicator):
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
        return self._all['ST_strategy']

    def get_all_ST(self) -> DataFrame:
        return self._all

    def _get_all_ST(self) -> DataFrame:
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


class PriceChannel(Indicator):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 support_period: int = 20,
                 resistance_period: int = 20,
                 channel_part: float = 1.0):
        self._support_period = support_period
        self._resistance_period = support_period
        self._high = high
        self._low = low
        self._part = channel_part
        self._run()

    @staticmethod
    def _run_lev(func, period, prices):
        channel = []
        for roll in prices.rolling(period):
            channel.append(func(roll))
        return channel

    def _handle_levels(self, support, resistance):
        self.high = []
        self.low = []

        for low, high in zip(support, resistance):
            mid = (low + high) / 2
            diff = high - low

            new_low = mid - (diff*self._part)/2
            new_high = mid + (diff*self._part)/2

            self.high.append(new_high)
            self.low.append(new_low)

    def _run(self):
        support = self._run_lev(lambda x: x.min(),
                                self._support_period,
                                self._low)
        resistance = self._run_lev(lambda x: x.max(),
                                   self._resistance_period,
                                   self._high)
        self._handle_levels(support=support,
                            resistance=resistance)

    def higher_line(self):
        return self.high

    def lower_line(self):
        return self.low
