# !/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)


# TODO:
#   add tradingview realtime signals
#   numpy
#   scalper and dca bot
#   unit-tests
#   more docs and examples
#   make normal logs
#   decimal
#   3.9

from copy import copy
from datetime import datetime
from threading import Thread
from time import ctime, sleep, time
from typing import Dict, List, Tuple, Any, Iterable, Union, Sized

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.others
import ta.trend
import ta.volatility
import ta.volume
from plotly.subplots import make_subplots
from quick_trade import brokers
from quick_trade import utils

Line = dict  # To avoid the deprecation warning


class Trader(object):
    """
    algo-trading system.
    ticker:   |     str      |  ticker/symbol of chart
    df:       |   dataframe  |  data of chart
    interval: |     str      |  interval of df.

    """
    _profit_calculate_coef: float
    returns: utils.PREDICT_TYPE_LIST = []
    df: pd.DataFrame
    ticker: str
    interval: str
    __exit_order__: bool = False
    _prev_predict: str = 'Exit'
    _stop_loss: float
    _take_profit: float
    _open_price: float
    trades: int = 0
    profits: int = 0
    losses: int = 0
    _stop_losses: List[float]
    _take_profits: List[float]
    _credit_leverages: List[float]
    deposit_history: List[float]
    year_profit: float
    average_growth: np.ndarray
    info: str
    _backtest_out_no_drop: pd.DataFrame
    backtest_out: pd.DataFrame
    _open_lot_prices: List[float]
    realtime_returns: Dict[str, Dict[str, Union[str, float]]]
    client: brokers.TradingClient
    __last_stop_loss: float
    __last_take_profit: float
    returns_strategy_diff: List[float]
    _sec_interval: int
    supports: Dict[int, float]
    resistances: Dict[int, float]
    __last_credit_leverage: float
    __prev_credit_lev: float = 1
    trading_on_client: bool

    def __init__(self,
                 ticker: str = 'BTC/USDT',
                 df: pd.DataFrame = pd.DataFrame(),
                 interval: str = '1d',
                 trading_on_client: bool = True):
        self.df = df.reset_index(drop=True)
        self.ticker = ticker
        self.interval = interval
        self.trading_on_client = trading_on_client
        self._profit_calculate_coef, self._sec_interval = utils.get_coef_sec(interval)
        self.__exit_order__ = False

    def __repr__(self):
        return f'{self.ticker} {self.interval} trader'

    def _get_attr(self, attr: str):
        return getattr(self, attr)

    @classmethod
    def _get_this_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __get_stop_take(self, sig: utils.PREDICT_TYPE) -> Dict[str, float]:
        """
        calculating stop loss and take profit.
        sig:        |     PREDICT_TYPE     |  signal to sell/buy/exit:
            EXIT -- exit.
            BUY -- buy.
            SELL -- sell.
        """

        _stop_loss: float
        take: float
        if self._stop_loss is not np.inf:
            _stop_loss = self._stop_loss / 10_000 * self._open_price
        else:
            _stop_loss = np.inf
        if self._take_profit is not np.inf:
            take = self._take_profit / 10_000 * self._open_price
        else:
            take = np.inf

        if sig == utils.BUY:
            _stop_loss = self._open_price - _stop_loss
            take = self._open_price + take
        elif sig == utils.SELL:
            take = self._open_price - take
            _stop_loss = self._open_price + _stop_loss
        else:
            if self._take_profit is not np.inf:
                take = self._open_price
            if self._stop_loss is not np.inf:
                _stop_loss = self._open_price
        utils.logger.debug(
            f'stop loss: {_stop_loss} ({self._stop_loss} pips), take profit: {take} ({self._take_profit} pips)')

        return {'stop': _stop_loss,
                'take': take}

    def sl_tp_adder(self, add_stop_loss: float = 0, add_take_profit: float = 0) -> Tuple[List[float], List[float]]:
        """

        :param add_stop_loss: add stop loss points
        :param add_take_profit: add take profit points
        :return: (stop losses, take profits)
        """
        stop_losses = []
        take_profits = []
        for stop_loss_price, take_profit_price, price, sig in zip(self._stop_losses,
                                                                  self._take_profits,
                                                                  self._open_lot_prices,
                                                                  self.returns):
            add_sl = (price / 10_000) * add_stop_loss
            add_tp = (price / 10_000) * add_take_profit

            if sig == utils.BUY:
                stop_losses.append(stop_loss_price - add_sl)
                take_profits.append(take_profit_price + add_tp)
            elif sig == utils.SELL:
                stop_losses.append(stop_loss_price + add_sl)
                take_profits.append(take_profit_price - add_tp)
            else:
                stop_losses.append(stop_loss_price)
                take_profits.append(take_profit_price)

        self._stop_losses = stop_losses
        self._take_profits = take_profits
        return self._stop_losses, self._take_profits

    def strategy_diff(self, frame_to_diff: pd.Series) -> utils.PREDICT_TYPE_LIST:
        """
        frame_to_diff:  |   pd.Series  |  example:  Trader.df['Close']
        """
        self.returns = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        return self.returns

    def strategy_buy_hold(self) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.BUY for _ in range(len(self.df))]
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_2_sma(self,
                       slow: int = 100,
                       fast: int = 30,
                       plot: bool = True) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        SMA1 = ta.trend.sma_indicator(self.df['Close'], fast)
        SMA2 = ta.trend.sma_indicator(self.df['Close'], slow)
        if plot:
            self.fig.add_trace(
                Line(
                    name=f'SMA{fast}',
                    y=SMA1.values,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.GREEN)), 1, 1)
            self.fig.add_trace(
                Line(
                    name=f'SMA{slow}',
                    y=SMA2.values,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.RED)), 1, 1)

        for SMA13, SMA26 in zip(SMA1, SMA2):
            if SMA26 < SMA13:
                self.returns.append(utils.BUY)
            elif SMA13 < SMA26:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        return self.returns

    def strategy_3_sma(self,
                       slow: int = 100,
                       mid: int = 26,
                       fast: int = 13,
                       plot: bool = True) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        SMA1 = ta.trend.sma_indicator(self.df['Close'], fast)
        SMA2 = ta.trend.sma_indicator(self.df['Close'], mid)
        SMA3 = ta.trend.sma_indicator(self.df['Close'], slow)

        if plot:
            for SMA, Co, name in zip([SMA1, SMA2, SMA3],
                                     [utils.GREEN, utils.BLUE, utils.RED],
                                     [fast, mid, slow]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=SMA.values,
                        line=dict(width=utils.SUB_LINES_WIDTH, color=Co)), 1, 1)

        for SMA13, SMA26, SMA100 in zip(SMA1, SMA2, SMA3):
            if SMA100 < SMA26 < SMA13:
                self.returns.append(utils.BUY)
            elif SMA100 > SMA26 > SMA13:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)

        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_3_ema(self,
                       slow: int = 46,
                       mid: int = 21,
                       fast: int = 3,
                       plot: bool = True) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        ema3 = ta.trend.ema_indicator(self.df['Close'], fast)
        ema21 = ta.trend.ema_indicator(self.df['Close'], mid)
        ema46 = ta.trend.ema_indicator(self.df['Close'], slow)

        if plot:
            for ema, Co, name in zip([ema3.values, ema21.values, ema46.values],
                                     [utils.GREEN, utils.BLUE, utils.RED], [slow, mid, fast]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=ema,
                        line=dict(width=utils.SUB_LINES_WIDTH, color=Co)), 1, 1)

        for EMA1, EMA2, EMA3 in zip(ema3, ema21, ema46):
            if EMA1 > EMA2 > EMA3:
                self.returns.append(utils.BUY)
            elif EMA1 < EMA2 < EMA3:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_macd(self,
                      slow: int = 100,
                      fast: int = 30) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        diff = ta.trend.macd_diff(self.df['Close'], slow, fast)

        for j in diff:
            if j > 0:
                self.returns.append(utils.BUY)
            elif 0 > j:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_rsi(self,
                     minimum: float = 20,
                     maximum: float = 80,
                     max_mid: float = 75,
                     min_mid: float = 35,
                     **rsi_kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        rsi = ta.momentum.rsi(close=self.df['Close'], **rsi_kwargs)
        flag: utils.PREDICT_TYPE = utils.EXIT

        for val in rsi.values:
            if val < minimum:
                flag = utils.BUY
            elif val > maximum:
                flag = utils.SELL
            elif flag == utils.BUY and val < max_mid:
                flag = utils.EXIT
            elif flag == utils.SELL and val > min_mid:
                flag = utils.EXIT
            self.returns.append(flag)

        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_parabolic_SAR(self, plot: bool = True, **sar_kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        sar: ta.trend.PSARIndicator = ta.trend.PSARIndicator(self.df['High'], self.df['Low'],
                                                             self.df['Close'], **sar_kwargs)
        sardown: np.ndarray = sar.psar_down().values
        sarup: np.ndarray = sar.psar_up().values
        self._stop_losses = list(sar.psar().values)

        if plot:
            for SAR_ in (sarup, sardown):
                self.fig.add_trace(
                    Line(
                        name='SAR', y=SAR_, line=dict(width=utils.SUB_LINES_WIDTH)),
                    1, 1)
        for price, up, down in zip(
                list(self.df['Close'].values), list(sarup), list(sardown)):
            numup = np.nan_to_num(up, nan=-9999.0)
            numdown = np.nan_to_num(down, nan=-9999.0)
            if numup != -9999:
                self.returns.append(utils.BUY)
            elif numdown != -9999:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_credit_leverages()
        self.set_open_stop_and_take(set_stop=False)
        return self.returns

    def strategy_macd_histogram_diff(self,
                                     slow: int = 23,
                                     fast: int = 12,
                                     **macd_kwargs) -> utils.PREDICT_TYPE_LIST:
        _MACD_ = ta.trend.MACD(self.df['Close'], slow, fast, **macd_kwargs)
        signal_ = _MACD_.macd_signal()
        macd_ = _MACD_.macd()
        histogram: pd.DataFrame = pd.DataFrame(macd_.values - signal_.values)
        for element in histogram.diff().values:
            if element == 0:
                self.returns.append(utils.EXIT)
            elif element > 0:
                self.returns.append(utils.BUY)
            else:
                self.returns.append(utils.SELL)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def strategy_supertrend(self, plot: bool = True, *st_args, **st_kwargs) -> utils.PREDICT_TYPE_LIST:
        st: utils.SuperTrendIndicator = utils.SuperTrendIndicator(self.df['Close'],
                                                                  self.df['High'],
                                                                  self.df['Low'],
                                                                  *st_args,
                                                                  **st_kwargs)
        if plot:
            self.fig.add_trace(Line(y=st.get_supertrend_upper(),
                                    name='supertrend upper',
                                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.RED)))
            self.fig.add_trace(Line(y=st.get_supertrend_lower(),
                                    name='supertrend lower',
                                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.GREEN)))
        self._stop_losses = list(st.get_supertrend())
        self.returns = list(st.get_supertrend_strategy_returns())
        self._stop_losses[0] = np.inf if self.returns[0] == utils.SELL else -np.inf
        self.set_open_stop_and_take(set_stop=False)
        self.set_credit_leverages()
        return self.returns

    def strategy_bollinger(self,
                           plot: bool = True,
                           to_mid: bool = True,
                           *bollinger_args,
                           **bollinger_kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        flag: utils.PREDICT_TYPE = utils.EXIT
        bollinger: ta.volatility.BollingerBands = ta.volatility.BollingerBands(self.df['Close'],
                                                                               fillna=True,
                                                                               *bollinger_args,
                                                                               **bollinger_kwargs)

        mid_: pd.Series = bollinger.bollinger_mavg()
        upper: pd.Series = bollinger.bollinger_hband()
        lower: pd.Series = bollinger.bollinger_lband()
        if plot:
            name: str
            TR: pd.Series
            for TR, name in zip([upper, mid_, lower], ['upper band', 'mid band', 'lower band']):
                self.fig.add_trace(Line(y=TR, name=name, line=dict(width=utils.SUB_LINES_WIDTH)), col=1, row=1)
        close: float
        up: float
        mid: float
        low: float
        for close, up, mid, low in zip(self.df['Close'].values,
                                       upper,
                                       mid_,
                                       lower):
            if close <= low:
                flag = utils.BUY
            if close >= up:
                flag = utils.SELL

            if to_mid:
                if flag == utils.SELL and close <= mid:
                    flag = utils.EXIT
                if flag == utils.BUY and close >= mid:
                    flag = utils.EXIT
            self.returns.append(flag)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        return self.returns

    def get_heikin_ashi(self, df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        """
        :param df: dataframe, standard: self.df
        :return: heikin ashi
        """
        if 'Close' not in df.columns:
            df: pd.DataFrame = self.df.copy()
        df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        df['HA_Open'] = (df['Open'].shift(1) + df['Open'].shift(1)) / 2
        df.iloc[0, df.columns.get_loc("HA_Open")] = (df.iloc[0]['Open'] + df.iloc[0]['Close']) / 2
        df['HA_High'] = df[['High', 'Low', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['High', 'Low', 'HA_Open', 'HA_Close']].min(axis=1)
        df = df.drop(['Open', 'High', 'Low', 'Close'], axis=1)
        df = df.rename(
            columns={"HA_Open": "Open",
                     "HA_High": "High",
                     "HA_Low": "Low",
                     "HA_Close": "Close"})

        return df

    def strategy_ichimoku(self,
                          tenkansen: int = 9,
                          kijunsen: int = 26,
                          senkouspan: int = 52,
                          chinkouspan: int = 26,
                          stop_loss_plus: float = 40.0,
                          plot: bool = True) -> utils.PREDICT_TYPE_LIST:
        cloud = ta.trend.IchimokuIndicator(self.df["High"],
                                           self.df["Low"],
                                           tenkansen,
                                           kijunsen,
                                           senkouspan,
                                           visual=True)
        tenkan_sen: np.ndarray = cloud.ichimoku_conversion_line().values
        kinjun_sen: np.ndarray = cloud.ichimoku_base_line().values
        senkou_span_a: np.ndarray = cloud.ichimoku_a().values
        senkou_span_b: np.ndarray = cloud.ichimoku_b().values
        prices: pd.Series = self.df['Close']
        chenkou_span: np.ndarray = prices.shift(-chinkouspan).values
        flag1: utils.PREDICT_TYPE = utils.EXIT
        flag2: utils.PREDICT_TYPE = utils.EXIT
        flag3: utils.PREDICT_TYPE = utils.EXIT
        trade: utils.PREDICT_TYPE = utils.EXIT
        name: str
        data: np.ndarray
        e: int
        close: float
        tenkan: float
        kijun: float
        A: float
        B: float
        chickou: float

        if plot:
            for name, data, color in zip(['tenkan-sen',
                                          'kijun-sen',
                                          'chinkou-span'],
                                         [tenkan_sen,
                                          kinjun_sen,
                                          chenkou_span],
                                         ['red',
                                          'blue',
                                          'green']):
                self.fig.add_trace(Line(y=data, name=name, line=dict(width=utils.ICHIMOKU_LINES_WIDTH, color=color)),
                                   col=1, row=1)

            self.fig.add_trace(Line(y=senkou_span_a,
                                    fill=None,
                                    line_color=utils.RED,
                                    ))
            self.fig.add_trace(Line(
                y=senkou_span_b,
                fill='tonexty',
                line_color=utils.ICHIMOKU_CLOUD_COLOR))

            self.returns = [utils.EXIT for i in range(chinkouspan)]
            self._stop_losses = [np.inf] * chinkouspan
            for e, (close, tenkan, kijun, A, B) in enumerate(zip(
                    prices.values[chinkouspan:],
                    tenkan_sen[chinkouspan:],
                    kinjun_sen[chinkouspan:],
                    senkou_span_a[chinkouspan:],
                    senkou_span_b[chinkouspan:],
            ), chinkouspan):
                max_cloud = max((A, B))
                min_cloud = min((A, B))

                if not min_cloud < close < max_cloud:
                    if tenkan > kijun:
                        flag1 = utils.BUY
                    elif tenkan < kijun:
                        flag1 = utils.SELL

                    if close > max_cloud:
                        flag2 = utils.BUY
                    elif close < min_cloud:
                        flag2 = utils.SELL

                    if close > prices[e - chinkouspan]:
                        flag3 = utils.BUY
                    elif close < prices[e - chinkouspan]:
                        flag3 = utils.SELL

                    if flag3 == flag1 == flag2:
                        trade = flag1
                    if (trade == utils.BUY and flag1 == utils.SELL) or (trade == utils.SELL and flag1 == utils.BUY):
                        trade = utils.EXIT
                self.returns.append(trade)

        self.set_open_stop_and_take(set_take=True,
                                    set_stop=False)
        self.set_credit_leverages()
        self.sl_tp_adder(add_stop_loss=stop_loss_plus)
        return self.returns

    def crossover(self, fast: Iterable, slow: Iterable):
        self.returns = []
        for s, f in zip(slow, fast):
            if s < f:
                self.returns.append(utils.BUY)
            elif s > f:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def inverse_strategy(self, swap_tpop_take: bool = True) -> utils.PREDICT_TYPE_LIST:
        """
        makes signals inverse:
        buy = sell.
        sell = buy.
        exit = exit.
        """

        returns = []
        flag: utils.PREDICT_TYPE = utils.EXIT
        for signal_key in self.returns:
            if signal_key == utils.BUY:
                flag = utils.SELL
            elif signal_key == utils.SELL:
                flag = utils.BUY
            elif signal_key == utils.EXIT:
                flag = utils.EXIT
            returns.append(flag)
        self.returns = returns
        if swap_tpop_take:
            self._stop_losses, self._take_profits = self._take_profits, self._stop_losses
        return self.returns

    def backtest(self,
                 deposit: float = 10_000.0,
                 bet: float = np.inf,
                 commission: float = 0.0,
                 plot: bool = True,
                 print_out: bool = True,
                 show: bool = True) -> pd.DataFrame:
        """
        testing the strategy.
        :param deposit: start deposit.
        :param bet: fixed bet to quick_trade. np.inf = all moneys.
        :param commission: percentage commission (0 -- 100).
        :param plot: plotting.
        :param print_out: printing.
        :param show: show the graph
        returns: pd.DataFrame with data of test
        """

        exit_take_stop: bool
        no_order: bool
        stop_loss: float
        take_profit: float
        converted_element: utils.CONVERTED_TYPE
        diff: float
        lin_calc_df: pd.DataFrame
        high: float
        low: float
        credit_lev: float

        start_bet: float = bet
        data_column: pd.Series = self.df['Close']
        data_high: pd.Series = self.df['High']
        data_low: pd.Series = self.df['Low']
        self.deposit_history = [deposit]
        converted: utils.CONVERTED_TYPE_LIST = utils.convert(self.returns)
        self.trades = 0
        self.profits = 0
        self.losses = 0
        moneys_open_bet: float = deposit
        money_start: float = deposit
        oldsig = utils.EXIT

        e: int
        sig: utils.PREDICT_TYPE
        ignore_breakout: bool = False
        for e, (sig,
                stop_loss,
                take_profit,
                converted_element,
                credit_lev,
                high,
                low,
                next_h,
                next_l) in enumerate(zip(self.returns[:-1],
                                         self._stop_losses[:-1],
                                         self._take_profits[:-1],
                                         converted[:-1],
                                         self._credit_leverages[:-1],
                                         data_high[:-1],
                                         data_low[:-1],
                                         data_high[1:],
                                         data_low[1:])):

            if converted_element is not np.nan:
                if oldsig != utils.EXIT:
                    commission_reuse = 2
                else:
                    commission_reuse = 1
                bet = start_bet
                if bet > deposit:
                    bet = deposit
                open_price = data_column[e]
                for i in range(commission_reuse):
                    deposit -= bet * (commission / 100) * credit_lev
                    if bet > deposit:
                        bet = deposit
                moneys_open_bet = deposit
                no_order = False
                exit_take_stop = False
                ignore_breakout = True

            next_not_breakout = min(stop_loss, take_profit) < next_l <= next_h < max(stop_loss, take_profit)
            if (min(stop_loss, take_profit) < low <= high < max(stop_loss,
                                                                take_profit) and next_not_breakout) or ignore_breakout:
                diff = data_column[e + 1] - data_column[e]
            else:
                exit_take_stop = True
                if sig == utils.BUY and high >= take_profit:
                    diff = take_profit - data_column[e]

                elif sig == utils.BUY and low <= stop_loss:
                    diff = stop_loss - data_column[e]

                elif sig == utils.SELL and high >= stop_loss:
                    diff = stop_loss - data_column[e]

                elif sig == utils.SELL and low <= take_profit:
                    diff = take_profit - data_column[e]

                else:
                    diff = 0.0

            if sig == utils.SELL:
                diff = -diff
            elif sig == utils.EXIT:
                diff = 0.0
            if not no_order:
                deposit += bet * credit_lev * diff / open_price
            no_order = exit_take_stop
            self.deposit_history.append(deposit)
            oldsig = sig
            if converted_element is not np.nan:
                if sig != utils.EXIT:
                    self.trades += 1
                if oldsig != utils.EXIT:
                    if deposit > moneys_open_bet:
                        self.profits += 1
                    elif deposit < moneys_open_bet:
                        self.losses += 1
            ignore_breakout = False

        self.average_growth = utils.get_exponential_growth(self.deposit_history)
        self.year_profit = utils.profit_factor(self.deposit_history) ** (self._profit_calculate_coef - 1)
        #  Compound interest. View https://www.investopedia.com/terms/c/compoundinterest.asp
        self.year_profit -= 1  # The initial deposit does not count as profit
        self.year_profit *= 100  # Percentage
        if self.trades != 0:
            self.winrate = (self.profits / self.trades) * 100
        else:
            self.winrate = 0
        self._info = f"""losses: {self.losses}
trades: {self.trades}
profits: {self.profits}
mean year percentage profit: {self.year_profit}%
winrate: {self.winrate}%"""
        utils.logger.info(f'trader info: {self._info}')
        if print_out:
            print(self._info)
        self.returns_strategy_diff = list(pd.Series(self.deposit_history).diff().values)
        self.returns_strategy_diff[0] = 0
        self._backtest_out_no_drop = pd.DataFrame(
            (self.deposit_history, self._stop_losses, self._take_profits, self.returns,
             self._open_lot_prices, data_column, self.average_growth, self.returns_strategy_diff),
            index=[
                f'deposit', 'stop loss', 'take profit',
                'predictions', 'open trade', 'Close',
                f"average growth deposit data",
                "returns"
            ]).T
        self.backtest_out = self._backtest_out_no_drop.dropna()
        if plot:
            loc: pd.Series = self.df['Close']
            self.fig.add_trace(
                Line(
                    y=self._backtest_out_no_drop['returns'].values,
                    line=dict(color=utils.COLOR_DEPOSIT),
                    name='returns'
                ),
                row=3,
                col=1
            )
            self.fig.add_candlestick(
                close=self.df['Close'],
                high=self.df['High'],
                low=self.df['Low'],
                open=self.df['Open'],
                row=1,
                col=1,
                name=f'{self.ticker} {self.interval}')
            self.fig.add_trace(
                Line(
                    y=self._take_profits,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.GREEN),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='take profit'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self._stop_losses,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.RED),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='stop loss'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self._open_lot_prices,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.BLUE),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='open trade'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.deposit_history,
                    line=dict(color=utils.COLOR_DEPOSIT),
                    name=f'deposit (start: {money_start})'), 2, 1)
            self.fig.add_trace(Line(y=self.average_growth, name='average growth'), 2, 1)
            preds: Dict[str, List[Union[int, float]]] = {'sellind': [],
                                                         'exitind': [],
                                                         'buyind': [],
                                                         'bprice': [],
                                                         'sprice': [],
                                                         'eprice': []}
            for e, i in enumerate(converted):
                if i == utils.SELL:
                    preds['sellind'].append(e)
                    preds['sprice'].append(loc[e])
                elif i == utils.BUY:
                    preds['buyind'].append(e)
                    preds['bprice'].append(loc[e])
                elif i == utils.EXIT:
                    preds['exitind'].append(e)
                    preds['eprice'].append(loc[e])
            name: str
            index: int
            price: float
            for name, index, price, triangle_type, color in zip(
                    ['Buy', 'Sell', 'Exit'],
                    [preds['buyind'], preds['sellind'], preds['exitind']],
                    [preds['bprice'], preds['sprice'], preds['eprice']],
                    ['triangle-up', 'triangle-down', 'triangle-left'],
                    [utils.GREEN, utils.RED, utils.BLUE]
            ):
                self.fig.add_scatter(
                    mode='markers',
                    name=name,
                    y=price,
                    x=index,
                    row=1,
                    col=1,
                    line=dict(color=color),
                    marker=dict(
                        symbol=triangle_type,
                        size=utils.SCATTER_SIZE,
                        opacity=utils.SCATTER_ALPHA))
            if show:
                self.fig.show()

        return self.backtest_out

    def multi_backtest(self,
                       tickers: Union[Sized, Iterable[str]],
                       strategy_name: str = 'strategy_macd',
                       strategy_kwargs: Dict[str, Any] = {},
                       deposit: float = 10_000.0,
                       bet: float = np.inf,
                       commission: float = 0.0,
                       plot: bool = True,
                       print_out: bool = True,
                       show: bool = True,
                       limit: int = 1000) -> pd.DataFrame:
        winrates: List[float] = []
        percentage_profits: List[float] = []
        losses: List[int] = []
        trades: List[int] = []
        profits: List[int] = []
        depo: List[np.ndarray] = []
        lens_dep: List[int] = []

        for ticker in tickers:
            df = self.client.get_data_historical(ticker=ticker, limit=limit, interval=self.interval)
            new_trader = self._get_this_instance(interval=self.interval, df=df, ticker=ticker)
            new_trader.set_client(your_client=self.client)
            new_trader.set_pyplot()
            new_trader._get_attr(strategy_name)(**strategy_kwargs)
            new_trader.backtest(deposit=deposit / len(tickers),
                                bet=bet,
                                commission=commission,
                                plot=False,
                                print_out=False,
                                show=False)
            winrates.append(new_trader.winrate)
            percentage_profits.append(new_trader.year_profit)
            losses.append(new_trader.losses)
            trades.append(new_trader.trades)
            profits.append(new_trader.profits)
            depo.append(np.array(new_trader.deposit_history))
            lens_dep.append(len(new_trader.deposit_history))
        self.losses = sum(losses)
        self.trades = sum(profits)
        self.profits = sum(profits)
        self.year_profit = float(np.mean(percentage_profits))
        self.winrate = float(np.mean(winrates))

        for enum, elem in enumerate(depo):
            depo[enum] = np.array(elem[-min(lens_dep):]) / (elem[-min(lens_dep)] / (deposit / len(tickers)))
        self.deposit_history = list(sum(depo))

        self.average_growth = utils.get_exponential_growth(self.deposit_history)
        self.returns_strategy_diff = list(pd.Series(self.deposit_history).diff().values)
        self.returns_strategy_diff[0] = 0
        self._backtest_out_no_drop = pd.DataFrame(
            (self.deposit_history, self.average_growth, self.returns_strategy_diff),
            index=[
                f'deposit',
                f"average growth deposit data",
                "returns"
            ]).T
        self.backtest_out = self._backtest_out_no_drop.dropna()

        self._info = f"""losses: {self.losses}
trades: {self.trades}
profits: {self.profits}
mean year percentage profit: {self.year_profit}%
winrate: {self.winrate}%"""
        utils.logger.info(f'trader multi info: {self._info}')
        if print_out:
            print(self._info)
        if plot:
            self.fig.add_trace(
                Line(
                    y=self.deposit_history,
                    line=dict(color=utils.COLOR_DEPOSIT),
                    name=f'deposit (start: {deposit})'), 2, 1)
            self.fig.add_trace(Line(y=self.average_growth, name='average growth'), 2, 1)
            self.fig.add_trace(
                Line(
                    y=self.returns_strategy_diff,
                    line=dict(color=utils.COLOR_DEPOSIT),
                    name='returns'
                ),
                row=3,
                col=1
            )
        if show:
            self.fig.show()
        return self.backtest_out

    def set_pyplot(self,
                   height: int = 900,
                   width: int = 1300,
                   template: str = 'plotly_dark',
                   row_heights: list = [10, 16, 7],
                   **subplot_kwargs):
        """

        :param height: window height
        :param width: window width
        :param template: plotly template
        :param row_heights: standard
        """
        self.fig = make_subplots(3, 1, row_heights=row_heights, **subplot_kwargs)
        self.fig.update_layout(
            height=height,
            width=width,
            template=template,
            xaxis_rangeslider_visible=False)
        self.fig.update_xaxes(
            title_text='T I M E', row=3, col=1, color=utils.TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='M O N E Y S', row=2, col=1, color=utils.TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='R E T U R N S', row=3, col=1, color=utils.TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='D A T A', row=1, col=1, color=utils.TEXT_COLOR)

    def strategy_collider(self,
                          first_returns: utils.PREDICT_TYPE_LIST,
                          second_returns: utils.PREDICT_TYPE_LIST,
                          mode: str = 'minimalist') -> utils.PREDICT_TYPE_LIST:
        """
        :param second_returns: returns of strategy
        :param first_returns: returns of strategy
        :param mode:  mode of combining:

            example :
                mode = 'minimalist':
                    1,1 = 1

                    0,0 = 0

                    2,2 = 2

                    0,1 = 2

                    1,0 = 2

                    2,1 = 2

                    1,2 = 2

                    ...

                    first_returns = [1,1,0,0,2,0,2,2,0,0,1]

                    second_returns = [1,2,2,2,2,2,0,0,0,0,1]

                        [1,2,2,2,2,2,2,2,0,0,1]

                mode = 'maximalist':
                    1,1 = 1

                    0,0 = 0

                    2,2 = 2

                    0,1 = last sig

                    1,0 = last sig

                    2,1 = last sig

                    1,2 = last sig

                    ...

                    first_returns =  [1,1,0,0,2,0,2,2,0,0,1]

                    second_returns = [1,2,2,2,2,2,0,0,0,0,1]

                        [1,1,1,1,2,2,2,2,0,0,1]

                mode = 'super':
                    ...

                    first_returns =  [1,1,1,2,2,2,0,0,1]

                    second_returns = [1,0,0,0,1,1,1,0,0]

                        [1,0,0,2,1,1,0,0,1]

        :return: combining of 2 strategies
        """

        if mode == 'minimalist':
            self.returns = []
            for ret1, ret2 in zip(first_returns, second_returns):
                if ret1 == ret2:
                    self.returns.append(ret1)
                else:
                    self.returns.append(utils.EXIT)
        elif mode == 'maximalist':
            self.returns = self._maximalist(first_returns, second_returns)
        elif mode == 'super':
            self.returns = self._collide_super(first_returns, second_returns)
        else:
            raise ValueError('incorrect mode')
        return self.returns

    @staticmethod
    def _maximalist(returns1: utils.PREDICT_TYPE_LIST,
                    returns2: utils.PREDICT_TYPE_LIST) -> utils.PREDICT_TYPE_LIST:
        return_list: utils.PREDICT_TYPE_LIST = []
        flag = utils.EXIT
        for a, b in zip(returns1, returns2):
            if a == b:
                return_list.append(a)
                flag = a
            else:
                return_list.append(flag)
        return return_list

    @staticmethod
    def _collide_super(l1, l2) -> utils.PREDICT_TYPE_LIST:
        return_list: utils.PREDICT_TYPE_LIST = []
        for first, sec in zip(utils.convert(l1), utils.convert(l2)):
            if first is not np.nan and sec is not np.nan and first is not sec:
                return_list.append(utils.EXIT)
            elif first is sec:
                return_list.append(first)
            elif first is np.nan:
                return_list.append(sec)
            else:
                return_list.append(first)
        return list(map(lambda x: utils.PREDICT_TYPE(x), utils.anti_convert(return_list)))

    def multi_strategy_collider(self, *strategies, mode: str = 'minimalist') -> utils.PREDICT_TYPE_LIST:
        self.strategy_collider(strategies[0], strategies[1], mode=mode)
        if len(strategies) >= 3:
            for ret in strategies[2:]:
                self.strategy_collider(self.returns, ret, mode=mode)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        return self.returns

    def get_trading_predict(self,
                            bet_for_trading_on_client: float = np.inf,
                            coin_lotsize_division: bool = True
                            ) -> Dict[str, Union[str, float]]:
        """
        predict and trading.

        :param coin_lotsize_division: If for your api you specify the size of the bet in a coin, which is not in which you have a deposit, specify this parameter in the value: True. Otherwise: False, in Binance's case this is definitely the first case (True). If errors occur, try specifying the first ticker symbol instead of the second.
        :param bet_for_trading_on_client: standard: all deposit
        :return: dict with prediction
        """

        _moneys_: float
        bet: float
        close: np.ndarray = self.df["Close"].values

        # get prediction
        predict = utils.convert_signal_str(self.returns[-1])

        # trading
        self.__last_stop_loss = self._stop_losses[-1]
        self.__last_take_profit = self._take_profits[-1]
        self.__last_credit_leverage = self._credit_leverages[-1]
        if self._prev_predict != predict or self.__prev_credit_lev != self.__last_credit_leverage:
            utils.logger.info(f'open trade {predict}')
            self.__exit_order__ = False
            if self.trading_on_client:

                if predict == 'Exit':
                    self.client.exit_last_order()
                    self.__exit_order__ = True

                else:
                    _moneys_ = self.client.get_balance_ticker(self.ticker.split('/')[1])
                    ticker_price = self.client.get_ticker_price(self.ticker)
                    if bet_for_trading_on_client is not np.inf:
                        bet = bet_for_trading_on_client
                    else:
                        bet = _moneys_
                    if bet > _moneys_:
                        bet = _moneys_
                    if coin_lotsize_division:
                        bet /= ticker_price
                    self.client.exit_last_order()

                    self.client.order_create(predict,
                                             self.ticker,
                                             bet * self.__last_credit_leverage)
        utils.logger.debug("returning prediction")
        return {
            'predict': predict,
            'open trade price': self._open_price,
            'stop loss': self.__last_stop_loss,
            'take profit': self.__last_take_profit,
            'currency close': close[-1],
            'credit leverage': self.__last_credit_leverage
        }

    def realtime_trading(self,
                         strategy,
                         ticker: str = 'BTC/USDT',
                         print_out: bool = True,
                         bet_for_trading_on_client: float = np.inf,
                         coin_lotsize_division: bool = True,
                         ignore_exceptions: bool = False,
                         print_exc: bool = True,
                         wait_sl_tp_checking: float = 5,
                         limit: int = 1000,
                         strategy_in_sleep: bool = False,
                         *strategy_args,
                         **strategy_kwargs):
        """
        :param strategy_in_sleep: reuse strategy in one candle for new S/L, T/P or martingale
        :param limit: client.get_data_historical's limit argument
        :param wait_sl_tp_checking: sleeping time after stop-loss and take-profit checking (seconds)
        :param print_exc: print  exceptions in while loop
        :param ignore_exceptions: ignore binance exceptions in while loop
        :param coin_lotsize_division: If for your api you specify the size of the bet in a coin, which is not in which you have a deposit, specify this parameter in the value: True. Otherwise: False, in Binance's case this is definitely the first case (True). If errors occur, try specifying the first ticker symbol instead of the second.
        :param ticker: ticker for trading.
        :param strategy: trading strategy.
        :param print_out: printing.
        :param bet_for_trading_on_client: trading bet, standard: all deposit
        :param strategy_kwargs: named arguments to -strategy.
        :param strategy_args: arguments to -strategy.
        """

        self.realtime_returns = {}
        self.ticker = ticker
        open_time = time()
        while True:
            try:
                self.df = self.client.get_data_historical(ticker=self.ticker, limit=limit, interval=self.interval)
                utils.logger.debug("new dataframe loaded")

                strategy(*strategy_args, **strategy_kwargs)
                utils.logger.debug("strategy used")

                prediction = self.get_trading_predict(
                    bet_for_trading_on_client=bet_for_trading_on_client,
                    coin_lotsize_division=coin_lotsize_division)

                index = f'{self.ticker}, {ctime()}'
                utils.logger.info(f"trading prediction at {index}: {prediction}")
                if print_out:
                    print(index, prediction)
                self.realtime_returns[index] = prediction
                while True:
                    if not self.__exit_order__:
                        if (open_time + self._sec_interval) - time() < wait_sl_tp_checking:
                            sleep(wait_sl_tp_checking)
                        utils.logger.info(f"sleep {wait_sl_tp_checking} seconds")

                        price = self.client.get_ticker_price(ticker)
                        min_ = min(self.__last_stop_loss, self.__last_take_profit)
                        max_ = max(self.__last_stop_loss, self.__last_take_profit)
                        if (not (min_ < price < max_)) and prediction["predict"] != 'Exit':
                            self.__exit_order__ = True
                            utils.logger.info('exit trade')
                            index = f'{self.ticker}, {ctime()}'
                            utils.logger.info(f"trading prediction exit in sleeping at {index}: {prediction}")
                            if print_out:
                                print(f"trading prediction exit in sleeping at {index}: {prediction}")
                            self.realtime_returns[index] = prediction
                            if self.trading_on_client:
                                self.client.exit_last_order()
                        elif strategy_in_sleep:
                            break
                    if time() >= (open_time + self._sec_interval):
                        self._prev_predict = utils.convert_signal_str(self.returns[-1])
                        open_time += self._sec_interval
                        break
            except Exception as exc:
                utils.logger.error(f'An error occurred: {exc}', exc_info=True)
                self.client.exit_last_order()
                if ignore_exceptions:
                    if print_exc:
                        print(exc)
                    continue
                else:
                    raise exc

    def multi_realtime_trading(self,
                               tickers: List[str],
                               start_time: datetime,  # LOCAL TIME
                               strategy_name: str,
                               print_out: bool = True,
                               bet_for_trading_on_client: float = np.inf,  # for 1 trade
                               coin_lotsize_division: bool = True,
                               ignore_exceptions: bool = False,
                               print_exc: bool = True,
                               wait_sl_tp_checking: float = 5,
                               limit: int = 1000,
                               strategy_in_sleep: bool = False,
                               deposit_part: float = 1.0,  # for all trades
                               **strategy_kwargs):
        can_orders: int = len(tickers)
        bet_for_trading_on_client_copy: float = bet_for_trading_on_client

        class MultiRealTimeTrader(self.__class__):
            def get_trading_predict(self,
                                    bet_for_trading_on_client: float = np.inf,
                                    coin_lotsize_division: bool = True
                                    ) -> Dict[str, Union[str, float]]:
                balance = self.client.get_balance_ticker(self.ticker.split('/')[1])
                bet = (balance * 10) / (can_orders / deposit_part - brokers.TradingClient.cls_open_orders)
                bet /= 10  # decimal analog
                if bet > bet_for_trading_on_client_copy:
                    bet = bet_for_trading_on_client_copy
                return super().get_trading_predict(bet_for_trading_on_client=bet,
                                                   coin_lotsize_division=coin_lotsize_division)

        def start_trading(pare):
            trader = MultiRealTimeTrader(ticker=pare,
                                         interval=self.interval,
                                         trading_on_client=self.trading_on_client)
            trader.set_pyplot()
            trader.set_client(copy(self.client))

            while True:
                if datetime.now() >= start_time:
                    break
            trader.realtime_trading(strategy=trader._get_attr(strategy_name),
                                    ticker=pare,
                                    print_out=print_out,
                                    coin_lotsize_division=coin_lotsize_division,
                                    ignore_exceptions=ignore_exceptions,
                                    print_exc=print_exc,
                                    wait_sl_tp_checking=wait_sl_tp_checking,
                                    limit=limit,
                                    strategy_in_sleep=strategy_in_sleep,
                                    **strategy_kwargs)

        for ticker in tickers:
            thread = Thread(target=start_trading, args=(ticker,))
            thread.start()

    def log_data(self):
        self.fig.update_yaxes(row=1, col=1, type='log')
        utils.logger.debug('trader log data')

    def log_deposit(self):
        self.fig.update_yaxes(row=2, col=1, type='log')
        utils.logger.debug('trader log deposit')

    def log_returns(self):
        self.fig.update_yaxes(row=3, col=1, type='log')
        utils.logger.debug('trader log returns')

    def set_client(self, your_client: brokers.TradingClient):
        """
        :param your_client: trading client
        """
        self.client = your_client
        utils.logger.debug('trader set client')

    def convert_signal(self,
                       old: utils.PREDICT_TYPE = utils.SELL,
                       new: utils.PREDICT_TYPE = utils.EXIT) -> utils.PREDICT_TYPE_LIST:
        pos: int
        val: utils.PREDICT_TYPE
        for pos, val in enumerate(self.returns):
            if val == old:
                self.returns[pos] = new
        utils.logger.debug(f'trader signals converted: {old} >> {new}')
        return self.returns

    def set_open_stop_and_take(self,
                               take_profit: float = np.inf,
                               stop_loss: float = np.inf,
                               set_stop: bool = True,
                               set_take: bool = True):
        """
        :param set_take: create new take profits.
        :param set_stop: create new stop losses.
        :param take_profit: take profit in points
        :param stop_loss: stop loss in points
        """
        self._take_profit = take_profit
        self._stop_loss = stop_loss
        take_flag: float = np.inf
        stop_flag: float = np.inf
        self._open_lot_prices = []
        if set_stop:
            self._stop_losses = []
        if set_take:
            self._take_profits = []
        closes: np.ndarray = self.df['Close'].values
        sig: utils.PREDICT_TYPE
        close: float
        seted: utils.CONVERTED_TYPE
        ts: Dict[str, float]
        for sig, close, seted in zip(self.returns, closes, utils.convert(self.returns)):
            if seted is not np.nan:
                self._open_price = close
                if set_take or set_stop:
                    ts = self.__get_stop_take(sig)
                if set_take:
                    take_flag = ts['take']
                if set_stop:
                    stop_flag = ts['stop']
            self._open_lot_prices.append(self._open_price)
            if set_take:
                self._take_profits.append(take_flag)
            if set_stop:
                self._stop_losses.append(stop_flag)
        utils.logger.debug(f'trader stop loss: {stop_loss}, trader take profit: {take_profit}')

    def set_credit_leverages(self, credit_lev: float = 1.0):
        """
        Sets the leverage for bets.
        :param credit_lev: leverage in points
        """
        self.__prev_credit_lev = credit_lev
        self._credit_leverages = [credit_lev for i in range(len(self.df['Close']))]
        utils.logger.info(f'trader credit leverage: {credit_lev}')

    def _window_(self,
                 column: str,
                 n: int = 2,
                 *args,
                 **kwargs) -> List[Any]:
        return utils.get_window(self.df[column].values, n)

    def find_pip_bar(self,
                     min_diff_coef: float = 2.0,
                     body_coef: float = 10.0) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        flag = utils.EXIT
        e: int
        high: float
        low: float
        open_price: float
        close: float

        body: float
        shadow_high: float
        shadow_low: float
        for e, (high, low, open_price, close) in enumerate(
                zip(self.df['High'], self.df['Low'], self.df['Open'],
                    self.df['Close']), 1):
            body = abs(open_price - close)
            shadow_high = high - max(open_price, close)
            shadow_low = min(open_price, close) - low
            if body < (max(shadow_high, shadow_low) * body_coef):
                if shadow_low > (shadow_high * min_diff_coef):
                    flag = utils.BUY
                elif shadow_high > (shadow_low * min_diff_coef):
                    flag = utils.SELL
                self.returns.append(flag)
            else:
                self.returns.append(flag)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def find_DBLHC_DBHLC(self) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.EXIT]
        flag: utils.PREDICT_TYPE = utils.EXIT

        flag_stop_loss: float = np.inf
        self._stop_losses = [flag_stop_loss]
        high: List[float]
        low: List[float]
        open_pr: List[float]
        close: List[float]

        for high, low, open_pr, close in zip(
                self._window_('High'),
                self._window_('Low'),
                self._window_('Open'),
                self._window_('Close')
        ):
            if low[0] == low[1] and close[1] > high[0]:
                flag = utils.BUY
                flag_stop_loss = min(low[0], low[1])
            elif high[0] == high[1] and close[0] > low[1]:
                flag = utils.SELL
                flag_stop_loss = max(high[0], high[1])

            self.returns.append(flag)
            self._stop_losses.append(flag_stop_loss)
        self.set_credit_leverages()
        self.set_open_stop_and_take(set_stop=False)
        return self.returns

    def find_TBH_TBL(self) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.EXIT]
        flag: utils.PREDICT_TYPE = utils.EXIT
        high: List[float]
        low: List[float]
        open_: List[float]
        close: List[float]

        for e, (high, low, open_, close) in enumerate(
                zip(
                    self._window_('High'), self._window_('Low'),
                    self._window_('Open'), self._window_('Close')
                ), 1):
            if high[0] == high[1]:
                flag = utils.BUY
            elif low[0] == low[1]:
                flag = utils.SELL
            self.returns.append(flag)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def find_PPR(self) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.EXIT] * 2
        flag: utils.PREDICT_TYPE = utils.EXIT
        high: List[float]
        low: List[float]
        opn: List[float]
        close: List[float]
        for e, (high, low, opn, close) in enumerate(
                zip(
                    self._window_('High', 3), self._window_('Low', 3),
                    self._window_('Open', 3), self._window_('Close', 3)), 1):
            if min(low) == low[1] and close[1] < close[2] and high[2] < high[0]:
                flag = utils.BUY
            elif max(high
                     ) == high[1] and close[2] < close[1] and low[2] > low[0]:
                flag = utils.SELL
            self.returns.append(flag)
        self.set_credit_leverages()
        self.set_open_stop_and_take()
        return self.returns

    def get_support_resistance(self) -> Dict[str, Dict[int, float]]:
        lows = self.df['Low'].values
        highs = self.df['High'].values
        for i in range(2, len(lows) - 2):
            if lows[i - 2] >= lows[i - 1] >= lows[i] <= lows[i + 1] <= lows[i + 2]:
                self.supports[i] = lows[i]
            if highs[i - 2] <= highs[i - 1] <= highs[i] >= highs[i + 1] >= highs[i + 2]:
                self.resistances[i] = highs[i]
        return {'resistance': self.resistances,
                'supports': self.supports}
