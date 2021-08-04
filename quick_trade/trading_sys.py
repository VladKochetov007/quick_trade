# !/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)

# TODO:
#   scalper and dca bot
#   more docs and examples
#   decimal
#   3.9
#   decorator for strategies without exit condition (not converted data)
#   multi-backtest normal calculating(real multi-test, not sum of single tests)
#   add meta-data in tuner's returns

from copy import copy
from datetime import datetime
from re import fullmatch
from threading import Thread
from time import ctime
from time import sleep
from time import time
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sized
from typing import Tuple
from typing import Union
from warnings import warn

import ta
from numpy import array
from numpy import digitize
from numpy import inf
from numpy import mean
from numpy import nan
from numpy import nan_to_num
from numpy import ndarray
from pandas import DataFrame
from pandas import Series

from . import utils
from .brokers import TradingClient
from .plots import QuickTradeGraph


class Trader(object):
    """
    algo-trading main class.
    ticker:   |     str      |  ticker/symbol of chart
    df:       |   dataframe  |  data of chart
    interval: |     str      |  interval of df.

    """
    _profit_calculate_coef: Union[float, int]
    returns: utils.PREDICT_TYPE_LIST = []
    df: DataFrame
    ticker: str
    interval: str
    __exit_order__: bool = False
    _prev_predict: str = 'Exit'
    _stop_loss: Union[float, int]
    _take_profit: Union[float, int]
    _open_price: float
    trades: int = 0
    profits: int = 0
    losses: int = 0
    _stop_losses: List[float]
    _take_profits: List[float]
    _credit_leverages: List[Union[float, int]]
    deposit_history: List[Union[float, int]]
    year_profit: float
    average_growth: ndarray
    _info: str
    _backtest_out_no_drop: DataFrame
    backtest_out: DataFrame
    _open_lot_prices: List[float]
    client: TradingClient
    __last_stop_loss: float
    __last_take_profit: float
    returns_strategy_diff: List[float]
    _sec_interval: int
    supports: Dict[int, float]
    resistances: Dict[int, float]
    trading_on_client: bool
    fig: QuickTradeGraph

    @property
    def _converted(self) -> utils.CONVERTED_TYPE_LIST:
        return utils.convert(self.returns)

    @utils.assert_logger
    def __init__(self,
                 ticker: str = 'BTC/USDT',
                 df: DataFrame = DataFrame(),
                 interval: str = '1d',
                 trading_on_client: bool = True):
        ticker = ticker.upper()
        assert isinstance(ticker, str), 'The ticker can only be of type <str>.'
        assert fullmatch(utils.TICKER_PATTERN, ticker), f'Ticker must match the pattern <{utils.TICKER_PATTERN}>'
        assert isinstance(df, DataFrame), 'Dataframe can only be of type <DataFrame>.'
        assert isinstance(interval, str), 'interval can only be of the <str> type.'
        assert isinstance(trading_on_client, bool), 'trading_on_client can only be True or False (<bool>).'

        self.df = df.reset_index(drop=True)
        self.ticker = ticker
        self.interval = interval
        self.trading_on_client = trading_on_client
        self._profit_calculate_coef, self._sec_interval = utils.get_coef_sec(interval)
        self.__exit_order__ = False
        utils.logger.info('new trader: %s', self)

    def __repr__(self):
        return f'{self.ticker} {self.interval} trader. Trading: {self.trading_on_client}'

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
        if self._stop_loss is not inf:
            _stop_loss = self._stop_loss / 10_000 * self._open_price
        else:
            _stop_loss = inf
        if self._take_profit is not inf:
            take = self._take_profit / 10_000 * self._open_price
        else:
            take = inf

        if sig == utils.BUY:
            _stop_loss = self._open_price - _stop_loss
            take = self._open_price + take
        elif sig == utils.SELL:
            take = self._open_price - take
            _stop_loss = self._open_price + _stop_loss
        else:
            if self._take_profit is not inf:
                take = self._open_price
            if self._stop_loss is not inf:
                _stop_loss = self._open_price

        return {'stop': _stop_loss,
                'take': take}

    @utils.assert_logger
    def sl_tp_adder(self, add_stop_loss: Union[float, int] = 0.0, add_take_profit: Union[float, int] = 0.0) -> Tuple[
        List[float], List[float]]:
        """

        :param add_stop_loss: add stop loss points
        :param add_take_profit: add take profit points
        :return: (stop losses, take profits)
        """
        assert isinstance(add_stop_loss, (int, float)) and isinstance(add_take_profit, (int, float)), \
            'Arguments to this function can only be <float> or <int>.'

        utils.logger.debug('add stop-loss: %f pips, take-profit: %s pips', add_stop_loss, add_take_profit)
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

    def get_heikin_ashi(self, df: DataFrame = DataFrame()) -> DataFrame:
        """
        :param df: dataframe, standard: self.df
        :return: heikin ashi
        """
        if 'Close' not in df.columns:
            df: DataFrame = self.df.copy()
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

    @utils.assert_logger
    def crossover(self, fast: Iterable, slow: Iterable):
        assert isinstance(fast, Iterable) and isinstance(slow, Iterable), \
            'The arguments to this function must be iterable.'

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

    @utils.assert_logger
    def inverse_strategy(self, swap_stop_take: bool = True) -> utils.PREDICT_TYPE_LIST:
        """
        makes signals inverse:
        buy = sell.
        sell = buy.
        exit = exit.
        """
        assert isinstance(swap_stop_take, bool), 'swap_stop_take can only be <bool>'

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
        if swap_stop_take:
            self._stop_losses, self._take_profits = self._take_profits, self._stop_losses
        return self.returns

    @utils.assert_logger
    def backtest(self,
                 deposit: Union[float, int] = 10_000.0,
                 bet: Union[float, int] = inf,
                 commission: Union[float, int] = 0.0,
                 plot: bool = True,
                 print_out: bool = True,
                 show: bool = True) -> DataFrame:
        """
        testing the strategy.
        :param deposit: start deposit.
        :param bet: fixed bet to quick_trade. inf = all moneys.
        :param commission: percentage commission (0 -- 100).
        :param plot: plotting.
        :param print_out: printing.
        :param show: show the graph
        returns: DataFrame with data of test
        """
        assert isinstance(deposit, (float, int)), 'deposit must be of type <int> or <float>'
        assert deposit > 0, 'deposit can\'t be 0 or less'
        assert isinstance(bet, (float, int)), 'bet must be of type <int> or <float>'
        assert bet > 0, 'bet can\'t be 0 or less'
        assert isinstance(commission, (float, int)), 'commission must be of type <int> or <float>'
        assert 0 <= commission < 100, 'commission cannot be >=100% or less then 0'
        assert isinstance(plot, bool), 'plot must be of type <bool>'
        assert isinstance(print_out, bool), 'print_out must be of type <bool>'
        assert isinstance(show, bool), 'show must be of type <bool>'

        exit_take_stop: bool
        no_order: bool
        stop_loss: float
        take_profit: float
        converted_element: utils.CONVERTED_TYPE
        diff: float
        lin_calc_df: DataFrame
        high: float
        low: float
        credit_lev: Union[float, int]

        start_bet: Union[float, int] = bet
        data_column: Series = self.df['Close']
        data_high: Series = self.df['High']
        data_low: Series = self.df['Low']
        self.deposit_history = [deposit]
        self.trades = 0
        self.profits = 0
        self.losses = 0
        moneys_open_bet: Union[float, int] = deposit
        money_start: Union[float, int] = deposit
        prev_sig = utils.EXIT

        ignore_breakout: bool = False
        next_not_breakout: bool
        e: int
        sig: utils.PREDICT_TYPE
        stop_loss: float
        take_profit: float
        converted_element: utils.CONVERTED_TYPE
        credit_lev: Union[float, int]
        high: float
        low: float
        next_h: float
        next_l: float
        pass_math: bool = False
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
                                         self._converted[:-1],
                                         self._credit_leverages[:-1],
                                         data_high[:-1],
                                         data_low[:-1],
                                         data_high[1:],
                                         data_low[1:])):

            if converted_element is not nan:
                # count the number of profitable and unprofitable trades.
                if prev_sig != utils.EXIT:
                    self.trades += 1
                    if deposit > moneys_open_bet:
                        self.profits += 1
                    elif deposit < moneys_open_bet:
                        self.losses += 1

                # calculating commission
                if prev_sig != utils.EXIT:
                    commission_reuse = 2
                else:
                    commission_reuse = 1
                bet = start_bet
                if bet > deposit:
                    bet = deposit
                for i in range(commission_reuse):
                    deposit -= bet * (commission / 100) * credit_lev
                    if bet > deposit:
                        bet = deposit

                # reset service variables
                open_price = data_column[e]
                moneys_open_bet = deposit
                no_order = False
                exit_take_stop = False
                ignore_breakout = True
                if sig != utils.EXIT and not min(stop_loss, take_profit) <= open_price <= max(stop_loss, take_profit) and e > 0:
                    warn('The deal was opened out of range!')
                    utils.logger.error('The deal was opened out of range!')
                    self.winrate = 0.0
                    self.year_profit = 0.0
                    self.losses = 0
                    self.profits = 0
                    self.trades = 0
                    pass_math = True
                    break

            next_not_breakout = min(stop_loss, take_profit) < next_l <= next_h < max(stop_loss, take_profit)

            stop_loss = self._stop_losses[e - 1]
            take_profit = self._take_profits[e - 1]
            # be careful with e=0
            # haha))) no)
            now_not_breakout = min(stop_loss, take_profit) < low <= high < max(stop_loss, take_profit)
            if (ignore_breakout or now_not_breakout) and next_not_breakout:
                diff = data_column[e + 1] - data_column[e]
            else:
                # Here I am using the previous value,
                # because we do not know the value at this point
                # (it is generated only when the candle is closed).
                exit_take_stop = True

                if (not now_not_breakout) and not ignore_breakout:
                    stop_loss = self._stop_losses[e - 1]
                    take_profit = self._take_profits[e - 1]
                    diff = utils.get_diff(price=data_column[e],
                                          low=low,
                                          high=high,
                                          stop_loss=stop_loss,
                                          take_profit=take_profit,
                                          signal=sig)

                elif not next_not_breakout:
                    stop_loss = self._stop_losses[e]
                    take_profit = self._take_profits[e]
                    diff = utils.get_diff(price=data_column[e],
                                          low=next_l,
                                          high=next_h,
                                          stop_loss=stop_loss,
                                          take_profit=take_profit,
                                          signal=sig)
            if sig == utils.SELL:
                diff = -diff
            if sig == utils.EXIT:
                diff = 0.0
            if not no_order:
                deposit += bet * credit_lev * diff / open_price
            self.deposit_history.append(deposit)

            no_order = exit_take_stop
            prev_sig = sig
            ignore_breakout = False

        self.average_growth = utils.get_exponential_growth(self.deposit_history)
        if not pass_math:
            self.year_profit = utils.profit_factor(self.deposit_history) ** (self._profit_calculate_coef - 1)
            #  Compound interest. View https://www.investopedia.com/terms/c/compoundinterest.asp
            self.year_profit -= 1  # The initial deposit does not count as profit
            self.year_profit *= 100  # Percentage
            if self.trades != 0:
                self.winrate = (self.profits / self.trades) * 100
            else:
                self.winrate = 0
                utils.logger.critical('0 trades in %s', self)
        self._info = utils.INFO_TEXT.format(self.losses, self.trades, self.profits, self.year_profit, self.winrate)
        utils.logger.info('trader info: %s', self._info)
        if print_out:
            print(self._info)
        self.returns_strategy_diff = list(Series(self.deposit_history).diff().values)
        self.returns_strategy_diff[0] = 0
        self._backtest_out_no_drop = DataFrame(
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
            self.fig.plot_candlestick()
            self.fig.plot_trade_triangles()
            self.fig.plot_SL_TP_OPN()

            self.fig.plot_deposit()

            self.fig.plot_returns()
        if show:
            self.fig.show()

        return self.backtest_out

    @utils.assert_logger
    def multi_backtest(self,
                       tickers: Union[Sized, Iterable[str]],
                       strategy_name: str = 'strategy_macd',
                       strategy_kwargs: Dict[str, Any] = {},
                       limit: int = 1000,
                       deposit: Union[float, int] = 10_000.0,
                       bet: Union[float, int] = inf,
                       commission: Union[float, int] = 0.0,
                       plot: bool = True,
                       print_out: bool = True,
                       show: bool = True) -> DataFrame:
        assert isinstance(tickers, Iterable), 'tickers must be of type <Iterable[str]>'
        for el in tickers:
            assert isinstance(el, str), 'tickers must be of type <Iterable[str]>'
            assert fullmatch(utils.TICKER_PATTERN, el), f'all tickers must match the pattern <{utils.TICKER_PATTERN}>'
        assert isinstance(strategy_name, str), 'strategy_name must be of type <str>'
        assert strategy_name in self.__dir__(), 'There is no such strategy'
        assert isinstance(strategy_kwargs, dict)
        assert isinstance(limit, int), 'limit must be of type <int>'
        assert limit > 0, 'limit can\'t be 0 or less'
        assert isinstance(deposit, (float, int)), 'deposit must be of type <int> or <float>'
        assert deposit > 0, 'deposit can\'t be 0 or less'
        assert isinstance(bet, (float, int)), 'bet must be of type <int> or <float>'
        assert bet > 0, 'bet can\'t be 0 or less'
        assert isinstance(commission, (float, int)), 'commission must be of type <int> or <float>'
        assert 0 <= commission < 100, 'commission cannot be >=100% or less then 0'
        assert isinstance(plot, bool), 'plot must be of type <bool>'
        assert isinstance(print_out, bool), 'print_out must be of type <bool>'
        assert isinstance(show, bool), 'show must be of type <bool>'

        winrates: List[float] = []
        percentage_profits: List[float] = []
        losses: List[int] = []
        trades: List[int] = []
        profits: List[int] = []
        depo: List[ndarray] = []
        lens_dep: List[int] = []

        for ticker in tickers:
            df = self.client.get_data_historical(ticker=ticker, limit=limit, interval=self.interval)
            new_trader = self._get_this_instance(interval=self.interval, df=df, ticker=ticker)
            new_trader.set_client(your_client=self.client)
            try:
                new_trader.connect_graph(self.fig)
            except:
                pass
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
            depo.append(array(new_trader.deposit_history))
            lens_dep.append(len(new_trader.deposit_history))
        self.losses = sum(losses)
        self.trades = sum(profits)
        self.profits = sum(profits)
        self.year_profit = float(mean(percentage_profits))
        self.winrate = float(mean(winrates))

        for enum, elem in enumerate(depo):
            depo[enum] = array(elem[-min(lens_dep):]) / (elem[-min(lens_dep)] / (deposit / len(tickers)))
        self.deposit_history = list(sum(depo))

        self.average_growth = utils.get_exponential_growth(self.deposit_history)
        self.returns_strategy_diff = list(Series(self.deposit_history).diff().values)
        self.returns_strategy_diff[0] = 0
        self._backtest_out_no_drop = DataFrame(
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
        utils.logger.info('trader multi info: %s', self._info)
        if print_out:
            print(self._info)
        if plot:
            self.fig.plot_deposit()
            self.fig.plot_returns()
        if show:
            self.fig.show()
        return self.backtest_out

    @utils.assert_logger
    def connect_graph(self,
                      graph: QuickTradeGraph):
        """
        connect QuickTradeGraph
        """

        self.fig = graph
        self.fig.connect_trader(self)

    @utils.assert_logger
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
        assert isinstance(first_returns, utils.PREDICT_TYPE_LIST) and isinstance(second_returns,
                                                                                 utils.PREDICT_TYPE_LIST), \
            'Arguments to this function can only be <utils.PREDICT_TYPE>.'

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
            raise ValueError(f'incorrect mode: {mode}')
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
    def _collide_super(l1: utils.PREDICT_TYPE_LIST, l2: utils.PREDICT_TYPE_LIST) -> utils.PREDICT_TYPE_LIST:
        return_list: utils.PREDICT_TYPE_LIST = []
        for first, sec in zip(utils.convert(l1), utils.convert(l2)):
            if first is not nan and sec is not nan and first is not sec:
                return_list.append(utils.EXIT)
            elif first is sec:
                return_list.append(first)
            elif first is nan:
                return_list.append(sec)
            else:
                return_list.append(first)
        return list(map(lambda x: utils.PREDICT_TYPE(x), utils.anti_convert(return_list)))

    def multi_strategy_collider(self, *strategies, mode: str = 'minimalist') -> utils.PREDICT_TYPE_LIST:
        self.strategy_collider(strategies[0], strategies[1], mode=mode)
        if len(strategies) >= 3:
            for ret in strategies[2:]:
                self.strategy_collider(self.returns, ret, mode=mode)
        return self.returns

    def get_trading_predict(self,
                            bet_for_trading_on_client: Union[float, int] = inf,
                            coin_lotsize_division: bool = True
                            ) -> Dict[str, Union[str, float]]:
        """
        predict and trading.

        :param coin_lotsize_division: If for your api you specify the size of the bet in a coin, which is not in which you have a deposit, specify this parameter in the value: True. Otherwise: False, in Binance's case this is definitely the first case (True). If errors occur, try specifying the first ticker symbol instead of the second.
        :param bet_for_trading_on_client: standard: all deposit
        :return: dict with prediction
        """

        _moneys_: float
        bet: Union[float, int]
        close: ndarray = self.df["Close"].values

        # get prediction
        predict = utils.convert_signal_str(self.returns[-1])

        # trading
        self.__last_stop_loss = self._stop_losses[-1]
        self.__last_take_profit = self._take_profits[-1]

        conv_cred_lev = utils.convert(self._credit_leverages)

        if self._converted[-1] is not nan or conv_cred_lev[-1] is not nan:
            utils.logger.info('open trade %s', predict)
            self.__exit_order__ = False
            if self.trading_on_client:

                if predict == 'Exit':
                    self.client.exit_last_order()
                    self.__exit_order__ = True

                else:
                    _moneys_ = self.client.get_balance_ticker(self.ticker.split('/')[1])
                    ticker_price = self.client.get_ticker_price(self.ticker)
                    if bet_for_trading_on_client is not inf:
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
                                             bet * self._credit_leverages[-1])
        utils.logger.debug("returning prediction")
        return {
            'predict': predict,
            'open trade price': self._open_price,
            'stop loss': self.__last_stop_loss,
            'take profit': self.__last_take_profit,
            'currency close': close[-1],
            'credit leverage': self._credit_leverages[-1]
        }

    @utils.assert_logger
    def realtime_trading(self,
                         strategy,
                         ticker: str = 'BTC/USDT',
                         print_out: bool = True,
                         bet_for_trading_on_client: Union[float, int] = inf,
                         coin_lotsize_division: bool = True,
                         ignore_exceptions: bool = False,
                         print_exc: bool = True,
                         wait_sl_tp_checking: Union[float, int] = 5,
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
        assert fullmatch(utils.TICKER_PATTERN, ticker), f'Ticker must match the pattern <{utils.TICKER_PATTERN}>'
        assert isinstance(print_out, bool), 'print_out must be of type <bool>'
        assert isinstance(bet_for_trading_on_client,
                          (float, int)), 'bet_for_trading_on_client must be of type <float> or <int>'
        assert isinstance(ignore_exceptions, bool), 'ignore_exceptions must be of type <bool>'
        assert isinstance(print_exc, bool), 'print_exc must be of type <bool>'
        assert isinstance(wait_sl_tp_checking, (float, int)), 'wait_sl_tp_checking must be of type <float> or <int>'
        assert wait_sl_tp_checking < self._sec_interval, \
            'wait_sl_tp_checking cannot be greater than or equal to the timeframe'
        assert isinstance(limit, int), 'limit must be of type <int>'
        assert isinstance(strategy_in_sleep, bool), 'strategy_in_sleep must be of type <bool>'
        assert isinstance(coin_lotsize_division, bool), 'coin_lotsize_division must be of type <bool>'

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
                utils.logger.info("trading prediction at %s: %s", index, prediction)
                if print_out:
                    print(index, prediction)
                while True:
                    if not self.__exit_order__:
                        if (open_time + self._sec_interval) - time() > wait_sl_tp_checking:
                            utils.logger.debug("sleep %f seconds", wait_sl_tp_checking)
                            sleep(wait_sl_tp_checking)

                        price = self.client.get_ticker_price(ticker)
                        min_ = min(self.__last_stop_loss, self.__last_take_profit)
                        max_ = max(self.__last_stop_loss, self.__last_take_profit)
                        utils.logger.debug('checking SL/TP')
                        if (not (min_ < price < max_)) and prediction["predict"] != 'Exit':
                            self.__exit_order__ = True
                            utils.logger.info('exit trade')
                            index = f'{self.ticker}, {ctime()}'
                            utils.logger.info("trading prediction exit in sleeping at %s: %s", index, prediction)
                            if print_out:
                                print("trading prediction exit in sleeping at %s: %s", index, prediction)
                            if self.trading_on_client:
                                self.client.exit_last_order()
                        elif strategy_in_sleep:
                            break
                    if time() >= (open_time + self._sec_interval):
                        self._prev_predict = utils.convert_signal_str(self.returns[-1])
                        open_time += self._sec_interval
                        break
            except Exception as exc:
                utils.logger.error(f'An error occurred: {exc}', exc_info=True)  # how to concatenate with error?
                self.client.exit_last_order()
                if ignore_exceptions:
                    if print_exc:
                        print(exc)
                    continue
                else:
                    raise exc

    @utils.assert_logger
    def multi_realtime_trading(self,
                               tickers: List[str],
                               start_time: datetime,  # LOCAL TIME
                               strategy_name: str,
                               print_out: bool = True,
                               bet_for_trading_on_client: Union[float, int] = inf,  # for 1 trade
                               coin_lotsize_division: bool = True,
                               ignore_exceptions: bool = False,
                               print_exc: bool = True,
                               wait_sl_tp_checking: Union[float, int] = 5,
                               limit: int = 1000,
                               strategy_in_sleep: bool = False,
                               deposit_part: Union[float, int] = 1.0,  # for all trades
                               **strategy_kwargs):
        assert isinstance(tickers, Iterable), 'tickers must be of type <Iterable[str]>'
        for el in tickers:
            assert isinstance(el, str), 'tickers must be of type <Iterable[str]>'
            assert fullmatch(utils.TICKER_PATTERN, el), f'all tickers must match the pattern <{utils.TICKER_PATTERN}>'
        assert isinstance(print_out, bool), 'print_out must be of type <bool>'
        assert isinstance(bet_for_trading_on_client,
                          (float, int)), 'bet_for_trading_on_client must be of type <float> or <int>'
        assert isinstance(ignore_exceptions, bool), 'ignore_exceptions must be of type <bool>'
        assert isinstance(print_exc, bool), 'print_exc must be of type <bool>'
        assert isinstance(wait_sl_tp_checking, (float, int)), 'wait_sl_tp_checking must be of type <float> or <int>'
        assert wait_sl_tp_checking < self._sec_interval, \
            'wait_sl_tp_checking cannot be greater than or equal to the timeframe'
        assert isinstance(limit, int), 'limit must be of type <int>'
        assert isinstance(strategy_in_sleep, bool), 'strategy_in_sleep must be of type <bool>'
        assert isinstance(start_time, datetime), 'start_time must be of type <datetime.datetime>'
        assert start_time > datetime.now(), 'start_time cannot be earlier than the present time'
        assert isinstance(strategy_name, str), 'strategy_name must be of type <str>'
        assert strategy_name in self.__dir__(), 'There is no such strategy'
        assert isinstance(coin_lotsize_division, bool), 'coin_lotsize_division must be of type <bool>'
        assert isinstance(deposit_part, (int, float)), 'deposit_part must be of type <int> or <float>'
        assert 1 >= deposit_part > 0, 'deposit_part cannot be greater than 1 or less than 0(inclusively)'

        can_orders: int = len(tickers)
        bet_for_trading_on_client_copy: Union[float, int] = bet_for_trading_on_client

        class MultiRealTimeTrader(self.__class__):
            def get_trading_predict(self,
                                    bet_for_trading_on_client: Union[float, int] = inf,
                                    coin_lotsize_division: bool = True
                                    ) -> Dict[str, Union[str, float]]:
                balance = self.client.get_balance_ticker(self.ticker.split('/')[1])
                bet = (balance * 10) / (can_orders / deposit_part - TradingClient.cls_open_orders)
                bet /= 10  # decimal analog
                if bet > bet_for_trading_on_client_copy:
                    bet = bet_for_trading_on_client_copy
                return super().get_trading_predict(bet_for_trading_on_client=bet,
                                                   coin_lotsize_division=coin_lotsize_division)

        def start_trading(pare):
            trader = MultiRealTimeTrader(ticker=pare,
                                         interval=self.interval,
                                         trading_on_client=self.trading_on_client)
            trader.connect_graph(graph=self.fig)
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
        self.fig.log_y(_row=self.fig.data_row,
                       _col=self.fig.data_col)
        utils.logger.debug('trader log data')

    def log_deposit(self):
        self.fig.log_y(_row=self.fig.deposit_row,
                       _col=self.fig.deposit_col)
        utils.logger.debug('trader log deposit')

    def log_returns(self):
        self.fig.log_y(_row=self.fig.returns_row,
                       _col=self.fig.returns_col)
        utils.logger.debug('trader log returns')

    @utils.assert_logger
    def set_client(self, your_client: TradingClient):
        """
        :param your_client: trading client
        """
        assert isinstance(your_client, TradingClient), 'your_client must be of type <TradingClient>'

        self.client = your_client
        utils.logger.debug('trader set client')

    @utils.assert_logger
    def convert_signal(self,
                       old: utils.PREDICT_TYPE = utils.SELL,
                       new: utils.PREDICT_TYPE = utils.EXIT) -> utils.PREDICT_TYPE_LIST:
        assert isinstance(old, utils.PREDICT_TYPE) and isinstance(new, utils.PREDICT_TYPE), \
            'Arguments to this function can only be <utils.PREDICT_TYPE>.'

        pos: int
        val: utils.PREDICT_TYPE
        for pos, val in enumerate(self.returns):
            if val == old:
                self.returns[pos] = new
        utils.logger.debug("trader signals converted: %s >> %s", old, new)
        return self.returns

    @utils.assert_logger
    def set_open_stop_and_take(self,
                               take_profit: Union[float, int] = inf,
                               stop_loss: Union[float, int] = inf,
                               set_stop: bool = True,
                               set_take: bool = True):
        """
        :param set_take: create new take profits.
        :param set_stop: create new stop losses.
        :param take_profit: take profit in points
        :param stop_loss: stop loss in points
        """
        assert isinstance(take_profit, (float, int)), 'take_profit must be of type <float> or <int>'
        assert isinstance(stop_loss, (float, int)), 'stop_loss must be of type <float> or <int>'
        assert isinstance(set_stop, bool), 'set_stop must be of type <bool>'
        assert isinstance(set_take, bool), 'set_stop must be of type <bool>'

        self._take_profit = take_profit
        self._stop_loss = stop_loss
        take_flag: float = inf
        stop_flag: float = inf
        self._open_lot_prices = []
        if set_stop:
            self._stop_losses = []
        if set_take:
            self._take_profits = []
        closes: ndarray = self.df['Close'].values
        sig: utils.PREDICT_TYPE
        close: float
        converted: utils.CONVERTED_TYPE
        ts: Dict[str, float]
        for e, (sig, close, converted) in enumerate(zip(self.returns, closes, self._converted)):
            if converted is not nan:
                self._open_price = close
                if sig != utils.EXIT:
                    if set_take or set_stop:
                        ts = self.__get_stop_take(sig)
                    if set_take:
                        take_flag = ts['take']
                    if set_stop:
                        stop_flag = ts['stop']
                else:
                    take_flag = stop_flag = self._open_price

            self._open_lot_prices.append(self._open_price)
            if set_take:
                self._take_profits.append(take_flag)
            elif sig == utils.EXIT:
                self._take_profits[e] = take_flag
            if set_stop:
                self._stop_losses.append(stop_flag)
            elif sig == utils.EXIT:
                self._stop_losses[e] = take_flag
        utils.logger.debug('trader stop loss: %f pips, trader take profit: %f pips', stop_loss, take_profit)

    @utils.assert_logger
    def set_credit_leverages(self, credit_lev: Union[float, int] = 1.0):
        """
        Sets the leverage for bets.
        :param credit_lev: leverage in points
        """
        assert isinstance(credit_lev, (float, int)), 'credit_lev must be of type <float> or <int>'

        self._credit_leverages = [credit_lev for i in range(len(self.df['Close']))]
        utils.logger.debug('trader credit leverage: %f', credit_lev)

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

    @utils.assert_logger
    def strategy_diff(self, frame_to_diff: Series) -> utils.PREDICT_TYPE_LIST:
        """
        frame_to_diff:  |   pd.Series  |  example:  Trader.df['Close']
        """
        assert isinstance(frame_to_diff, Series), 'frame_to_diff must be of type <pd.Series>'

        self.returns = list(digitize(frame_to_diff.diff(), bins=[0]))
        self.convert_signal(1, utils.BUY)
        self.convert_signal(0, utils.SELL)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        return self.returns


class ExampleStrategies(Trader):

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

        flag_stop_loss: float = inf
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

    def strategy_ichimoku(self,
                          tenkansen: int = 9,
                          kijunsen: int = 26,
                          senkouspan: int = 52,
                          chinkouspan: int = 26,
                          stop_loss_plus: Union[float, int] = 40.0,  # sl_tp_adder
                          plot: bool = True) -> utils.PREDICT_TYPE_LIST:
        cloud = ta.trend.IchimokuIndicator(self.df["High"],
                                           self.df["Low"],
                                           tenkansen,
                                           kijunsen,
                                           senkouspan,
                                           visual=True,
                                           fillna=True)
        tenkan_sen: ndarray = cloud.ichimoku_conversion_line().values
        kinjun_sen: ndarray = cloud.ichimoku_base_line().values
        senkou_span_a: ndarray = cloud.ichimoku_a().values
        senkou_span_b: ndarray = cloud.ichimoku_b().values
        prices: Series = self.df['Close']
        chinkou_span: ndarray = prices.shift(-chinkouspan).values
        flag1: utils.PREDICT_TYPE = utils.EXIT
        flag2: utils.PREDICT_TYPE = utils.EXIT
        flag3: utils.PREDICT_TYPE = utils.EXIT
        trade: utils.PREDICT_TYPE = utils.EXIT
        name: str
        data: ndarray
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
                                          chinkou_span],
                                         ['red',
                                          'blue',
                                          'green']):
                self.fig.plot_line(
                    line=data,
                    name=name,
                    width=utils.ICHIMOKU_LINES_WIDTH,
                    color=color,
                    _row=self.fig.data_row,
                    _col=self.fig.data_col
                )

            self.fig.plot_area(fast=senkou_span_a,
                               slow=senkou_span_b,
                               name_fast=utils.SENKOU_SPAN_A_NAME,
                               name_slow=utils.SENKOU_SPAN_B_NAME)

            self.returns = [utils.EXIT for i in range(chinkouspan)]
            self._stop_losses = [self.df['Close'].values[0]] * chinkouspan
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
                min_cloud_now = min(senkou_span_a[e], senkou_span_b[e])
                max_cloud_now = max(senkou_span_a[e], senkou_span_b[e])
                if trade == utils.BUY:
                    self._stop_losses.append(min_cloud_now)
                elif trade == utils.SELL:
                    self._stop_losses.append(max_cloud_now)
                elif trade == utils.EXIT:
                    self._stop_losses.append(0.0)
                else:
                    raise ValueError('What???')

        self.set_open_stop_and_take(set_stop=False)
        self.set_credit_leverages()
        self.sl_tp_adder(add_stop_loss=stop_loss_plus)
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
            self.fig.plot_line(line=SMA1.values,
                               width=utils.MA_FAST_WIDTH,
                               color=utils.MA_FAST_COLOR,
                               name=utils.MA_FAST_NAME.format(fast),
                               opacity=utils.MA_FAST_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)

            self.fig.plot_line(line=SMA2.values,
                               width=utils.MA_SLOW_WIDTH,
                               color=utils.MA_SLOW_COLOR,
                               name=utils.MA_SLOW_NAME.format(slow),
                               opacity=utils.MA_SLOW_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)

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
            for SMA, color, speed, name, alpha, size in zip([SMA1, SMA2, SMA3],
                                                            [utils.MA_FAST_COLOR, utils.MA_MID_COLOR, utils.MA_SLOW_COLOR],
                                                            [fast, mid, slow],
                                                            [utils.MA_FAST_NAME, utils.MA_MID_NAME, utils.MA_SLOW_NAME],
                                                            [utils.MA_FAST_ALPHA, utils.MA_MID_ALPHA, utils.MA_SLOW_ALPHA],
                                                            [utils.MA_FAST_WIDTH, utils.MA_MID_WIDTH, utils.MA_SLOW_WIDTH]):
                self.fig.plot_line(line=SMA.values,
                                   width=size,
                                   color=color,
                                   name=name.format(speed),
                                   opacity=alpha,
                                   _row=self.fig.data_row,
                                   _col=self.fig.data_col)

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
            for SMA, color, speed, name, alpha, size in zip([ema3.values, ema21.values, ema46.values],
                                                            [utils.MA_FAST_COLOR, utils.MA_MID_COLOR, utils.MA_SLOW_COLOR],
                                                            [fast, mid, slow],
                                                            [utils.MA_FAST_NAME, utils.MA_MID_NAME, utils.MA_SLOW_NAME],
                                                            [utils.MA_FAST_ALPHA, utils.MA_MID_ALPHA, utils.MA_SLOW_ALPHA],
                                                            [utils.MA_FAST_WIDTH, utils.MA_MID_WIDTH, utils.MA_SLOW_WIDTH]):
                self.fig.plot_line(line=SMA.values,
                                   width=size,
                                   color=color,
                                   name=name.format(speed),
                                   opacity=alpha,
                                   _row=self.fig.data_row,
                                   _col=self.fig.data_col)

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
                     minimum: Union[float, int] = 20,
                     maximum: Union[float, int] = 80,
                     max_mid: Union[float, int] = 75,
                     min_mid: Union[float, int] = 35,
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
        sardown: ndarray = sar.psar_down().values
        sarup: ndarray = sar.psar_up().values
        self._stop_losses = list(sar.psar().values)

        if plot:
            for SAR_ in (sarup, sardown):
                self.fig.plot_line(line=sarup,
                                   width=utils.SAR_UP_WIDTH,
                                   color=utils.SAR_UP_COLOR,
                                   name=utils.SAR_UP_NAME,
                                   opacity=utils.SAR_UP_ALPHA,
                                   _row=self.fig.data_row,
                                   _col=self.fig.data_col)

                self.fig.plot_line(line=sardown,
                                   width=utils.SAR_DOWN_WIDTH,
                                   color=utils.SAR_DOWN_COLOR,
                                   name=utils.SAR_DOWN_NAME,
                                   opacity=utils.SAR_DOWN_ALPHA,
                                   _row=self.fig.data_row,
                                   _col=self.fig.data_col)

        for price, up, down in zip(
                list(self.df['Close'].values), list(sarup), list(sardown)):
            numup = nan_to_num(up, nan=-9999.0)
            numdown = nan_to_num(down, nan=-9999.0)
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
        histogram: DataFrame = DataFrame(macd_.values - signal_.values)
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
            self.fig.plot_line(line=st.get_supertrend_upper(),
                               width=utils.ST_UP_WIDTH,
                               color=utils.ST_UP_COLOR,
                               name=utils.ST_UP_NAME,
                               opacity=utils.ST_UP_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)

            self.fig.plot_line(line=st.get_supertrend_upper(),
                               width=utils.ST_DOWN_WIDTH,
                               color=utils.ST_DOWN_COLOR,
                               name=utils.ST_DOWN_NAME,
                               opacity=utils.ST_DOWN_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)
        self._stop_losses = list(st.get_supertrend())
        self.returns = list(st.get_supertrend_strategy_returns())
        self._stop_losses[0] = inf if self.returns[0] == utils.SELL else -inf
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

        mid_: Series = bollinger.bollinger_mavg()
        upper: Series = bollinger.bollinger_hband()
        lower: Series = bollinger.bollinger_lband()
        if plot:
            self.fig.plot_line(line=upper.values,
                               width=utils.UPPER_BB_WIDTH,
                               color=utils.UPPER_BB_COLOR,
                               name=utils.UPPER_BB_NAME,
                               opacity=utils.UPPER_BB_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)
            self.fig.plot_line(line=mid_.values,
                               width=utils.MID_BB_WIDTH,
                               color=utils.MID_BB_COLOR,
                               name=utils.MID_BB_NAME,
                               opacity=utils.MID_BB_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)
            self.fig.plot_line(line=lower.values,
                               width=utils.LOWER_BB_WIDTH,
                               color=utils.LOWER_BB_COLOR,
                               name=utils.LOWER_BB_NAME,
                               opacity=utils.LOWER_BB_ALPHA,
                               _row=self.fig.data_row,
                               _col=self.fig.data_col)
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

    def strategy_idris(self, points=20):
        self._stop_losses = [inf] * 2
        self._take_profits = [inf] * 2
        flag = utils.EXIT
        self.returns = [flag] * 2
        for e in range(len(self.df) - 2):
            bar3price = self.df['Close'][e + 2]
            mid2bar = (self.df['High'][e + 1] + self.df['Low'][e + 1]) / 2
            if bar3price < mid2bar:
                flag = utils.SELL
            elif bar3price > mid2bar:
                flag = utils.BUY
            f2 = flag
            self.returns.append(flag)
        self.set_open_stop_and_take(stop_loss=points * 2, take_profits=points * 20)
        self.set_credit_leverages()
        return self.returns
