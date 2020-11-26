#!/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)


# TODO:
#   eval to getatrr
#   rewrite extra in get-predict and realtime...

import itertools
import random
import time
import typing

import numpy as np
import pandas as pd
import ta
import ta.volatility
from binance.client import Client
from plotly.graph_objs import Line
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

try:
    from quick_trade import utils
except ImportError:
    from quick_trade.quick_trade import utils

try:  # in 3.9 tensorflow doesn't work
    from tensorflow.keras.layers import Dropout, Dense, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
except ImportError:
    utils.logger.critical('the tensorflow package will not work')  # python 3.9 support


class TradingClient(Client):
    ordered: bool = False
    __side__: str
    quantity: float
    ticker: str
    order: dict[str, typing.Any]

    def get_ticker_price(self, ticker: str):
        return float(self.get_symbol_ticker(symbol=ticker)['price'])

    @staticmethod
    def get_data(ticker: str = 'None', interval: str = 'None', **get_kw):
        return utils.get_binance_data(ticker, interval, **get_kw)

    def new_order_buy(self, ticker: str = 'None', quantity: float = 0.0, credit_leverage: float = 1.0):
        self.__side__ = 'Buy'
        self.quantity = quantity
        self.ticker = ticker
        self.order = self.order_market_buy(symbol=ticker, quantity=quantity)
        self.order_id = self.order['orderId']
        self.ordered = True

    def new_order_sell(self, ticker: str = 'None', quantity: float = 0.0, credit_leverage: float = 1.0):
        self.__side__ = 'Sell'
        self.quantity = quantity
        self.ticker = ticker
        self.order = self.order_market_sell(symbol=ticker, quantity=quantity)
        self.order_id = self.order['orderId']
        self.ordered = True

    def exit_last_order(self):
        if self.ordered:
            if self.__side__ == 'Sell':
                self.new_order_buy(self.ticker, self.quantity)
            elif self.__side__ == 'Buy':
                self.new_order_sell(self.ticker, self.quantity)
            self.__side__ = 'Exit'
            self.ordered = False

    def get_balance_ticker(self, ticker: str) -> float:
        for asset in self.get_account()['balances']:
            if asset['asset'] == ticker:
                return float(asset['free'])


class Trader(object):
    """
    algo-trading system.


    ticker:   |     str      |  ticker/symbol of chart

    df:       |   dataframe  |  data of chart

    interval: |     str      |  interval of df.
    one of:
    1m    30m    3h    3M
    2m    45m    4h    6M
    3m    1h     1d
    5m    90m    1w
    15m   2h     1M

    """
    profit_calculate_coef: float

    def __init__(self,
                 ticker='AAPL',
                 df: pd.DataFrame = np.nan,
                 interval='1d',
                 rounding=50,
                 *args,
                 **kwargs):
        df_: pd.DataFrame = round(df, rounding)
        self.__first__: bool = True
        self.__rounding__: int = rounding
        self.__oldsig = utils.EXIT
        self.df: pd.DataFrame = df_.reset_index(drop=True)
        self.ticker: str = ticker
        self.interval: str = interval
        if interval == '1m':
            self.profit_calculate_coef = 1 / (60 / 24 / 365)
        if interval == '2m':
            self.profit_calculate_coef = 1 / (30 / 24 / 365)
        elif interval == '3m':
            self.profit_calculate_coef = 1 / (20 / 24 / 365)
        elif interval == '5m':
            self.profit_calculate_coef = 1 / (12 / 24 / 365)
        elif interval == '15m':
            self.profit_calculate_coef = 1 / (4 / 24 / 365)
        elif interval == '30m':
            self.profit_calculate_coef = 1 / (2 / 24 / 365)
        elif interval == '45m':
            self.profit_calculate_coef = 1 / (32 / 365)
        elif interval == '1h':
            self.profit_calculate_coef = 1 / (24 / 365)
        elif interval == '90m':
            self.profit_calculate_coef = 1 / (18 / 365)
        elif interval == '2h':
            self.profit_calculate_coef = 1 / (12 / 365)
        elif interval == '3h':
            self.profit_calculate_coef = 1 / (8 / 365)
        elif interval == '4h':
            self.profit_calculate_coef = 1 / (6 / 365)
        elif interval == '1d':
            self.profit_calculate_coef = 1 / 365
        elif interval == '1w':
            self.profit_calculate_coef = 1 / 52
        elif interval == '1M':
            self.profit_calculate_coef = 1 / 12
        elif interval == '3M':
            self.profit_calculate_coef = 1 / 4
        elif interval == '6M':
            self.profit_calculate_coef = 1 / 2
        else:
            raise ValueError('I N C O R R E C T   I N T E R V A L')
        self.__inputs: int = utils.INPUTS
        self.__exit_order__: bool = False

    def __repr__(self):
        return 'trader'

    def _get_attr(self, attr: str):
        return getattr(self, attr)

    @classmethod
    def _get_this_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def kalman_filter(self,
                      df='self.df["Close"]',
                      iters=5,
                      plot: bool = True,
                      *args,
                      **kwargs):
        filtered: np.ndarray
        k_filter: KalmanFilter = KalmanFilter()
        if isinstance(df, str):
            df: pd.Series = eval(df)
        df: pd.Series
        filtered = k_filter.filter(np.array(df))[0]
        for i in range(iters):
            filtered = k_filter.smooth(filtered)[0]
        if plot:
            self.fig.add_trace(
                Line(
                    name='kalman filter',
                    y=filtered.T[0],
                    line=dict(width=utils.SUB_LINES_WIDTH)), 1, 1)
        return pd.DataFrame(filtered)

    def scipy_filter(self,
                     window_length=101,
                     df='self.df["Close"]',
                     polyorder=3,
                     plot=True,
                     **scipy_savgol_filter_kwargs):
        if isinstance(df, str):
            df = eval(df)
        filtered = signal.savgol_filter(
            df,
            window_length=window_length,
            polyorder=polyorder,
            **scipy_savgol_filter_kwargs)
        if plot:
            self.fig.add_trace(
                Line(
                    name='savgol filter',
                    y=filtered,
                    line=dict(width=utils.SUB_LINES_WIDTH)), 1, 1)
        return pd.DataFrame(filtered)

    def bull_power(self, periods):
        EMA = ta.trend.ema(self.df['Close'], periods)
        return np.array(self.df['High']) - EMA

    def tema(self, periods, *args, **kwargs):
        ema = ta.trend.ema(self.df['Close'], periods)
        ema2 = ta.trend.ema(ema, periods)
        ema3 = ta.trend.ema(ema2, periods)
        return pd.DataFrame(3 * ema.values - 3 * ema2.values + ema3.values)

    def linear_(self, dataset):
        """
        linear data. mean + (mean diff * n)

        """
        if isinstance(dataset, str):
            dataset = eval(dataset)
        data = pd.DataFrame(dataset).copy()

        mean = float(data.mean())
        mean_diff = float(data.diff().mean())
        start = mean - (mean_diff * (len(data) / 2))
        end = start + (mean - start) * 2

        length = len(data)
        return_list = []
        mean_diff = (end - start) / length
        for i in range(length):
            return_list.append(start + mean_diff * i)
        self.mean_diff = mean_diff
        return np.array(return_list)

    def __get_stop_take(self, sig):
        """
        calculating stop loss and take profit.


        sig:        |     int     |  signal to sell/buy/exit:
            EXIT -- exit.
            BUY -- buy.
            SELL -- sell.

        """

        if self.stop_loss is not None:
            _stop_loss = self.stop_loss / 10_000 * self.open_price
        else:
            _stop_loss = np.inf
        if self.take_profit is not None:
            take = self.take_profit / 10_000 * self.open_price
        else:
            take = np.inf

        if sig == utils.BUY:
            _stop_loss = self.open_price - _stop_loss
            take = self.open_price + take
        elif sig == utils.SELL:
            take = self.open_price - take
            _stop_loss = self.open_price + _stop_loss
        else:
            if self.take_profit is not None:
                take = self.open_price
            if self.stop_loss is not None:
                _stop_loss = self.open_price

        return {'stop': _stop_loss, 'take': take}

    def strategy_diff(self, frame_to_diff, *args, **kwargs):
        """
        frame_to_diff:  |   pd.DataFrame  |  example:  Trader.df['Close']

        """
        if isinstance(frame_to_diff, str):
            frame_to_diff = eval(frame_to_diff)
        self.returns = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        return self.returns

    def strategy_buy_hold(self, *args, **kwargs):
        self.returns = [utils.BUY for _ in range(len(self.df))]
        return self.returns

    def strategy_2_sma(self, slow=100, fast=30, plot=True, *args, **kwargs):
        return_list = []
        SMA1 = ta.trend.sma(self.df['Close'], fast)
        SMA2 = ta.trend.sma(self.df['Close'], slow)
        if plot:
            self.fig.add_trace(
                Line(
                    name=f'SMA{fast}',
                    y=SMA1.values,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.G)), 1, 1)
            self.fig.add_trace(
                Line(
                    name=f'SMA{slow}',
                    y=SMA2.values,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.R)), 1, 1)

        for SMA13, SMA26 in zip(SMA1, SMA2):
            if SMA26 < SMA13:
                return_list.append(utils.BUY)
            elif SMA13 < SMA26:
                return_list.append(utils.SELL)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def strategy_3_sma(self,
                       slow=100,
                       mid=26,
                       fast=13,
                       plot=True,
                       *args,
                       **kwargs):
        return_list = []
        SMA1 = ta.trend.sma(self.df['Close'], fast)
        SMA2 = ta.trend.sma(self.df['Close'], mid)
        SMA3 = ta.trend.sma(self.df['Close'], slow)

        if plot:
            for SMA, Co, name in zip([SMA1, SMA2, SMA3],
                                     [utils.G, utils.B, utils.R],
                                     [fast, mid, slow]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=SMA.values,
                        line=dict(width=utils.SUB_LINES_WIDTH, color=Co)), 1, 1)

        for SMA13, SMA26, SMA100 in zip(SMA1, SMA2, SMA3):
            if SMA100 < SMA26 < SMA13:
                return_list.append(utils.BUY)
            elif SMA100 > SMA26 > SMA13:
                return_list.append(utils.SELL)
            else:
                return_list.append(utils.EXIT)

        self.returns = return_list
        return return_list

    def strategy_3_ema(self,
                       slow=3,
                       mid=21,
                       fast=46,
                       plot=True,
                       *args,
                       **kwargs):
        ema3 = ta.trend.ema(self.df['Close'], slow)
        ema21 = ta.trend.ema(self.df['Close'], mid)
        ema46 = ta.trend.ema(self.df['Close'], fast)
        return_list = []

        if plot:
            for ema, Co, name in zip([ema3.values, ema21.values, ema46.values],
                                     [utils.G, utils.B, utils.R], [slow, mid, fast]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=ema,
                        line=dict(width=utils.SUB_LINES_WIDTH, color=Co)), 1, 1)

        for EMA1, EMA2, EMA3 in zip(ema3, ema21, ema46):
            if EMA1 > EMA2 > EMA3:
                return_list.append(utils.BUY)
            elif EMA1 < EMA2 < EMA3:
                return_list.append(utils.SELL)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def strategy_macd(self, slow=100, fast=30, *args, **kwargs):
        return_list = []
        level = ta.trend.macd_signal(self.df['Close'], slow, fast)
        macd = ta.trend.macd(self.df['Close'], slow, fast)

        for j, k in zip(level.values, macd.values):
            if j > k:
                return_list.append(utils.SELL)
            elif k > j:
                return_list.append(utils.BUY)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def strategy_exp_diff(self, period=70, plot=True, *args, **kwargs):
        exp = self.tema(period)
        return_list = self.strategy_diff(exp)
        if plot:
            self.fig.add_trace(
                Line(
                    name=f'EMA{period}',
                    y=exp.values.T[0],
                    line=dict(width=utils.SUB_LINES_WIDTH)), 1, 1)

        self.returns = return_list
        return return_list

    def strategy_rsi(self,
                     minimum=20,
                     maximum=80,
                     max_mid=75,
                     min_mid=35,
                     *args,
                     **rsi_kwargs):
        rsi = ta.momentum.rsi(self.df['Close'], **rsi_kwargs)
        return_list = []
        flag = utils.EXIT

        for val, diff in zip(rsi.values, rsi.diff().values):
            if val < minimum and diff > 0 and val is not pd.NA:
                flag = utils.BUY
            elif val > maximum and diff < 0 and val is not pd.NA:
                flag = utils.SELL
            elif flag == utils.BUY and val < max_mid:
                flag = utils.EXIT
            elif flag == utils.SELL and val > min_mid:
                flag = utils.EXIT
            return_list.append(flag)

        self.returns = return_list
        return return_list

    def strategy_macd_rsi(self,
                          mac_slow=26,
                          mac_fast=12,
                          rsi_level=50,
                          rsi_kwargs: dict = None,
                          *args,
                          **macd_kwargs):
        if rsi_kwargs is None:
            rsi_kwargs = {}
        return_list = []
        macd = ta.trend.macd(self.df['Close'], mac_slow, mac_fast,
                             **macd_kwargs)
        rsi = ta.momentum.rsi(self.df['Close'], **rsi_kwargs)
        for MACD, RSI in zip(macd.values, rsi.values):
            if MACD > 0 and RSI > rsi_level:
                return_list.append(utils.BUY)
            elif MACD < 0 and RSI < rsi_level:
                return_list.append(utils.SELL)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def strategy_parabolic_SAR(self, plot=True, *args, **sar_kwargs):
        return_list = []
        sar = ta.trend.PSARIndicator(self.df['High'], self.df['Low'],
                                     self.df['Close'], **sar_kwargs)
        sardown = sar.psar_down().values
        sarup = sar.psar_up().values

        if plot:
            for SAR_ in (sarup, sardown):
                self.fig.add_trace(
                    Line(
                        name='SAR', y=SAR_, line=dict(width=utils.SUB_LINES_WIDTH)),
                    1, 1)
        for price, up, down in zip(
                list(self.df['Close'].values), list(sarup), list(sardown)):
            numup = np.nan_to_num(up, nan=-9999)
            numdown = np.nan_to_num(down, nan=-9999)
            if numup != -9999:
                return_list.append(utils.BUY)
            elif numdown != -9999:
                return_list.append(utils.SELL)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def strategy_macd_histogram_diff(self,
                                     slow=23,
                                     fast=12,
                                     *args,
                                     **macd_kwargs):
        _MACD_ = ta.trend.MACD(self.df['Close'], slow, fast, **macd_kwargs)
        signal_ = _MACD_.macd_signal()
        macd_ = _MACD_.macd()
        histogram = pd.DataFrame(macd_.values - signal_.values)
        return_list = utils.digit(histogram.diff().values)
        self.returns = return_list
        return return_list

    def strategy_regression_model(self, plot=True, *args, **kwargs):
        return_list = []
        for i in range(self.__inputs - 1):
            return_list.append(utils.EXIT)
        data_to_pred = np.array(
            utils.get_window(np.array([self.df['Close'].values]).T, self.__inputs))

        data_to_pred = data_to_pred.T
        for e, data in enumerate(data_to_pred):
            data_to_pred[e] = self.scaler.fit_transform(data)
        data_to_pred = data_to_pred.T

        predictions = itertools.chain.from_iterable(
            self.model.predict(data_to_pred))
        predictions = pd.DataFrame(predictions)
        frame = predictions
        predictions = self.strategy_diff(predictions)
        frame = self.scaler.inverse_transform(frame.values.T).T
        self.returns = [*return_list, *predictions]
        nans = itertools.chain.from_iterable([(np.nan,) * self.__inputs])
        filt = (*nans, *frame.T[0])
        if plot:
            self.fig.add_trace(
                Line(
                    name='predict',
                    y=filt,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.C)),
                row=1,
                col=1)
        return self.returns, filt

    def get_network_regression(self,
                               dataframes,
                               inputs=60,
                               network_save_path='./model_regression',
                               **fit_kwargs):
        """based on
        https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb

        please, use one dataframe.

        """

        self.__inputs = inputs
        model = Sequential()
        model.add(
            LSTM(units=50, return_sequences=True, input_shape=(inputs, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        for df in dataframes:
            data = df.filter(['Close'])
            dataset = data.values
            scaled_data = self.scaler.fit_transform(dataset)
            train_data = scaled_data[0:len(scaled_data), :]
            x_train = []
            y_train = []
            for i in range(inputs, len(train_data)):
                x_train.append(train_data[i - inputs:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train,
                                 (x_train.shape[0], x_train.shape[1], 1))
            model.fit(x_train, y_train, **fit_kwargs)
        self.model = model
        model.save(network_save_path)
        return model

    def prepare_scaler(self, dataframe: pd.DataFrame, regression_net=True):
        """
        if you are loading a neural network.

        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        if regression_net:
            data = dataframe.filter(['Close'])
            dataset = data.values
        else:
            dataset = dataframe.values
        scaled_data = scaler.fit_transform(dataset)
        self.scaler = scaler
        return scaled_data

    def get_trained_network(self,
                            dataframes,
                            filter_='kalman_filter',
                            filter_kwargs=None,
                            optimizer='adam',
                            loss='mse',
                            metrics=None,
                            network_save_path='./model_predicting',
                            **fit_kwargs):
        """
        getting trained neural network to trading.

        dataframes:  | list, tuple. |   list of pandas dataframes with columns:
            'High'
            'Low'
            'Open'
            'Close'
            'Volume'

        optimizer:    |       str         |   optimizer for .compile of network.

        filter_:      |       str         |    filter to training.

        filter_kwargs:|       dict        |    named arguments for the filter.

        loss:         |       str         |   loss for .compile of network.

        metrics:      |  list of strings  |   metrics for .compile of network:
            standard: ['acc']

        fit_kwargs:   | *named arguments* |   arguments to .fit of network.


        returns:
            (tensorflow model,
            history of training,
            (input training data, output train data))

        """

        if filter_kwargs is None:
            filter_kwargs = dict()
        if metrics is None:
            metrics = ['acc']
        list_input = []
        list_output = []
        flag = self.df

        for df in dataframes:
            self.df = df
            all_ta = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close',
                                            'Volume', True)
            filter_kwargs['plot'] = False
            output1 = self.strategy_diff(
                eval('self.' + filter_ + '(**filter_kwargs)'))

            for output in output1:
                list_output.append(output[0])
            list_input.append(
                pd.DataFrame(
                    self.prepare_scaler(
                        pd.DataFrame(all_ta), regression_net=False)))
        self.df = flag
        del flag
        input_df = pd.concat(list_input, axis=0).dropna(1)

        input_train_array = input_df.values
        output_train_array = np.array([list_output]).T

        model = Sequential()
        model.add(
            Dense(20, input_dim=len(input_train_array[0]), activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(1, 'sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        hist = model.fit(input_train_array, output_train_array, **fit_kwargs)

        self.model = model
        self.history = hist
        self.training_set = (input_train_array, output_train_array)
        model.save(network_save_path)
        return model, hist, self.training_set

    def strategy_random_pred(self, *args, **kwargs):
        self.returns = []
        for i in range(len(self.df)):
            self.returns.append(random.randint(0, 2))

    def strategy_with_network(self, rounding=0, *args, **kwargs):
        scaler = MinMaxScaler(feature_range=(0, 1))
        all_ta = ta.add_all_ta_features(self.df, "Open", 'High', 'Low',
                                        'Close', "Volume", True).values
        preds = self.model.predict(scaler.fit_transform(all_ta))
        for e, i in enumerate(preds):
            preds[e] = round(i[0], rounding)
        self.returns = preds
        return preds

    def strategy_bollinger(self, plot=True, to_mid=True, *bollinger_args, **bollinger_kwargs):
        return_list = []
        flag = utils.EXIT
        bollinger = ta.volatility.BollingerBands(self.df['Close'], fillna=True, *bollinger_args, **bollinger_kwargs)

        mid_ = bollinger.bollinger_mavg()
        upper = bollinger.bollinger_hband()
        lower = bollinger.bollinger_lband()
        if plot:
            for TR, name in zip([upper, mid_, lower], ['upper band', 'mid band', 'lower band']):
                self.fig.add_trace(Line(y=TR, name=name, line=dict(width=utils.SUB_LINES_WIDTH)), col=1, row=1)
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
            return_list.append(flag)
        self.returns = return_list

    def get_heikin_ashi(self, df: pd.DataFrame = pd.DataFrame()):
        """

        :param df: dataframe, standard: self.df
        :return: heikin ashi
        """
        if 'Close' not in df.columns:
            df: pd.DataFrame = self.df
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

    def inverse_strategy(self, *args, **kwargs):
        """
        makes signals inverse:

        buy = sell.
        sell = buy.
        exit = exit.

        """

        return_list = []
        for signal_key in self.returns:
            if signal_key == utils.BUY:
                return_list.append(utils.SELL)
            elif signal_key == utils.SELL:
                return_list.append(utils.BUY)
            else:
                return_list.append(utils.EXIT)
        self.returns = return_list
        return return_list

    def basic_backtest(self,
                       deposit: float = 10_000.0,
                       bet: float = np.inf,
                       commission: float = 0.0,
                       plot: bool = True,
                       print_out: bool = True,
                       column: str = 'Close',
                       *args,
                       **kwargs):
        """
        testing the strategy.


        deposit:         | int, float. | start deposit.

        bet:             | int, float, | fixed bet to quick_trade. np.inf = all moneys.

        commission:      | int, float. | percentage commission (0 -- 100).

        plot:            |    bool.    | plotting.

        print_out:       |    bool.    | printing.

        column:          |     str     | column of dataframe to backtest.



        returns: pd.DataFrame with data of:
            signals,
            deposit'
            stop loss,
            take profit,
            linear deposit,
            <<column>> price,
            open lot price.


        """

        exit_take_stop: bool
        no_order: bool
        stop_loss: float
        diff: float
        lin_calc_df: pd.DataFrame
        price: float

        start_bet: float = bet
        data_column: pd.Series = self.df[column]
        self.deposit_history = [deposit]
        seted_ = utils.set_(self.returns)
        self.trades: int = 0
        self.profits: int = 0
        self.losses: int = 0
        moneys_open_bet: float = deposit
        money_start: float = deposit
        oldsig = utils.EXIT
        start_commision: float = commission

        for e, (sig,
                stop_loss,
                take_profit,
                seted,
                credit_lev) in enumerate(zip(self.returns[:-1],
                                             self.stop_losses[:-1],
                                             self.take_profits[:-1],
                                             seted_[:-1],
                                             self.credit_leverages[:-1]), 1):
            price = data_column[e]

            if seted is not np.nan:
                if oldsig != utils.EXIT:
                    commission = start_commision * 2
                else:
                    commission = start_commision
                if bet > deposit:
                    bet = deposit
                open_price = price
                bet *= credit_lev

                def coefficient(difference: float) -> float:
                    return bet * difference / open_price

                deposit -= bet * (commission / 100)
                if bet > deposit:
                    bet = deposit
                self.trades += 1
                if deposit > moneys_open_bet:
                    self.profits += 1
                elif deposit < moneys_open_bet:
                    self.losses += 1
                moneys_open_bet = deposit
                no_order = False
                exit_take_stop = False

            if not e:
                diff = 0.0
            if min(stop_loss, take_profit) < price < max(stop_loss, take_profit):
                diff = data_column[e] - data_column[e - 1]
            else:
                exit_take_stop = True
                if sig == utils.BUY and price >= take_profit:
                    diff = take_profit - data_column[e - 1]
                elif sig == utils.BUY and price <= stop_loss:
                    diff = stop_loss - data_column[e - 1]
                elif sig == utils.SELL and price >= stop_loss:
                    diff = stop_loss - data_column[e - 1]
                elif sig == utils.SELL and price <= take_profit:
                    diff = take_profit - data_column[e - 1]
                else:
                    diff = 0.0

            if sig == utils.SELL:
                diff = -diff
            elif sig == utils.EXIT:
                diff = 0.0
            if not no_order:
                deposit += coefficient(diff)
            no_order = exit_take_stop
            self.deposit_history.append(deposit)
            oldsig = sig

        self.linear = self.linear_(self.deposit_history)
        lin_calc_df = pd.DataFrame(self.linear)
        mean_diff = float(lin_calc_df.diff().mean())
        self.year_profit = mean_diff / self.profit_calculate_coef + money_start
        self.year_profit = ((self.year_profit - money_start) / money_start) * 100
        self.info = (
            f"L O S S E S: {self.losses}\n"
            f"T R A D E S: {self.trades}\n"
            f"P R O F I T S: {self.profits}\n"
            f"M E A N   Y E A R   P E R C E N T A G E   R O F I T: {self.year_profit}%\n"
        )
        if print_out:
            print(self.info)
        self.backtest_out_no_drop = pd.DataFrame(
            (self.deposit_history, self.stop_losses, self.take_profits, self.returns,
             self.open_lot_prices, data_column, self.linear),
            index=[
                f'deposit ({column})', 'stop loss', 'take profit',
                'predictions', 'open deal/lot', column,
                f"linear deposit data ({column})"
            ]).T
        self.backtest_out = self.backtest_out_no_drop.dropna()
        if plot:
            loc: pd.Series = self.df[column]
            self.fig.add_candlestick(
                close=self.df['Close'],
                high=self.df['High'],
                low=self.df['Low'],
                open=self.df['Open'],
                row=1,
                col=1,
                name=self.ticker)
            self.fig.add_trace(
                Line(
                    y=self.take_profits,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.G),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='take profit'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.stop_losses,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.R),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='stop loss'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.open_lot_prices,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.B),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='open lot'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.deposit_history,
                    line=dict(color=utils.COLOR_DEPOSIT),
                    name=f'D E P O S I T  (S T A R T: ${money_start})'), 2, 1)
            self.fig.add_trace(Line(y=self.linear, name='L I N E A R'), 2, 1)
            for e, i in enumerate(utils.set_(self.returns)):
                if i == utils.SELL:
                    self.fig.add_scatter(
                        name='Sell',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#FF0000'),
                        marker=dict(
                            symbol='triangle-down',
                            size=utils.SCATTER_SIZE,
                            opacity=utils.SCATTER_ALPHA))
                elif i == utils.BUY:
                    self.fig.add_scatter(
                        name='Buy',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#00FF00'),
                        marker=dict(
                            symbol='triangle-up',
                            size=utils.SCATTER_SIZE,
                            opacity=utils.SCATTER_ALPHA))
                elif i == utils.EXIT:
                    self.fig.add_scatter(
                        name='Exit',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#2a00ff'),
                        marker=dict(
                            symbol='triangle-left',
                            size=utils.SCATTER_SIZE,
                            opacity=utils.SCATTER_ALPHA))
            self.fig.update_layout(xaxis_rangeslider_visible=False)
            self.fig.show()
        return self.backtest_out

    def set_pyplot(self,
                   height: int = 900,
                   width: int = 1300,
                   template: str = 'plotly_dark',
                   row_heights: list = None,
                   **subplot_kwargs):
        """

        :param height: window height
        :param width: window width
        :param template: plotly template
        :param row_heights: standard [100, 160]
        :param subplot_kwargs: kw
        :return:
        """
        if row_heights is None:
            row_heights = [100, 160]
        self.fig = make_subplots(2, 1, row_heights=row_heights, **subplot_kwargs)
        self.fig.update_layout(
            height=height,
            width=width,
            template=template,
            xaxis_rangeslider_visible=False)
        if self.interval == '1m':
            title_ax = 'M I N U T E S'
        else:
            title_ax = 'D A Y S'
        self.fig.update_xaxes(
            title_text=title_ax, row=2, col=1, color=utils.TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='M O N E Y S', row=2, col=1, color=utils.TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='D A T A', row=1, col=1, color=utils.TEXT_COLOR)

    def strategy_collider(self,
                          first_func=utils.nothing,
                          second_func=utils.nothing,
                          kwargs_first_func: dict = None,
                          kwargs_second_func: dict = None,
                          mode: str = 'minimalist',
                          *args,
                          **kwargs):
        """
        first_func:      |  trading strategy  |   strategy to combine.

        standard: utils.nothing.

        example:  Trader.strategy_macd.

        second_func:     |  trading strategy  |   strategy to combine.

        standard: nothing.

        kwargs_first_func: |       dict         |   named arguments to first function.

        kwargs_second_func:|       dict         |   named arguments to second function.

        mode:            |         str        |   mode of combining:
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
                                ====
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

                    first_returns = [1,1,0,0,2,0,2,2,0,0,1]
                    second_returns = [1,2,2,2,2,2,0,0,0,0,1]
                                ====
                        [1,1,1,1,2,2,2,2,0,0,1]
                mode = 'super':
                    ...

                    first_returns = [1,1,1,2,2,2,0,0,1]
                    second_returns = [1,0,0,0,1,1,1,0,0]
                                ====
                        [1,0,0,2,1,1,0,0,1]


        !!!

        if your function/strategy is <<nothing>>, then your <<args_x_func>>
        should be an output data: [1,1,0 ... 0,2,1]

        !!!


        returns: combining of 2 strategies:

        example:

        Trader.strategy_collider(
           Trader.strategy_2_sma,
           Trader.strategy_3_ema,
           (30, 10)
        )

        or:

        Trader.strategy_collider(Trader.strategy_2_sma,
                             Trader.strategy_2_sma,
                             (300, 200),
                             (200, 100))
                                  =
                   Trader.strategy_3_sma(300, 200, 100)


        """

        if kwargs_second_func is None:
            kwargs_second_func = {}
        if kwargs_first_func is None:
            kwargs_first_func = {}
        first_returns = first_func(**kwargs_first_func)
        second_returns = second_func(**kwargs_second_func)
        return_list = []
        if mode == 'minimalist':
            for ret1, ret2 in zip(first_returns, second_returns):
                if ret1 == ret2:
                    return_list.append(ret1)
                else:
                    return_list.append(utils.EXIT)
        elif mode == 'maximalist':
            return_list = self.__maximalist(first_returns, second_returns)
        elif mode == 'super':
            return_list = self.__collide_super(first_returns, second_returns)
        else:
            raise ValueError('I N C O R R E C T   M O D E')
        self.returns = return_list
        return return_list

    @staticmethod
    def __maximalist(returns1, returns2):
        return_list = []
        flag = utils.EXIT
        for a, b in zip(returns1, returns2):
            if a == b:
                return_list.append(a)
                flag = a
            else:
                return_list.append(flag)
        return return_list

    @staticmethod
    def __collide_super(l1, l2):
        return_list = []
        for first, sec in zip(utils.set_(l1), utils.set_(l2)):
            if first is not np.nan and sec is not np.nan and first is not sec:
                return_list.append(utils.EXIT)
            elif first is sec:
                return_list.append(first)
            elif first is np.nan:
                return_list.append(sec)
            else:
                return_list.append(first)
        return utils.anti_set_(return_list)

    def get_trading_predict(self,
                            trading_on_client: bool = False,
                            bet_for_trading_on_client: float = np.inf,
                            second_symbol_of_ticker: str = 'None',
                            rounding_bet: int = 4,
                            #*args,
                            #**kwargs
                            ):
        """
        :param rounding_bet: maximum permissible accuracy with your api
        :param second_symbol_of_ticker: BTCUSDT -> USDT
        :param trading_on_client: trading on real client (boll)
        :param bet_for_trading_on_client: standard: all deposit
        :return: dict with prediction
        """

        def convert():
            nonlocal predict
            if predict == utils.BUY:
                predict = 'Buy'
            elif predict == utils.SELL:
                predict = 'Sell'
            elif predict == utils.EXIT:
                predict = 'Exit'

        credit_leverage = self.credit_leverages[-1]

        if trading_on_client:
            _moneys_ = self.client.get_balance_ticker(second_symbol_of_ticker)
            if bet_for_trading_on_client is np.inf:
                bet = _moneys_
            elif bet_for_trading_on_client > _moneys_:
                bet = _moneys_
            else:
                bet = bet_for_trading_on_client
            bet /= self.client.get_ticker_price(self.ticker)

            def min_r(r):
                return round(float('0.' + '0' * (r - 1) + '1'), r)

            bet = round(bet, rounding_bet) - min_r(rounding_bet)
            bet = round(bet, rounding_bet)

        predict = self.returns[-1]
        if self.__exit_order__:
            predict = utils.EXIT

        cond = "_predict" not in self.__dir__()
        if not cond:
            cond = self._predict != predict
        close = self.df['Close'].values

        if cond:
            convert()
            utils.logger.info(f'open lot {predict}')
            if trading_on_client:
                if predict == 'Buy':
                    if not self.__first__:
                        self.client.exit_last_order()
                        utils.logger.info('client exit')
                    self.client.new_order_buy(self.ticker, bet, credit_leverage=credit_leverage)
                    utils.logger.info('client buy')
                    self.__exit_order__ = False
                    self.__first__ = False

                if predict == 'Sell':
                    if not self.__first__:
                        self.client.exit_last_order()
                        utils.logger.info('client exit')
                    self.client.new_order_sell(self.ticker, bet, credit_leverage=credit_leverage)
                    utils.logger.info('client sell')
                    self.__first__ = False
                    self.__exit_order__ = False

                if predict == 'Exit':
                    if not self.__first__:
                        self.client.exit_last_order()
                        utils.logger.info('client exit')
                        self.__exit_order__ = True

            for sig, close_ in zip(self.returns[::-1],
                                   self.df['Close'].values[::-1]):
                if sig != utils.EXIT:
                    self.open_price = close_
                    break
            self._predict = predict
            self.__stop_loss = self.stop_losses[-1]
            self.__take_profit = self.take_profits[-1]
        convert()
        return {
            'predict': predict,
            'open lot price': self.open_price,
            'stop loss': self.__stop_loss,
            'take profit': self.__take_profit,
            'currency close': close[-1]
        }

    def realtime_trading(self,
                         ticker: str,
                         strategy,
                         get_data_kwargs=None,
                         sleeping_time: float = 60,
                         print_out: bool = True,
                         trading_on_client: bool = False,
                         bet_for_trading_on_client: float = np.inf,
                         second_symbol_of_ticker: str = None,
                         rounding_bet: int = 4,
                         *strategy_args,
                         **strategy_kwargs):
        """

        :param ticker: ticker for trading.
        :param strategy: trading strategy.
        :param get_data_kwargs: named arguments to self.client.get_data WITHOUT TICKER.
        :param sleeping_time: sleeping time.
        :param print_out: printing.
        :param trading_on_client: trading on client
        :param bet_for_trading_on_client: trading bet, standard: all deposit
        :param second_symbol_of_ticker: USDUAH -> UAH
        :param rounding_bet: maximum accuracy for trading
        :param strategy_kwargs: named arguments to <<strategy>>.
        :param strategy_args: arguments to <<strategy>>.

        """


        if get_data_kwargs is None:
            get_data_kwargs = dict()
        self.realtie_returns = {}
        self.ticker = ticker
        try:
            __now__ = time.time()
            while True:
                self.prepare_realtime = True
                self.df = self.client.get_data(self.ticker, **get_data_kwargs).reset_index(drop=True)
                strategy(*strategy_args, **strategy_kwargs)

                prediction = self.get_trading_predict(
                    trading_on_client=trading_on_client,
                    bet_for_trading_on_client=bet_for_trading_on_client,
                    second_symbol_of_ticker=second_symbol_of_ticker,
                    rounding_bet=rounding_bet)

                index = f'{self.ticker}, {time.ctime()}'
                if print_out:
                    print(index, prediction)
                utils.logger.info(f"trading prediction at {index}: {prediction}")
                self.realtie_returns[index] = prediction
                while True:
                    if not self.__exit_order__:
                        price = self.client.get_ticker_price(ticker)
                        min_ = min(self.__stop_loss, self.__take_profit)
                        max_ = max(self.__stop_loss, self.__take_profit)
                        if (not min_ < price < max_) and (not self.__exit_order__):
                            if self._predict != utils.EXIT:
                                self.__exit_order__ = True
                                utils.logger.info('exit lot')
                                prediction['predict'] = 'Exit'
                                prediction['currency close'] = price
                                index = f'{self.ticker}, {time.ctime()}'
                                if print_out:
                                    print(index, prediction)
                                utils.logger.info(f"trading prediction exit in sleeping at {index}: {prediction}")
                                self.realtie_returns[index] = prediction
                                if trading_on_client:
                                    self.client.exit_last_order()
                                    utils.logger.info('client exit lot')
                    if not (time.time() < (__now__ + sleeping_time)):
                        __now__ += sleeping_time
                        break

        except Exception as e:
            self.prepare_realtime = False
            if print_out:
                print(e)

    def log_data(self):
        self.fig.update_yaxes(row=1, col=1, type='log')

    def log_deposit(self):
        self.fig.update_yaxes(row=2, col=1, type='log')

    def backtest(self,
                 deposit: float = 10_000,
                 credit_leverage: float = 1,
                 bet: float = np.inf,
                 commission: float = 0,
                 plot: bool = True,
                 print_out: bool = True,
                 show: bool = True,
                 log_profit_calc: bool = True,
                 *args,
                 **kwargs):
        """

        :param deposit: start deposit.
        :param credit_leverage: trading leverage. 1 = none.
        :param bet: fixed bet to quick_trade--. np.inf = all moneys.
        :param commission: percentage commission (0 -- 100).
        :param plot: plotting.
        :param print_out: printing.
        :param show: showing figure.
        :param log_profit_calc: calculating profit logarithmic

        """

    def load_model(self, path: str):
        self.model = load_model(path)

    def set_client(self, your_client: TradingClient):
        """
        :param your_client: TradingClient object

        """
        self.client = your_client

    def convert_signal(self, old=utils.SELL, new=utils.EXIT):
        for pos, val in enumerate(self.returns):
            if val == old:
                self.returns[pos] = new

    def set_open_stop_and_take(self,
                               take_profit: float = None,
                               stop_loss: float = None,
                               set_stop: bool = True,
                               set_take: bool = True):
        """
        :param set_take: create new take profits.
        :param set_stop: create new stop losses.
        :param take_profit: take profit in points
        :param stop_loss: stop loss in points

        """
        self.take_profit: float = take_profit
        self.stop_loss: float = stop_loss
        take_flag: float = np.inf
        stop_flag: float = np.inf
        open_flag: float = np.inf
        self.open_lot_prices: list = []
        if set_stop:
            self.stop_losses = []
        if set_take:
            self.take_profits = []
        closes = self.df['Close'].values
        for sig, close, seted in zip(self.returns, closes, utils.set_(self.returns)):
            if seted is not np.nan:
                self.open_price = close
                if set_take or set_stop:
                    ts = self.__get_stop_take(sig)
                if set_take:
                    take_flag = ts['take']
                if set_stop:
                    stop_flag = ts['stop']
            self.open_lot_prices.append(self.open_price)
            if set_take:
                self.take_profits.append(take_flag)
            if set_stop:
                self.stop_losses.append(stop_flag)

    def set_credit_leverages(self, credit_lev: float):
        """
        Sets the leverage for bets.

        """
        self.credit_leverages = [credit_lev for i in range(len(self.df['Close']))]

    def _window_(self, column: str, n: int = 2):
        return utils.get_window(self.df[column].values, n)

    def find_pip_bar(self, min_diff_coef: float = 2, body_coef: float = 10):
        ret = []
        flag = utils.EXIT
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
                ret.append(flag)
            else:
                ret.append(flag)
        self.returns = ret
        return ret

    def find_DBLHC_DBHLC(self):
        ret = [utils.EXIT]
        flag = utils.EXIT

        flag_stop_loss = np.inf
        self.stop_losses = [flag_stop_loss]

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

            ret.append(flag)
            self.stop_losses.append(flag_stop_loss)
        self.returns = ret
        self.set_open_stop_and_take(set_take=False, set_stop=False)
        return ret

    def find_TBH_TBL(self):
        ret = [utils.EXIT]
        flag = utils.EXIT
        for e, (high, low, open_, close) in enumerate(
                zip(
                    self._window_('High'), self._window_('Low'),
                    self._window_('Open'), self._window_('Close')), 1):
            if high[0] == high[1]:
                flag = utils.BUY
            elif low[0] == low[1]:
                flag = utils.SELL
            ret.append(flag)
        self.returns = ret
        return ret

    def find_PPR(self):
        ret = [utils.EXIT] * 2
        flag = utils.EXIT
        for e, (high, low, opn, close) in enumerate(
                zip(
                    self._window_('High', 3), self._window_('Low', 3),
                    self._window_('Open', 3), self._window_('Close', 3)), 1):
            if min(low) == low[1] and close[1] < close[2] and high[2] < high[0]:
                flag = utils.BUY
            elif max(high
                     ) == high[1] and close[2] < close[1] and low[2] > low[0]:
                flag = utils.SELL
            ret.append(flag)
        self.returns = ret
        return ret

    def is_doji(self):
        """
        :returns: list of booleans.

        """
        ret = []
        for close, open_ in zip(self.df['Close'].values,
                                self.df['Open'].values):
            if close == open_:
                ret.append(True)
            else:
                ret.append(False)
        return ret
