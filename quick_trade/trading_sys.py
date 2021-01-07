"""
Trading project:
    - testing
    - trading

"""

# !/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)


# TODO:
#   add inner class with non-trading utils
#   debug neural networks
#   do quick-trade-tuner better

import itertools
import random
import time
import typing
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import quick_trade.brokers as brokers
import ta
import ta.momentum
import ta.others
import ta.trend
import ta.volatility
import ta.volume
import talib
from plotly.graph_objs import Line
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from quick_trade import utils
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from binance.exceptions import BinanceAPIException


class Trader(object):
    """
    algo-trading system.
    ticker:   |     str      |  ticker/symbol of chart
    df:       |   dataframe  |  data of chart
    interval: |     str      |  interval of df.
    one of:
    1m    30m    3h    1M
    2m    45m    4h    3M
    3m    1h     1d    6M
    5m    90m    3d
    15m   2h     1w

    """
    profit_calculate_coef: float
    returns: utils.PREDICT_TYPE_LIST = []
    df: pd.DataFrame
    ticker: str
    interval: str
    __exit_order__: bool = False
    _old_predict: str = 'Exit'
    _regression_inputs: int
    stop_loss: float
    take_profit: float
    open_price: float
    scaler: MinMaxScaler
    history: Dict[str, List[float]]
    training_set: Tuple[np.ndarray, np.ndarray]
    trades: int = 0
    profits: int = 0
    losses: int = 0
    stop_losses: List[float]
    take_profits: List[float]
    credit_leverages: List[float]
    deposit_history: List[float]
    year_profit: float
    linear: np.ndarray
    info: str
    backtest_out_no_drop: pd.DataFrame
    backtest_out: pd.DataFrame
    open_lot_prices: List[float]
    realtime_returns: Dict[str, Dict[str, typing.Union[str, float]]]
    client: brokers.TradingClient
    __last_stop_loss: float
    __last_take_profit: float
    model: Sequential
    returns_strategy: List[float]

    def __init__(self,
                 ticker: str = 'AAPL',
                 df: pd.DataFrame = pd.DataFrame(),
                 interval: str = '1d',
                 rounding: int = 50,
                 *args,
                 **kwargs):
        df_ = round(df, rounding)
        self.df = df_.reset_index(drop=True)
        self.ticker = ticker
        self.interval = interval
        if interval == '1m':
            self.profit_calculate_coef = 1 / (60 * 24 * 365)
        elif interval == '2m':
            self.profit_calculate_coef = 1 / (30 * 24 * 365)
        elif interval == '3m':
            self.profit_calculate_coef = 1 / (20 * 24 * 365)
        elif interval == '5m':
            self.profit_calculate_coef = 1 / (12 * 24 * 365)
        elif interval == '15m':
            self.profit_calculate_coef = 1 / (4 * 24 * 365)
        elif interval == '30m':
            self.profit_calculate_coef = 1 / (2 * 24 * 365)
        elif interval == '45m':
            self.profit_calculate_coef = 1 / (32 * 365)
        elif interval == '1h':
            self.profit_calculate_coef = 1 / (24 * 365)
        elif interval == '90m':
            self.profit_calculate_coef = 1 / (18 * 365)
        elif interval == '2h':
            self.profit_calculate_coef = 1 / (12 * 365)
        elif interval == '3h':
            self.profit_calculate_coef = 1 / (8 * 365)
        elif interval == '4h':
            self.profit_calculate_coef = 1 / (6 * 365)
        elif interval == '1d':
            self.profit_calculate_coef = 1 / 365
        elif interval == '3d':
            self.profit_calculate_coef = 1 / (365 / 3)
        elif interval == '1w':
            self.profit_calculate_coef = 1 / 52
        elif interval == '1M':
            self.profit_calculate_coef = 1 / 12
        elif interval == '3M':
            self.profit_calculate_coef = 1 / 4
        elif interval == '6M':
            self.profit_calculate_coef = 1 / 2
        else:
            raise ValueError(f'I N C O R R E C T   I N T E R V A L; {interval}')
        self._regression_inputs = utils.REGRESSION_INPUTS
        self.__exit_order__ = False

    def __repr__(self):
        return 'trader'

    def _get_attr(self, attr: str):
        return getattr(self, attr)

    @classmethod
    def _get_this_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def kalman_filter(self,
                      df: pd.Series,
                      iters: int = 5,
                      plot: bool = True,
                      *args,
                      **kwargs) -> pd.Series:
        filtered: np.ndarray
        k_filter: KalmanFilter = KalmanFilter()
        df: pd.Series
        i: int
        filtered = k_filter.filter(np.array(df))[0]
        for i in range(iters):
            filtered = k_filter.smooth(filtered)[0]
        if plot:
            self.fig.add_trace(
                Line(
                    name='kalman filter',
                    y=filtered.T[0],
                    line=dict(width=utils.SUB_LINES_WIDTH)), 1, 1)
        return pd.Series(filtered.T[0])

    def scipy_filter(self,
                     df: pd.Series,
                     window_length: int = 101,
                     polyorder: int = 3,
                     plot: bool = True,
                     **scipy_savgol_filter_kwargs) -> pd.Series:
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
        return pd.Series(filtered)

    def bull_power(self, periods: int) -> np.ndarray:
        EMA = ta.trend.ema_indicator(self.df['Close'], periods)
        return np.array(self.df['High']) - EMA

    def tema(self, periods: int, *args, **kwargs) -> pd.Series:
        """
        :rtype: pd.Series
        """
        ema = ta.trend.ema_indicator(self.df['Close'], periods)
        ema2 = ta.trend.ema_indicator(ema, periods)
        ema3 = ta.trend.ema_indicator(ema2, periods)
        return pd.Series(3 * ema.values - 3 * ema2.values + ema3.values)

    def get_linear(self, dataset) -> np.ndarray:
        """
        linear data. mean + (mean diff * n)
        """
        mean_diff: float
        data: pd.DataFrame = pd.DataFrame(dataset)

        mean: float = float(data.mean())
        mean_diff = float(data.diff().mean())
        start: float = mean - (mean_diff * (len(data) / 2))
        end: float = start + (mean - start) * 2

        length: int = len(data)
        return_list: List[float] = []
        mean_diff = (end - start) / length
        i: int
        for i in range(length):
            return_list.append(start + mean_diff * i)
        utils.logger.debug(f'in linear: self.mean_diff={mean_diff}')
        return np.array(return_list)

    def __get_stop_take(self, sig: utils.PREDICT_TYPE) -> Dict[str, float]:
        """
        calculating stop loss and take profit.
        sig:        |     int     |  signal to sell/buy/exit:
            EXIT -- exit.
            BUY -- buy.
            SELL -- sell.
        """

        _stop_loss: float
        take: float
        if self.stop_loss is not np.inf:
            _stop_loss = self.stop_loss / 10_000 * self.open_price
        else:
            _stop_loss = np.inf
        if self.take_profit is not np.inf:
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
            if self.take_profit is not np.inf:
                take = self.open_price
            if self.stop_loss is not np.inf:
                _stop_loss = self.open_price
        utils.logger.debug(
            f'stop loss: {_stop_loss} ({self.stop_loss} pips), take profit: {take} ({self.take_profit} pips)')

        return {'stop': _stop_loss,
                'take': take}

    def strategy_diff(self, frame_to_diff: pd.Series, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
        """
        frame_to_diff:  |   pd.Series  |  example:  Trader.df['Close']
        """
        self.returns = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        self.set_open_stop_and_take()
        return self.returns

    def strategy_buy_hold(self, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.BUY for _ in range(len(self.df))]
        self.set_open_stop_and_take()
        return self.returns

    def strategy_2_sma(self,
                       slow: int = 100,
                       fast: int = 30,
                       plot: bool = True,
                       *args,
                       **kwargs) -> utils.PREDICT_TYPE_LIST:
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
        return self.returns

    def strategy_3_sma(self,
                       slow: int = 100,
                       mid: int = 26,
                       fast: int = 13,
                       plot: bool = True,
                       *args,
                       **kwargs) -> utils.PREDICT_TYPE_LIST:
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

        self.set_open_stop_and_take()
        return self.returns

    def strategy_3_ema(self,
                       slow: int = 46,
                       mid: int = 21,
                       fast: int = 3,
                       plot: bool = True,
                       *args,
                       **kwargs) -> utils.PREDICT_TYPE_LIST:
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
        self.set_open_stop_and_take()
        return self.returns

    def strategy_macd(self,
                      slow: int = 100,
                      fast: int = 30,
                      *args,
                      **kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        diff = ta.trend.macd_diff(self.df['Close'], slow, fast)

        for j in diff:
            if j > 0:
                self.returns.append(utils.BUY)
            elif 0 > j:
                self.returns.append(utils.SELL)
            else:
                self.returns.append(utils.EXIT)
        self.set_open_stop_and_take()
        return self.returns

    def strategy_exp_diff(self,
                          period: int = 70,
                          plot: bool = True,
                          *args,
                          **kwargs) -> utils.PREDICT_TYPE_LIST:
        exp: pd.Series = self.tema(period)
        self.strategy_diff(exp)
        if plot:
            self.fig.add_trace(
                Line(
                    name=f'EMA{period}',
                    y=exp.values.T[0],
                    line=dict(width=utils.SUB_LINES_WIDTH)), 1, 1)

        self.set_open_stop_and_take()
        return self.returns

    def strategy_rsi(self,
                     minimum: float = 20,
                     maximum: float = 80,
                     max_mid: float = 75,
                     min_mid: float = 35,
                     *args,
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

        self.set_open_stop_and_take()
        return self.returns

    def strategy_parabolic_SAR(self, plot: bool = True, *args, **sar_kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = []
        sar: ta.trend.PSARIndicator = ta.trend.PSARIndicator(self.df['High'], self.df['Low'],
                                                             self.df['Close'], **sar_kwargs)
        sardown: np.ndarray = sar.psar_down().values
        sarup: np.ndarray = sar.psar_up().values
        self.stop_losses = list(sar.psar().values)

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
        self.set_open_stop_and_take(set_stop=False)
        return self.returns

    def strategy_macd_histogram_diff(self,
                                     slow: int = 23,
                                     fast: int = 12,
                                     *args,
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
        self.set_open_stop_and_take()
        return self.returns

    def strategy_regression_model(self, plot: bool = True, *args, **kwargs):  # TODO: fix
        self.returns = [utils.EXIT for i in range(self._regression_inputs - 1)]
        data_to_pred: np.ndarray = np.array(
            utils.get_window(np.array([self.df['Close'].values]).T, self._regression_inputs)
        ).T

        for e, data in enumerate(data_to_pred):
            data_to_pred[e] = self.scaler.fit_transform(data)
        data_to_pred = data_to_pred.T

        predictions = itertools.chain.from_iterable(
            self.model.predict(data_to_pred))
        predictions = pd.Series(predictions)
        frame = predictions
        predictions = self.strategy_diff(predictions)
        frame = self.scaler.inverse_transform(frame.values.T).T
        self.returns = [*self.returns, *predictions]
        nans = itertools.chain.from_iterable([(np.nan,) * self._regression_inputs])
        filt = (*nans, *frame.T[0])
        if plot:
            self.fig.add_trace(
                Line(
                    name='predict',
                    y=filt,
                    line=dict(width=utils.SUB_LINES_WIDTH, color=utils.CYAN)),
                row=1,
                col=1)
        self.set_open_stop_and_take()
        return self.returns, filt

    def get_network_regression(self,
                               dataframes: typing.Iterable[pd.DataFrame],
                               inputs: int = utils.REGRESSION_INPUTS,
                               network_save_path: str = './model_regression.h5',
                               **fit_kwargs) -> Sequential:  # TODO: fix
        """based on
        https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
        """

        self.model = Sequential()
        self.model.add(
            LSTM(units=50, return_sequences=True, input_shape=(inputs, 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data: np.ndarray
        for df in dataframes:
            scaled_data = self.prepare_scaler(df)
            train_data = scaled_data[0:len(scaled_data), :]
            x_train = []
            y_train = []
            for i in range(inputs, len(train_data)):
                x_train.append(train_data[i - inputs:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train,
                                 (x_train.shape[0], x_train.shape[1], 1))
            self.model.fit(x_train, y_train, **fit_kwargs)
        self.model.save(network_save_path)
        return self.model

    def prepare_scaler(self,
                       dataframe: pd.DataFrame,
                       regression_net: bool = True) -> np.ndarray:  # TODO: fix
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data: pd.DataFrame
        dataset: np.ndarray
        if regression_net:
            data = dataframe.filter(['Close'])
            dataset = data.values
        else:
            dataset = dataframe.values
        scaled_data: np.ndarray = self.scaler.fit_transform(dataset)
        return scaled_data

    def get_trained_network(self,
                            dataframes: typing.Iterable[pd.DataFrame],
                            filter_: str = 'kalman_filter',
                            filter_kwargs: Dict[str, typing.Any] = {},
                            optimizer: str = 'adam',
                            loss: str = 'mse',
                            metrics: typing.Iterable[str] = ['mse'],
                            network_save_path: str = './model_predicting.h5',
                            **fit_kwargs) -> Tuple[Sequential,
                                                   Dict[str, List[float]],
                                                   Tuple[np.ndarray, np.ndarray]]:  # TODO: fix
        """
        getting trained neural network to trading.
        dataframes:  | typing.Iterable[pd.DataFrame] |   list of pandas dataframes with columns:
            'High'
            'Low'
            'Open'
            'Close'
            'Volume'
        optimizer:    |        str         |   optimizer for .compile of network.
        filter_:      |        str         |    filter for training.
        filter_kwargs:|       dict         |    named arguments for the filter.
        loss:         |        str         |   loss for .compile of network.
        metrics:      |typing.Iterable[str]|   metrics for .compile of network:
            standard: ['acc']
        fit_kwargs:   |  *named arguments* |   arguments to .fit of network.
        returns:
            (tensorflow model,
            history of training,
            (input training data, output train data))
        """

        list_input: List[pd.DataFrame] = []
        list_output: List[int] = []
        flag: pd.DataFrame = self.df

        df: pd.DataFrame
        filter_kwargs['plot'] = False
        for df in dataframes:
            self.df = df
            all_ta = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close',
                                            'Volume', True)
            output1 = self.strategy_diff(
                self._get_attr(filter_)(**filter_kwargs))

            for output in output1:
                list_output.append(output[0])
            list_input.append(
                pd.DataFrame(
                    self.prepare_scaler(
                        pd.DataFrame(all_ta), regression_net=False)))
        self.df = flag
        del flag
        input_df: pd.DataFrame = pd.concat(list_input, axis=0).dropna(1)

        input_train_array: np.ndarray = input_df.values
        output_train_array: np.ndarray = np.array([list_output]).T

        self.model = Sequential()
        self.model.add(
            Dense(20, input_dim=len(input_train_array[0]), activation='tanh'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, 'sigmoid'))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.history = self.model.fit(input_train_array, output_train_array, **fit_kwargs)
        self.training_set = (input_train_array, output_train_array)
        self.model.save(network_save_path)
        return self.model, self.history, self.training_set

    def strategy_random_pred(self, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = [random.choice([utils.EXIT, utils.SELL, utils.BUY]) for i in range(len(self.df))]
        self.set_open_stop_and_take()
        return self.returns

    def strategy_with_network(self,
                              rounding: int = 0,
                              _rounding_prediction_func=round,
                              *args,
                              **kwargs) -> utils.PREDICT_TYPE_LIST:  # TODO: fix
        """
        :param rounding: rounding degree for _rounding_prediction_func
        :param _rounding_prediction_func: A function that will be used to round off the neural network result.
        """
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        all_ta: np.ndarray = ta.add_all_ta_features(self.df, "Open", 'High', 'Low',
                                                    'Close', "Volume", True).values
        preds: np.ndarray = self.model.predict(scaler.fit_transform(all_ta))
        for e, i in enumerate(preds):
            preds[e] = _rounding_prediction_func(i[0], rounding)
        self.returns = list(preds)
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
        self.stop_losses = list(st.get_supertrend())
        self.returns = list(st.get_supertrend_strategy_returns())
        self.stop_losses[0] = np.inf if self.returns[0] == utils.SELL else -np.inf
        self.set_open_stop_and_take(set_stop=False)
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
        return self.returns

    def get_heikin_ashi(self, df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
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

    def strategy_ichimoku(self,
                          tenkansen: int = 9,
                          kijunsen: int = 26,
                          senkouspan: int = 52,
                          chinkouspan: int = 26,
                          stop_loss_plus: float = 40.0,
                          plot: bool = True,
                          *args,
                          **kwargs) -> utils.PREDICT_TYPE_LIST:
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
            self.stop_losses = [np.inf] * chinkouspan
            for e, (close, tenkan, kijun, A, B) in enumerate(zip(
                    prices.values[chinkouspan:],
                    tenkan_sen[chinkouspan:],
                    kinjun_sen[chinkouspan:],
                    senkou_span_a[chinkouspan:],
                    senkou_span_b[chinkouspan:],
            ), chinkouspan):
                max_cloud = max((A, B))
                min_cloud = min((A, B))

                stop_loss_adder = stop_loss_plus * (close / 10_000)

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
                if trade == utils.BUY:
                    self.stop_losses.append(min_cloud - stop_loss_adder)
                else:
                    self.stop_losses.append(max_cloud + stop_loss_adder)
        self.set_open_stop_and_take(set_take=True,
                                    set_stop=False)
        self.set_open_stop_and_take()
        return self.returns

    def inverse_strategy(self, swap_tpop_take: bool = True, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
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
            self.stop_losses, self.take_profits = self.take_profits, self.stop_losses
        return self.returns

    def backtest(self,
                 deposit: float = 10_000.0,
                 bet: float = np.inf,
                 commission: float = 0.0,
                 plot: bool = True,
                 print_out: bool = True,
                 column: str = 'Close',
                 show: bool = True,
                 *args,
                 **kwargs) -> pd.DataFrame:
        """
        testing the strategy.
        :param deposit: start deposit.
        :param bet: fixed bet to quick_trade. np.inf = all moneys.
        :param commission: percentage commission (0 -- 100).
        :param plot: plotting.
        :param print_out: printing.
        :param column: column of dataframe to backtest
        :param show: show the graph
        returns: pd.DataFrame with data of test
        """

        exit_take_stop: bool
        no_order: bool
        stop_loss: float
        take_profit: float
        seted: List[typing.Any]
        diff: float
        lin_calc_df: pd.DataFrame
        price: float
        credit_lev: float

        start_bet: float = bet
        data_column: pd.Series = self.df[column]
        self.deposit_history = [deposit]
        seted_ = utils.set_(self.returns)
        self.trades = 0
        self.profits = 0
        self.losses = 0
        moneys_open_bet: float = deposit
        money_start: float = deposit
        oldsig = utils.EXIT
        start_commission: float = commission

        e: int
        sig: utils.PREDICT_TYPE
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
                    commission = start_commission * 2
                else:
                    commission = start_commission
                if bet > deposit:
                    bet = deposit
                open_price = price
                bet *= credit_lev
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
                deposit += bet * diff / open_price
            no_order = exit_take_stop
            self.deposit_history.append(deposit)
            oldsig = sig

        self.linear = self.get_linear(self.deposit_history)
        lin_calc_df = pd.DataFrame(self.linear)
        mean_diff = float(lin_calc_df.diff().mean())
        self.year_profit = mean_diff / self.profit_calculate_coef
        self.year_profit = (self.year_profit / money_start) * 100
        self.winrate = (self.profits / self.trades) * 100
        self.info = f"""losses: {self.losses}
trades: {self.trades}
profits: {self.profits}
mean year percentage profit: {self.year_profit}%
winrate: {self.winrate}%"""
        utils.logger.info(f'trader info: {self.info}')
        if print_out:
            print(self.info)
        self.returns_strategy = list(pd.Series(self.deposit_history).diff().values)
        self.backtest_out_no_drop = pd.DataFrame(
            (self.deposit_history, self.stop_losses, self.take_profits, self.returns,
             self.open_lot_prices, data_column, self.linear, self.returns_strategy),
            index=[
                f'deposit ({column})', 'stop loss', 'take profit',
                'predictions', 'open deal/lot', column,
                f"linear deposit data ({column})",
                "returns"
            ]).T
        self.backtest_out = self.backtest_out_no_drop.dropna()
        if plot:
            loc: pd.Series = self.df[column]
            self.fig.add_trace(
                Line(
                    y=self.backtest_out_no_drop['returns'].values,
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
                    y=self.take_profits,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.GREEN),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='take profit'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.stop_losses,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.RED),
                    opacity=utils.STOP_TAKE_OPN_ALPHA,
                    name='stop loss'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.open_lot_prices,
                    line=dict(width=utils.TAKE_STOP_OPN_WIDTH, color=utils.BLUE),
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
            preds: Dict[str, List[typing.Union[int, float]]] = {'sellind': [],
                                                                'exitind': [],
                                                                'buyind': [],
                                                                'bprice': [],
                                                                'sprice': [],
                                                                'eprice': []}
            for e, i in enumerate(utils.set_(self.returns)):
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
                          mode: str = 'minimalist',
                          *args,
                          **kwargs) -> utils.PREDICT_TYPE_LIST:
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

                    first_returns = [1,1,0,0,2,0,2,2,0,0,1]

                    second_returns = [1,2,2,2,2,2,0,0,0,0,1]

                        [1,1,1,1,2,2,2,2,0,0,1]

                mode = 'super':
                    ...

                    first_returns = [1,1,1,2,2,2,0,0,1]

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
            self.returns = self.__maximalist(first_returns, second_returns)
        elif mode == 'super':
            self.returns = self.__collide_super(first_returns, second_returns)
        else:
            raise ValueError('I N C O R R E C T   M O D E')
        return self.returns

    @staticmethod
    def __maximalist(returns1: utils.PREDICT_TYPE_LIST,
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
    def __collide_super(l1, l2) -> utils.PREDICT_TYPE_LIST:
        return_list: utils.PREDICT_TYPE_LIST = []
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

    def multi_strategy_collider(self, *strategies, mode: str = 'minimalist') -> utils.PREDICT_TYPE_LIST:
        self.strategy_collider(strategies[0], strategies[1], mode=mode)
        if len(strategies) >= 3:
            for ret in strategies[2:]:
                self.strategy_collider(self.returns, ret, mode=mode)
        return self.returns

    def get_trading_predict(self,
                            trading_on_client: bool = False,
                            bet_for_trading_on_client: float = np.inf,
                            second_symbol_of_ticker: str = 'None',
                            rounding_bet: int = 4,
                            coin_lotsize_division=True,
                            *args,
                            **kwargs
                            ) -> Dict[str, typing.Union[str, float]]:
        """
        predict and trading.

        :param coin_lotsize_division: If for your api you specify the size of the bet in a coin, which is not in which you have a deposit, specify this parameter in the value: True. Otherwise: False, in Binance's case this is definitely the first case (True). If errors occur, try specifying the first ticker symbol instead of the second.
        :param rounding_bet: maximum permissible accuracy with your api. Bigger than 0
        :param second_symbol_of_ticker: BTCUSDT -> USDT, for calculate bet. As deposit
        :param trading_on_client: trading on real client
        :param bet_for_trading_on_client: standard: all deposit
        :return: dict with prediction
        """

        credit_leverage: float = self.credit_leverages[-1]
        _moneys_: float
        bet: float
        close: np.ndarray = self.df["Close"].values
        cond: bool

        # get prediction
        predict = self.returns[-1]
        predict = utils.convert_signal_str(predict)
        if self.__exit_order__ and self._old_predict == predict:
            predict = 'Exit'
        if predict != 'Exit':
            self.__exit_order__ = False

        # trading
        self.__last_stop_loss = self.stop_losses[-1]
        self.__last_take_profit = self.take_profits[-1]
        if self._old_predict != predict:
            utils.logger.info(f'open lot {predict}')
            self.open_price = close[-1]
            if trading_on_client:

                if predict == 'Exit':
                    self.client.exit_last_order()

                else:
                    _moneys_ = self.client.get_balance_ticker(second_symbol_of_ticker)
                    ticker_price = self.client.get_ticker_price(self.ticker)
                    if coin_lotsize_division:
                        _moneys_ /= ticker_price
                    if bet_for_trading_on_client is not np.inf:
                        bet = bet_for_trading_on_client / ticker_price
                    else:
                        bet = _moneys_
                    if bet > _moneys_:
                        bet = _moneys_
                    self.client.exit_last_order()

                    self.client.order_create(predict,
                                             self.ticker,
                                             bet,
                                             credit_leverage=credit_leverage,
                                             rounding_bet=rounding_bet,
                                             _moneys_=_moneys_)
                    self.__exit_order__ = False
        return {
            'predict': predict,
            'open lot price': self.open_price,
            'stop loss': self.__last_stop_loss,
            'take profit': self.__last_take_profit,
            'currency close': close[-1]
        }

    def realtime_trading(self,
                         strategy,
                         ticker: str = 'AAPL',
                         get_data_kwargs: Dict[str, typing.Any] = {},
                         sleeping_time: float = 60.0,
                         print_out: bool = True,
                         trading_on_client: bool = False,
                         bet_for_trading_on_client: float = np.inf,
                         second_symbol_of_ticker: str = 'None',
                         rounding_bet: int = 4,
                         coin_lotsize_division: bool = True,
                         ignore_exceptions: bool = True,
                         print_exc: bool = True,
                         *strategy_args,
                         **strategy_kwargs):
        """
        :param print_exc: print binance exceptions in while loop
        :param ignore_exceptions: ignore binance exceptions in while loop
        :param coin_lotsize_division: If for your api you specify the size of the bet in a coin, which is not in which you have a deposit, specify this parameter in the value: True. Otherwise: False, in Binance's case this is definitely the first case (True). If errors occur, try specifying the first ticker symbol instead of the second.
        :param ticker: ticker for trading.
        :param strategy: trading strategy.
        :param get_data_kwargs: named arguments to self.client.get_data WITHOUT TICKER.
        :param sleeping_time: sleeping time / timeframe in seconds.
        :param print_out: printing.
        :param trading_on_client: trading on client
        :param bet_for_trading_on_client: trading bet, standard: all deposit
        :param second_symbol_of_ticker: USDUAH -> UAH
        :param rounding_bet: maximum accuracy for trading
        :param strategy_kwargs: named arguments to -strategy.
        :param strategy_args: arguments to -strategy.
        """

        self.realtime_returns = {}
        self.ticker = ticker
        try:
            __now__ = time.time()
            while True:
                try:
                    self.df = self.client.get_data(self.ticker, **get_data_kwargs).reset_index(drop=True)
                    strategy(*strategy_args, **strategy_kwargs)

                    prediction = self.get_trading_predict(
                        trading_on_client=trading_on_client,
                        bet_for_trading_on_client=bet_for_trading_on_client,
                        second_symbol_of_ticker=second_symbol_of_ticker,
                        rounding_bet=rounding_bet,
                        coin_lotsize_division=coin_lotsize_division)

                    index = f'{self.ticker}, {time.ctime()}'
                    utils.logger.info(f"trading prediction at {index}: {prediction}")
                    if print_out:
                        print(index, prediction)
                    self.realtime_returns[index] = prediction
                    while True:
                        if not self.__exit_order__:
                            price = self.client.get_ticker_price(ticker)
                            min_ = min(self.__last_stop_loss, self.__last_take_profit)
                            max_ = max(self.__last_stop_loss, self.__last_take_profit)
                            if (not (min_ < price < max_)) and prediction["predict"] != 'Exit':
                                self.__exit_order__ = True
                                utils.logger.info('exit lot')
                                prediction['predict'] = 'Exit'
                                prediction['currency close'] = price
                                index = f'{self.ticker}, {time.ctime()}'
                                utils.logger.info(f"trading prediction exit in sleeping at {index}: {prediction}")
                                if print_out:
                                    print(f"trading prediction exit in sleeping at {index}: {prediction}")
                                self.realtime_returns[index] = prediction
                                if trading_on_client:
                                    self.client.exit_last_order()
                        if not (time.time() < (__now__ + sleeping_time)):
                            self._old_predict = utils.convert_signal_str(self.returns[-1])
                            __now__ += sleeping_time
                            break
                except BinanceAPIException as binance_exc:
                    if ignore_exceptions:
                        if print_exc:
                            print(binance_exc)
                        utils.logger.error('error :(', exc_info=True)
                        continue
                    else:
                        raise binance_exc

        except Exception as e:
            utils.logger.critical('error :(', exc_info=True)
            raise e

    def log_data(self, *args, **kwargs):
        self.fig.update_yaxes(row=1, col=1, type='log')
        utils.logger.info('trader log data')

    def log_deposit(self, *args, **kwargs):
        self.fig.update_yaxes(row=2, col=1, type='log')
        utils.logger.info('trader log deposit')

    def log_returns(self, *args, **kwargs):
        self.fig.update_yaxes(row=3, col=1, type='log')
        utils.logger.info('trader log returns')

    def load_model(self, path: str, *args, **kwargs):
        self.model = load_model(path)

    def set_client(self, your_client: brokers.TradingClient, *args, **kwargs):
        """
        :param your_client: trading client
        """
        self.client = your_client
        utils.logger.info('trader set client')

    def convert_signal(self,
                       old: utils.PREDICT_TYPE = utils.SELL,
                       new: utils.PREDICT_TYPE = utils.EXIT,
                       *args,
                       **kwargs) -> utils.PREDICT_TYPE_LIST:
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
                               set_take: bool = True,
                               *args,
                               **kwargs):
        """
        :param set_take: create new take profits.
        :param set_stop: create new stop losses.
        :param take_profit: take profit in points
        :param stop_loss: stop loss in points
        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        take_flag: float = np.inf
        stop_flag: float = np.inf
        self.open_lot_prices = []
        if set_stop:
            self.stop_losses = []
        if set_take:
            self.take_profits = []
        closes: np.ndarray = self.df['Close'].values
        sig: utils.PREDICT_TYPE
        close: float
        seted: utils.SETED_TYPE
        ts: Dict[str, float]
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
        utils.logger.debug(f'trader stop loss: {stop_loss}, trader take profit: {take_profit}')

    def set_credit_leverages(self, credit_lev: float = 1.0, *args, **kwargs):
        """
        Sets the leverage for bets.
        :param credit_lev: leverage in points
        """
        self.credit_leverages = [credit_lev for i in range(len(self.df['Close']))]
        utils.logger.debug(f'trader credit leverage: {credit_lev}')

    def _window_(self,
                 column: str,
                 n: int = 2,
                 *args,
                 **kwargs) -> List[typing.Any]:
        return utils.get_window(self.df[column].values, n)

    def find_pip_bar(self,
                     min_diff_coef: float = 2.0,
                     body_coef: float = 10.0,
                     *args,
                     **kwargs) -> utils.PREDICT_TYPE_LIST:
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
        self.set_open_stop_and_take()
        return self.returns

    def find_DBLHC_DBHLC(self, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
        self.returns = [utils.EXIT]
        flag: utils.PREDICT_TYPE = utils.EXIT

        flag_stop_loss: float = np.inf
        self.stop_losses = [flag_stop_loss]
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
            self.stop_losses.append(flag_stop_loss)
        self.set_open_stop_and_take(set_take=False, set_stop=False)
        return self.returns

    def find_TBH_TBL(self, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
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
        self.set_open_stop_and_take()
        return self.returns

    def find_PPR(self, *args, **kwargs) -> utils.PREDICT_TYPE_LIST:
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
        self.set_open_stop_and_take()
        return self.returns

    def is_doji(self, *args, **kwargs) -> List[bool]:
        """
        :returns: list of booleans.
        """
        ret: List[bool] = []
        for close, open_ in zip(self.df['Close'].values,
                                self.df['Open'].values):
            if close == open_:
                ret.append(True)
            else:
                ret.append(False)
        self.set_open_stop_and_take()
        return ret

    def find_all_talib_patterns(self, *args, **kwargs):
        open_ = self.df['Open']
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']

        patterns = map(utils.ta_lib_collider_all,
                       [
                           talib.CDL2CROWS(open_, high, low, close),
                           talib.CDL3BLACKCROWS(open_, high, low, close),
                           talib.CDL3INSIDE(open_, high, low, close),
                           talib.CDL3LINESTRIKE(open_, high, low, close),
                           talib.CDL3OUTSIDE(open_, high, low, close),
                           talib.CDL3STARSINSOUTH(open_, high, low, close),
                           talib.CDL3WHITESOLDIERS(open_, high, low, close),
                           talib.CDLABANDONEDBABY(open_, high, low, close),
                           talib.CDLADVANCEBLOCK(open_, high, low, close),
                           talib.CDLBELTHOLD(open_, high, low, close),
                           talib.CDLBREAKAWAY(open_, high, low, close),
                           talib.CDLCLOSINGMARUBOZU(open_, high, low, close),
                           talib.CDLCONCEALBABYSWALL(open_, high, low, close),
                           talib.CDLCOUNTERATTACK(open_, high, low, close),
                           talib.CDLDARKCLOUDCOVER(open_, high, low, close),
                           talib.CDLDOJI(open_, high, low, close),
                           talib.CDLDOJISTAR(open_, high, low, close),
                           talib.CDLDRAGONFLYDOJI(open_, high, low, close),
                           talib.CDLENGULFING(open_, high, low, close),
                           talib.CDLEVENINGDOJISTAR(open_, high, low, close),
                           talib.CDLEVENINGSTAR(open_, high, low, close),
                           talib.CDLGAPSIDESIDEWHITE(open_, high, low, close),
                           talib.CDLGRAVESTONEDOJI(open_, high, low, close),
                           talib.CDLHAMMER(open_, high, low, close),
                           talib.CDLHANGINGMAN(open_, high, low, close),
                           talib.CDLHARAMI(open_, high, low, close),
                           talib.CDLHARAMICROSS(open_, high, low, close),
                           talib.CDLHIGHWAVE(open_, high, low, close),
                           talib.CDLHIKKAKE(open_, high, low, close),
                           talib.CDLHIKKAKEMOD(open_, high, low, close),
                           talib.CDLHOMINGPIGEON(open_, high, low, close),
                           talib.CDLIDENTICAL3CROWS(open_, high, low, close),
                           talib.CDLINNECK(open_, high, low, close),
                           talib.CDLINVERTEDHAMMER(open_, high, low, close),
                           talib.CDLKICKING(open_, high, low, close),
                           talib.CDLKICKINGBYLENGTH(open_, high, low, close),
                           talib.CDLLADDERBOTTOM(open_, high, low, close),
                           talib.CDLLONGLEGGEDDOJI(open_, high, low, close),
                           talib.CDLLONGLINE(open_, high, low, close),
                           talib.CDLMARUBOZU(open_, high, low, close),
                           talib.CDLMATCHINGLOW(open_, high, low, close),
                           talib.CDLMATHOLD(open_, high, low, close),
                           talib.CDLMORNINGDOJISTAR(open_, high, low, close),
                           talib.CDLMORNINGSTAR(open_, high, low, close),
                           talib.CDLONNECK(open_, high, low, close),
                           talib.CDLPIERCING(open_, high, low, close),
                           talib.CDLRICKSHAWMAN(open_, high, low, close),
                           talib.CDLRISEFALL3METHODS(open_, high, low, close),
                           talib.CDLSEPARATINGLINES(open_, high, low, close),
                           talib.CDLSHOOTINGSTAR(open_, high, low, close),
                           talib.CDLSHORTLINE(open_, high, low, close),
                           talib.CDLSPINNINGTOP(open_, high, low, close),
                           talib.CDLSTALLEDPATTERN(open_, high, low, close),
                           talib.CDLSTICKSANDWICH(open_, high, low, close),
                           talib.CDLTAKURI(open_, high, low, close),
                           talib.CDLTASUKIGAP(open_, high, low, close),
                           talib.CDLTHRUSTING(open_, high, low, close),
                           talib.CDLTRISTAR(open_, high, low, close),
                           talib.CDLUNIQUE3RIVER(open_, high, low, close),
                           talib.CDLUPSIDEGAP2CROWS(open_, high, low, close),
                           talib.CDLXSIDEGAP3METHODS(open_, high, low, close)
                       ]
                       )

        patterns = map(utils.anti_set_, patterns)
        self.set_open_stop_and_take()
        return self.multi_strategy_collider(*patterns, mode='super')
