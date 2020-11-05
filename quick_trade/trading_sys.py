#!/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)
import random
import time

from plotly.graph_objs import Line
import ta
import ta.volatility
from binance.client import Client
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from quick_trade.utils import *
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


# я понял свои ошибки
# я пишу на русском комментарии
# и они без смысла
# и нет комментов к коду

class TradingClient(Client):
    ordered = False

    def get_ticker_price(self, ticker):
        return float(self.get_symbol_ticker(symbol=ticker)['price'])

    @staticmethod
    def get_data(ticker=None, interval=None, **get_kw):
        return get_binance_data(ticker, interval, **get_kw)

    def new_order_buy(self, ticker=None, quantity=None, credit_leverage=None):
        self.__side__ = 'Buy'
        self.quantity = quantity
        self.ticker = ticker
        self.order = self.order_market_buy(symbol=ticker, quantity=quantity)
        self.order_id = self.order['orderId']
        self.ordered = True

    def new_order_sell(self, ticker=None, quantity=None, credit_leverage=None):
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

    def get_balance_ticker(self, ticker):
        for asset in self.get_account()['balances']:
            if asset['asset'] == ticker:
                return float(asset['free'])


class Strategies(object):
    """

    basic class for PatternFinder (but without patterns).

    """

    def __init__(self,
                 ticker='AAPL',
                 df: pd.DataFrame = np.nan,
                 interval='1d',
                 rounding=5,
                 *args,
                 **kwargs):
        df_ = round(df, rounding)
        self.__first__ = True
        self.__rounding__ = rounding
        diff = digit(df_['Close'].diff().values)[1:]
        self.__oldsig = EXIT
        self.diff = [EXIT, *diff]
        self.df = df_.reset_index(drop=True)
        self.ticker = ticker
        self.interval = interval
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
        self.__inputs = INPUTS
        self.__exit_order__ = False

    def __repr__(self):
        return 'trader'

    @classmethod
    def _get_this_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def kalman_filter(self,
                      df='self.df["Close"]',
                      iters=5,
                      plot=True,
                      *args,
                      **kwargs):
        k_filter = KalmanFilter()
        if isinstance(df, str):
            df = eval(df)
        filtered = k_filter.filter(np.array(df))[0]
        for i in range(iters):
            filtered = k_filter.smooth(filtered)[0]
        if plot:
            self.fig.add_trace(
                Line(
                    name='kalman filter',
                    y=filtered.T[0],
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)
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
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)
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

        if sig == BUY:
            _stop_loss = self.open_price - _stop_loss
            take = self.open_price + take
        elif sig == SELL:
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
        frame_to_diff:  |   pd.DataFrame  |  example:  Strategies.df['Close']

        """
        if isinstance(frame_to_diff, str):
            frame_to_diff = eval(frame_to_diff)
        self.returns = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        return self.returns

    def strategy_buy_hold(self, *args, **kwargs):
        self.returns = [BUY for _ in range(len(self.df))]
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
                    line=dict(width=SUB_LINES_WIDTH, color=G)), 1, 1)
            self.fig.add_trace(
                Line(
                    name=f'SMA{slow}',
                    y=SMA2.values,
                    line=dict(width=SUB_LINES_WIDTH, color=R)), 1, 1)

        for SMA13, SMA26 in zip(SMA1, SMA2):
            if SMA26 < SMA13:
                return_list.append(BUY)
            elif SMA13 < SMA26:
                return_list.append(SELL)
            else:
                return_list.append(EXIT)
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
            for SMA, Co, name in zip([SMA1, SMA2, SMA3], [G, B, R],
                                     [fast, mid, slow]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=SMA.values,
                        line=dict(width=SUB_LINES_WIDTH, color=Co)), 1, 1)

        for SMA13, SMA26, SMA100 in zip(SMA1, SMA2, SMA3):
            if SMA100 < SMA26 < SMA13:
                return_list.append(BUY)
            elif SMA100 > SMA26 > SMA13:
                return_list.append(SELL)
            else:
                return_list.append(EXIT)

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
                                     [G, B, R], [slow, mid, fast]):
                self.fig.add_trace(
                    Line(
                        name=f'SMA{name}',
                        y=ema,
                        line=dict(width=SUB_LINES_WIDTH, color=Co)), 1, 1)

        for EMA1, EMA2, EMA3 in zip(ema3, ema21, ema46):
            if EMA1 > EMA2 > EMA3:
                return_list.append(BUY)
            elif EMA1 < EMA2 < EMA3:
                return_list.append(SELL)
            else:
                return_list.append(EXIT)
        self.returns = return_list
        return return_list

    def strategy_macd(self, slow=100, fast=30, *args, **kwargs):
        return_list = []
        lavel = ta.trend.macd_signal(self.df['Close'], slow, fast)
        macd = ta.trend.macd(self.df['Close'], slow, fast)

        for j, k in zip(lavel.values, macd.values):
            if j > k:
                return_list.append(SELL)
            elif k > j:
                return_list.append(BUY)
            else:
                return_list.append(EXIT)
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
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)

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
        flag = EXIT

        for val, diff in zip(rsi.values, rsi.diff().values):
            if val < minimum and diff > 0 and val is not pd.NA:
                return_list.append(BUY)
                flag = BUY
            elif val > maximum and diff < 0 and val is not pd.NA:
                return_list.append(SELL)
                flag = SELL
            elif flag == BUY and val < max_mid:
                flag = EXIT
                return_list.append(EXIT)
            elif flag == SELL and val > min_mid:
                flag = EXIT
                return_list.append(EXIT)
            else:
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
                return_list.append(BUY)
            elif MACD < 0 and RSI < rsi_level:
                return_list.append(SELL)
            else:
                return_list.append(EXIT)
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
                        name='SAR', y=SAR_, line=dict(width=SUB_LINES_WIDTH)),
                    1, 1)
        for price, up, down in zip(
                list(self.df['Close'].values), list(sarup), list(sardown)):
            numup = np.nan_to_num(up, nan=-9999)
            numdown = np.nan_to_num(down, nan=-9999)
            if numup != -9999:
                return_list.append(BUY)
            elif numdown != -9999:
                return_list.append(SELL)
            else:
                return_list.append(EXIT)
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
        return_list = digit(histogram.diff().values)
        self.returns = return_list
        return return_list

    def strategy_regression_model(self, plot=True, *args, **kwargs):
        return_list = []
        for i in range(self.__inputs - 1):
            return_list.append(EXIT)
        data_to_pred = np.array(
            get_window(np.array([self.df['Close'].values]).T, self.__inputs))

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
                    line=dict(width=SUB_LINES_WIDTH, color=C)),
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
        flag = EXIT
        bollinger = ta.volatility.BollingerBands(self.df['Close'], fillna=True, *bollinger_args, **bollinger_kwargs)

        mid_ = bollinger.bollinger_mavg()
        upper = bollinger.bollinger_hband()
        lower = bollinger.bollinger_lband()
        if plot:
            for TR, name in zip([upper, mid_, lower], ['upper band', 'mid band', 'lower band']):
                self.fig.add_trace(Line(y=TR, name=name, line=dict(width=SUB_LINES_WIDTH)), col=1, row=1)
        for close, up, mid, low in zip(self.df['Close'].values,
                                       upper,
                                       mid_,
                                       lower):
            if close <= low:
                flag = BUY
            if close >= up:
                flag = SELL

            if to_mid:
                if flag == SELL and close <= mid:
                    flag = EXIT
                if flag == BUY and close >= mid:
                    flag = EXIT
            return_list.append(flag)
        self.returns = return_list

    def inverse_strategy(self, *args, **kwargs):
        """
        makes signals inverse:

        buy = sell.
        sell = buy.
        exit = exit.

        """

        return_list = []
        for signal_key in self.returns:
            if signal_key == BUY:
                return_list.append(SELL)
            elif signal_key == SELL:
                return_list.append(BUY)
            else:
                return_list.append(EXIT)
        self.returns = return_list
        return return_list

    def basic_backtest(self,
                       deposit=10_000,
                       bet=None,
                       commission: float = 0.0,
                       plot=True,
                       print_out=True,
                       column='Close',
                       *args,
                       **kwargs):
        """
        testing the strategy.


        deposit:         | int, float. | start deposit.

        bet:             | int, float, | fixed bet to quick_trade. None = all moneys.

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
        start_bet = bet

        data_column = self.df[column]
        self.deposit_history = [deposit]
        seted_ = set_(self.returns)
        self.trades = 0
        self.profits = 0
        self.losses = 0
        moneys_open_bet = deposit
        money_start = deposit

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
                if start_bet is None:
                    bet = deposit
                open_price = price
                bet *= credit_lev

                def coefficient(difference):
                    return bet * difference / open_price

                deposit -= bet * (commission / 100)
                self.trades += 1
                if deposit > moneys_open_bet:
                    self.profits += 1
                elif deposit < moneys_open_bet:
                    self.losses += 1
                moneys_open_bet = deposit

            if not e:
                diff = 0
            if min(stop_loss, take_profit) < price < max(stop_loss, take_profit):
                diff = data_column[e] - data_column[e - 1]
            else:
                if sig == BUY and price >= take_profit:
                    diff = take_profit - data_column[e - 1]
                elif sig == BUY and price <= stop_loss:
                    diff = stop_loss - data_column[e - 1]
                elif sig == SELL and price >= stop_loss:
                    diff = stop_loss - data_column[e - 1]
                elif sig == SELL and price <= take_profit:
                    diff = take_profit - data_column[e - 1]
                else:
                    diff = 0

            if sig == SELL:
                diff = -diff
            elif sig == EXIT:
                diff = 0

            deposit += coefficient(diff)
            self.deposit_history.append(deposit)

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
        self.backtest_out = pd.DataFrame(
            (self.deposit_history, self.stop_losses, self.take_profits, self.returns,
             self.open_lot_prices, data_column, self.linear),
            index=[
                f'deposit ({column})', 'stop loss', 'take profit',
                'predictions', 'open deal/lot', column,
                f"linear deposit data ({column})"
            ]).T.dropna()
        return self.backtest_out

    def set_pyplot(self,
                   height=900,
                   width=1300,
                   template='plotly_dark',
                   row_heights=None,
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
            title_text=title_ax, row=2, col=1, color=TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='M O N E Y S', row=2, col=1, color=TEXT_COLOR)
        self.fig.update_yaxes(
            title_text='D A T A', row=1, col=1, color=TEXT_COLOR)

    def strategy_collider(self,
                          first_func=nothing,
                          second_func=nothing,
                          args_first_func=(),
                          args_second_func=(),
                          mode='minimalist',
                          *args,
                          **kwargs):
        """
        first_func:      |  trading strategy  |   strategy to combine.

        standard: nothing.

        example:  Strategies.strategy_macd.

        second_func:     |  trading strategy  |   strategy to combine.

        standard: nothing.

        args_first_func: |    tuple, list     |   arguments to first function.

        args_second_func:|    tuple, list     |   arguments to second function.

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

        Strategies.strategy_collider(
           Strategies.strategy_2_sma,
           Strategies.strategy_3_ema,
           (30, 10)
        )

        or:

        Strategies.strategy_collider(Strategies.strategy_2_sma,
                             Strategies.strategy_2_sma,
                             (300, 200),
                             (200, 100))
                                  =
                   Strategies.strategy_3_sma(300, 200, 100)


        """

        first_returns = first_func(*args_first_func)
        second_returns = second_func(*args_second_func)
        return_list = []
        if mode == 'minimalist':
            for ret1, ret2 in zip(first_returns, second_returns):
                if ret1 == ret2:
                    return_list.append(ret1)
                else:
                    return_list.append(EXIT)
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
        flag = EXIT
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
        for first, sec in zip(set_(l1), set_(l2)):
            if first is not np.nan and sec is not np.nan and first is not sec:
                return_list.append(EXIT)
            elif first is sec:
                return_list.append(first)
            elif first is np.nan:
                return_list.append(sec)
            else:
                return_list.append(first)
        return anti_set_(return_list)

    def get_trading_predict(self,
                            take_profit=None,
                            stop_loss=None,
                            inverse=False,
                            trading_on_client=False,
                            bet_for_trading_on_client='all depo',
                            credit_leverage=None,
                            second_symbol_of_ticker=None,
                            can_sell=False,
                            rounding_bet=4,
                            *args,
                            **kwargs):
        """
        :param rounding_bet: maximum permissible accuracy with your api
        :param second_symbol_of_ticker: BTCUSDT -> USDT
        :param can_sell: use order sell (client)
        :param credit_leverage: credit leverage for trading on client
        :param take_profit: take profit(float)
        :param stop_loss: stop loss(float)
        :param inverse: inverting(bool)
        :param trading_on_client: trading on real client (boll)
        :param bet_for_trading_on_client: (float or "all depo")
        :return: dict with prediction
        """

        def convert():
            nonlocal predict
            if predict == BUY:
                predict = 'Buy'
            elif predict == SELL:
                predict = 'Sell'
            elif predict == EXIT:
                predict = 'Exit'

        if trading_on_client:
            _moneys_ = self.client.get_balance_ticker(second_symbol_of_ticker)
            if bet_for_trading_on_client == 'all depo':
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
        if take_profit is None:
            self.take_profit = np.inf
        else:
            self.take_profit = take_profit

        if stop_loss is None:
            self.stop_loss = np.inf
        else:
            self.stop_loss = stop_loss

        if inverse:
            self.inverse_strategy()

        predicts = self.returns
        predict = predicts[len(predicts) - 1]
        if self.__exit_order__:
            predict = EXIT
        if not can_sell:
            self.convert_signal(SELL, EXIT)

        cond = "_predict" not in self.__dir__()
        if not cond:
            cond = self._predict != predict
        close = self.df['Close'].values

        if cond:
            convert()
            logger.info(f'open lot {predict}')
            if trading_on_client:
                if predict == 'Buy':
                    if not self.__first__:
                        self.client.exit_last_order()
                        logger.info('client exit')
                    self.client.new_order_buy(self.ticker, bet, credit_leverage=credit_leverage)
                    logger.info('client buy')
                    self.__exit_order__ = False
                    self.__first__ = False

                if predict == 'Sell':
                    if not self.__first__:
                        self.client.exit_last_order()
                        logger.info('client exit')
                    if can_sell:
                        self.client.new_order_sell(self.ticker, bet, credit_leverage=credit_leverage)
                        logger.info('client sell')
                    self.__first__ = False
                    self.__exit_order__ = False

                if predict == 'Exit':
                    if not self.__first__:
                        self.client.exit_last_order()
                        logger.info('client exit')
                        self.__exit_order__ = True

            predict = predicts[len(predicts) - 1]
            for sig, close_ in zip(self.returns[::-1],
                                   self.df['Close'].values[::-1]):
                if sig != EXIT:
                    self.open_price = close_
                    break
            rets = self.__get_stop_take(predict)
            self._predict = predict
            self.__stop_loss = rets['stop']
            self.__take_profit = rets['take']
        convert()
        curr_price = close[len(close) - 1]
        return {
            'predict': predict,
            'open lot price': self.open_price,
            'stop loss': self.__stop_loss,
            'take profit': self.__take_profit,
            'currency close': curr_price
        }

    def realtime_trading(self,
                         ticker,
                         strategy,
                         get_data_kwargs=None,
                         sleeping_time=60,
                         print_out=True,
                         take_profit=None,
                         stop_loss=None,
                         inverse=False,
                         trading_on_client=False,
                         bet_for_trading_on_client='all depo',
                         can_sell_client=False,
                         second_symbol_of_ticker=None,
                         rounding_bet=4,
                         **strategy_kwargs):
        """
        ticker:           |             str             |  ticker for trading.

        strategy:         |   Strategies.some_strategy  |  trading strategy.

        get_data_kwargs:  |             dict            |  named arguments to self.client.get_data WITHOUT TICKER.

        **strategy_kwargs:|             kwargs          |  named arguments to <<strategy>>.

        sleeping_time:    |             int             |  sleeping time.

        print_out:        |             bool            |  printing.

        """

        if get_data_kwargs is None:
            get_data_kwargs = dict()
        if get_data_kwargs is None:
            get_data_kwargs = dict()
        self._ret = {}
        self.ticker = ticker
        __now__ = time.time()
        try:
            while True:
                self.prepare_realtime = True
                self.df = self.client.get_data(ticker, **get_data_kwargs).reset_index(drop=True)
                strategy(**strategy_kwargs)
                prediction = self.get_trading_predict(
                    take_profit=take_profit, stop_loss=stop_loss,
                    inverse=inverse, trading_on_client=trading_on_client,
                    bet_for_trading_on_client=bet_for_trading_on_client,
                    can_sell=can_sell_client,
                    second_symbol_of_ticker=second_symbol_of_ticker,
                    rounding_bet=rounding_bet)
                # едрить, жизнь такая крутаяяяяяяяяя
                index = f'{self.ticker}, {time.ctime()}'
                if print_out:
                    print(index, prediction)
                logger.info(f"trading prediction at {index}: {prediction}")
                self._ret[index] = prediction
                while True:
                    if not self.__exit_order__:
                        price = self.client.get_ticker_price(ticker)
                        min_ = min(self.__stop_loss, self.__take_profit)
                        max_ = max(self.__stop_loss, self.__take_profit)
                        if (not min_ < price < max_) and (not self.__exit_order__):
                            if self._predict != EXIT:
                                self.__exit_order__ = True
                                logger.info('exit lot')
                                prediction['predict'] = 'Exit'
                                prediction['currency close'] = price
                                index = f'{self.ticker}, {time.ctime()}'
                                if print_out:
                                    print(index, prediction)
                                logger.info(f"trading prediction exit in sleeping at {index}: {prediction}")
                                self._ret[index] = prediction
                                if trading_on_client:
                                    self.client.exit_last_order()
                    if not (time.time() < (__now__ + sleeping_time)):
                        __now__ = time.time()
                        break
            # как-же меня это всё достало, мне просто хочется заработать и жить спокойно
            # но нет, блин, нужно было этим разрабам из python-binance сморозить такую дичь
            # представляю, что-бы было, если-б юзал официальное API.
            # мне ещё географию переписывать в тетрадь
            # я просто хочу хорошо жить, никого не напрягаяя.

        except Exception as e:
            self.prepare_realtime = False
            if print_out:
                print(e)

    def log_data(self):
        self.fig.update_yaxes(row=1, col=1, type='log')

    def log_deposit(self):
        self.fig.update_yaxes(row=2, col=1, type='log')

    def backtest(self,
                 deposit=10_000,
                 credit_leverage=1,
                 bet: int = None,
                 commission: float = 0,
                 stop_loss: int = None,
                 take_profit: int = None,
                 plot=True,
                 print_out=True,
                 show=True,
                 log_profit_calc=True,
                 *args,
                 **kwargs):
        """
        testing the strategy.


        deposit:         | int, float. | start deposit.

        credit_leverage: | int, float. | tradeing leverage. 1 = none.

        bet:             | int, float, | fixed bet to quick_trade--. None = all moneys.

        commission:      | int, float. | percentage commission (0 -- 100).

        stop_loss:       | int, float. | stop loss in points.

        take_profit:     | int, float. | take profit in points.

        plot:            |    bool.    | plotting.

        print_out:       |    bool.    | printing.

        show:            |    bool.    | showing figure.



        returns: 2 pd.DataFrames with data of:
            signals,
            deposit (high, low, open, close)'
            stop loss,
            take profit,
            linear deposit,
            price (high, low, open, close),
            open lot price.

        """

        money_start = deposit
        loc = self.df['Close'].values
        _df_flag = self.df
        self.df = pd.concat(
            [
                self.df['Open'], self.df['High'],
                self.df['Low'], self.df['Close']
            ],
            axis=1)
        __df = self.df
        df = inverse_4_col_df(self.df, ['Open', 'High', 'Low', 'Close'])

        self.df = df
        _returns_flag = self.returns

        self.returns = []
        for pred in _returns_flag:
            for column in range(4):
                self.returns.append(pred)
        for i in range(3):
            self.returns.insert(0, EXIT)
        del self.returns[:-3]

        self.basic_backtest(
            deposit=deposit,
            credit_leverage=credit_leverage,
            bet=bet,
            commission=commission,
            stop_loss=stop_loss,
            take_profit=take_profit,
            plot=False,
            print_out=False)

        self.df = _df_flag
        self.returns = _returns_flag

        rets = self.backtest_out

        def __4_div(obj, columns):
            frame = list(np.array(obj))[::4]
            frame = pd.DataFrame(frame, columns=columns).reset_index()
            del frame['index']
            return frame

        deposit_df = rets['deposit (Close)'].values
        deposit_df = to_4_col_df(deposit_df, 'deposit Open',
                                 'deposit High', 'deposit Low', 'deposit Close')

        self.linear = pd.DataFrame(
            self.linear_(deposit_df['deposit Close'].values),
            columns=['deposit Close linear'])

        self.open_lot_prices = __4_div(
            self.open_lot_prices, columns=['open lot price'])
        self.take_profits = __4_div(self.take_profits, columns=['take profit'])
        self.stop_losses = __4_div(self.stop_losses, columns=['stop loss'])
        self.backtest_out = pd.concat(
            [
                __df.reset_index(),
                deposit_df.reset_index(),
                pd.DataFrame(self.returns, columns=['predictions'
                                                    ]), self.stop_losses,
                self.take_profits, self.open_lot_prices, self.linear
            ],
            axis=1)
        del __4_div
        del self.backtest_out['index']
        self.backtest_out_no_drop = self.backtest_out
        self.backtest_out = self.backtest_out.dropna()
        if log_profit_calc:
            _log = np.log
        else:
            _log = nothing
        self.lin_calc = self.linear_(
            MinMaxScaler(
                feature_range=(
                    min(self.deposit_history),
                    max(self.deposit_history)
                )
            ).fit_transform(
                np.array([_log(deposit_df['deposit Close'].values)]).T
            ).T[0])
        lin_calc_df = pd.DataFrame(self.lin_calc)
        self.mean_diff = float(lin_calc_df.diff().mean())
        self.year_profit = self.mean_diff / self.profit_calculate_coef + money_start
        self.year_profit = ((self.year_profit - money_start) / money_start) * 100
        self.info = (
            f"L O S S E S: {self.losses}\n"
            f"T R A D E S: {self.trades}\n"
            f"P R O F I T S: {self.profits}\n"
            f"M E A N   Y E A R   P E R C E N T A G E   R O F I T: {self.year_profit}%\n"
        )

        MIN = []
        MAX = []
        CLOSE = []
        OPEN = []
        for C__, O, H, L in deposit_df.values:
            sorted_ = sorted([C__, O, H, L])
            MIN.append(min(sorted_))
            MAX.append(max(sorted_))
            CLOSE.append(sorted_[1])
            OPEN.append(sorted_[2])
        deposit_df = pd.DataFrame({'deposit Close': CLOSE,
                                   'deposit Open': OPEN,
                                   'deposit High': MAX,
                                   'deposit Low': MIN})

        if print_out:
            print(self.info)
        if plot:
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
                    y=self.take_profits.values.T[0],
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=G),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='take profit'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.stop_losses.values.T[0],
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=R),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='stop loss'),
                row=1,
                col=1)
            self.fig.add_trace(
                Line(
                    y=self.open_lot_prices.values.T[0],
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=B),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='open lot'),
                row=1,
                col=1)
            self.fig.add_candlestick(
                low=deposit_df['deposit Low'],
                high=deposit_df['deposit High'],
                close=deposit_df['deposit Close'],
                open=deposit_df['deposit Open'],
                increasing_line_color=DEPO_COLOR_UP,
                decreasing_line_color=DEPO_COLOR_DOWN,
                name=f'D E P O S I T  (S T A R T: ${money_start})',
                row=2,
                col=1)
            self.fig.add_trace(
                Line(y=self.linear.values.T[0], name='L I N E A R'),
                row=2,
                col=1)
            for e, i in enumerate(set_(self.returns)):
                if i == SELL:
                    self.fig.add_scatter(
                        name='Sell',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#FF0000'),
                        marker=dict(
                            symbol='triangle-down',
                            size=SCATTER_SIZE,
                            opacity=SCATTER_ALPHA))
                elif i == BUY:
                    self.fig.add_scatter(
                        name='Buy',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#00FF00'),
                        marker=dict(
                            symbol='triangle-up',
                            size=SCATTER_SIZE,
                            opacity=SCATTER_ALPHA))
                elif i == EXIT:
                    self.fig.add_scatter(
                        name='Exit',
                        y=[loc[e]],
                        x=[e],
                        row=1,
                        col=1,
                        line=dict(color='#2a00ff'),
                        marker=dict(
                            symbol='triangle-left',
                            size=SCATTER_SIZE,
                            opacity=SCATTER_ALPHA))
            self.fig.update_layout(xaxis_rangeslider_visible=False)
            if show:
                self.fig.show()

        return self.backtest_out

    def load_model(self, path):
        self.model = load_model(path)

    def set_client(self, your_client):
        """
        :param your_client: TradingClient object

        """
        self.client = your_client

    def convert_signal(self, old=SELL, new=EXIT):
        for pos, val in enumerate(self.returns):
            if val == old:
                self.returns[pos] = new

    def set_stop_and_take(self,
                          take_profit=None,
                          stop_loss=None):
        """
        :param take_profit: take profit in points
        :param stop_loss: stop loss in points

        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.open_lot_prices = []
        self.stop_losses = []
        self.take_profits = []
        closes = self.df['Close'].values
        for sig, close, seted in zip(self.returns, closes, set_(closes)):
            if seted is not np.nan:
                self.open_price = close
            self.open_lot_prices.append(self.open_price)
            ts = self.__get_stop_take(sig)
            self.take_profits.append(ts['take'])
            self.stop_losses.append(ts['stop'])

    def set_credit_leverages(self, credit_lev):
        """
        Sets the leverage for bets.

        """
        self.credit_leverages = [credit_lev for i in range(len(self.df['Close']))]


class PatternFinder(Strategies):
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


    step 1:
    creating a plotter

    step 2:
    using strategy

    step 3:
    backtesting a strategy


    example:

    trader = PatternFinder(df=mydf)

    trader.set_pyplot()

    trader.strategy_3_sma()

    return = trader.backtest(credit_leverage=40)

    """

    def _window_(self, column, n=2):
        return get_window(self.df[column].values, n)

    def find_pip_bar(self, min_diff_coef=2, body_coef=10):
        ret = []
        flag = EXIT
        for e, (high, low, open_price, close) in enumerate(
                zip(self.df['High'], self.df['Low'], self.df['Open'],
                    self.df['Close']), 1):
            body = abs(open_price - close)
            shadow_high = high - max(open_price, close)
            shadow_low = min(open_price, close) - low
            if body < (max(shadow_high, shadow_low) * body_coef):
                if shadow_low > (shadow_high * min_diff_coef):
                    ret.append(BUY)
                    flag = BUY
                elif shadow_high > (shadow_low * min_diff_coef):
                    ret.append(SELL)
                    flag = SELL
                else:
                    ret.append(flag)
            else:
                ret.append(flag)
        self.returns = ret
        return ret

    def find_DBLHC_DBHLC(self):
        ret = [EXIT]
        flag = EXIT
        for e, (high, low, open_pr, close) in enumerate(
                zip(
                    self._window_('High'), self._window_('Low'),
                    self._window_('Open'), self._window_('Close')), 1):
            if low[0] == low[1] and close[1] > high[0]:
                ret.append(BUY)
                flag = BUY
            elif high[0] == high[1] and close[0] > low[1]:
                ret.append(SELL)
                flag = SELL
            else:
                ret.append(flag)
        self.returns = ret
        return ret

    def find_TBH_TBL(self):
        ret = [EXIT]
        flag = EXIT
        for e, (high, low, open_, close) in enumerate(
                zip(
                    self._window_('High'), self._window_('Low'),
                    self._window_('Open'), self._window_('Close')), 1):
            if high[0] == high[1]:
                ret.append(BUY)
                flag = BUY
            elif low[0] == low[1]:
                ret.append(SELL)
                flag = SELL
            else:
                ret.append(flag)
        self.returns = ret
        return ret

    def find_PPR(self):
        ret = [EXIT, EXIT]
        flag = EXIT
        for e, (high, low, opn, close) in enumerate(
                zip(
                    self._window_('High', 3), self._window_('Low', 3),
                    self._window_('Open', 3), self._window_('Close', 3)), 1):
            if min(low) == low[1] and close[1] < close[2] and high[2] < high[0]:
                ret.append(BUY)
                flag = BUY
            elif max(high
                     ) == high[1] and close[2] < close[1] and low[2] > low[0]:
                ret.append(SELL)
                flag = SELL
            else:
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
