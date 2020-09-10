#!/usr/bin/python
# -*- coding: utf-8 -*-
# used ta by Darío López Padial (Bukosabino https://github.com/bukosabino/ta)
import time

import plotly.graph_objects as go
import ta
from binance.client import Client
from plotly.subplots import make_subplots as sub_make
from pykalman import KalmanFilter
from quick_trade.utils import *
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


class TradingClient(Client):
    ordered = False

    def get_ticker_price(self, ticker):
        return float(self.get_symbol_ticker(symbol=ticker)['price'])

    def get_data(self, ticker=None, interval=None, **get_kw):
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

    basic class for PatternFinder.
    please use PatternFinder class.

    """

    def __init__(self,
                 ticker='AAPL',
                 days_undo=100,
                 df=np.nan,
                 interval='1d',
                 rounding=5,
                 *args,
                 **kwargs):
        df_ = round(df, rounding)
        self.first = True
        self.rounding = rounding
        diff = digit(df_['Close'].diff().values)[1:]
        self.diff = [EXIT, *diff]
        self.df = df_.reset_index(drop=True)
        self.drop = 0
        self.ticker = ticker
        self.days_undo = days_undo
        self.interval = interval
        if interval == '1m':
            self.profit_calculate_coef = 1 / 60 / 24 / 365
        elif interval == '1d':
            self.profit_calculate_coef = 1 / 365
        else:
            raise ValueError('I N C O R R E C T   I N T E R V A L')
        self.inputs = INPUTS

    def __repr__(self):
        return 'trader'

    def __iter__(self):
        try:
            return iter(list(self.backtest_out.values))
        except AttributeError:
            raise ValueError('D O   B A C K T E S T')

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
                go.Line(
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
                go.Line(
                    name='savgol filter',
                    y=filtered,
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)
        return pd.DataFrame(filtered)

    def bull_power(self, periods):
        EMA = ta.trend.ema(self.df['Close'], periods)
        ret = np.array(self.df['High']) - EMA
        return ret

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
        ret = []
        mean_diff = (end - start) / length
        for i in range(length):
            ret.append(start + mean_diff * i)
        self.mean_diff = mean_diff
        return np.array(ret)

    def get_stop_take(self, sig):
        """
        calculating stop loss and take profit.


        sig:        |     int     |  signsl to sell/buy/exit:
            2 -- exit.
            1 -- buy.
            0 -- sell.

        """

        if self.stop_loss is not np.inf:
            _stop_loss = self.stop_loss / 10_000 * self.open_price
        else:
            _stop_loss = np.inf
        if self.take_profit is not np.inf:
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
            if self.take_profit is not np.inf:
                take = self.open_price
            if self.stop_loss is not np.inf:
                _stop_loss = self.open_price

        return {'stop': _stop_loss, 'take': take}

    def strategy_diff(self, frame_to_diff, *args, **kwargs):
        """
        frame_to_diff:  |   pd.DataFrame  |  example:  Strategies.df['Close']

        """
        if isinstance(frame_to_diff, str):
            frame_to_diff = eval(frame_to_diff)
        ret = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        self.returns = ret
        return ret

    def strategy_buy_hold(self, *args, **kwargs):
        self.returns = [BUY for _ in range(len(self.df))]
        return self.returns

    def strategy_2_sma(self, slow=100, fast=30, plot=True, *args, **kwargs):
        ret = []
        SMA1 = ta.trend.sma(self.df['Close'], fast)
        SMA2 = ta.trend.sma(self.df['Close'], slow)
        if plot:
            self.fig.add_trace(
                go.Line(
                    name=f'SMA{fast}',
                    y=SMA1.values,
                    line=dict(width=SUB_LINES_WIDTH, color=G)), 1, 1)
            self.fig.add_trace(
                go.Line(
                    name=f'SMA{slow}',
                    y=SMA2.values,
                    line=dict(width=SUB_LINES_WIDTH, color=R)), 1, 1)

        for SMA13, SMA26 in zip(SMA1, SMA2):
            if SMA26 < SMA13:
                ret.append(BUY)
            elif SMA13 < SMA26:
                ret.append(SELL)
            else:
                ret.append(EXIT)
        self.returns = ret
        return ret

    def strategy_3_sma(self,
                       slow=100,
                       mid=26,
                       fast=13,
                       plot=True,
                       *args,
                       **kwargs):
        ret = []
        SMA1 = ta.trend.sma(self.df['Close'], fast)
        SMA2 = ta.trend.sma(self.df['Close'], mid)
        SMA3 = ta.trend.sma(self.df['Close'], slow)

        if plot:
            for SMA, C, name in zip([SMA1, SMA2, SMA3], [G, B, R],
                                    [fast, mid, slow]):
                self.fig.add_trace(
                    go.Line(
                        name=f'SMA{name}',
                        y=SMA.values,
                        line=dict(width=SUB_LINES_WIDTH, color=C)), 1, 1)

        for SMA13, SMA26, SMA100 in zip(SMA1, SMA2, SMA3):
            if SMA100 < SMA26 < SMA13:
                ret.append(BUY)
            elif SMA100 > SMA26 > SMA13:
                ret.append(SELL)
            else:
                ret.append(EXIT)

        self.returns = ret
        return ret

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
        ret = []

        if plot:
            for ema, C, name in zip([ema3.values, ema21.values, ema46.values],
                                    [G, B, R], [slow, mid, fast]):
                self.fig.add_trace(
                    go.Line(
                        name=f'SMA{name}',
                        y=ema,
                        line=dict(width=SUB_LINES_WIDTH, color=C)), 1, 1)

        for EMA1, EMA2, EMA3 in zip(ema3, ema21, ema46):
            if EMA1 > EMA2 > EMA3:
                ret.append(BUY)
            elif EMA1 < EMA2 < EMA3:
                ret.append(SELL)
            else:
                ret.append(EXIT)
        self.returns = ret
        return ret

    def strategy_macd(self, slow=100, fast=30, *args, **kwargs):
        ret = []
        lavel = ta.trend.macd_signal(self.df['Close'], slow, fast)
        macd = ta.trend.macd(self.df['Close'], slow, fast)

        for j, k in zip(lavel.values, macd.values):
            if j > k:
                ret.append(SELL)
            elif k > j:
                ret.append(BUY)
            else:
                ret.append(EXIT)
        self.returns = ret
        return ret

    def strategy_exp_diff(self, period=70, plot=True, *args, **kwargs):
        exp = self.tema(period)
        ret = self.strategy_diff(exp)
        if plot:
            self.fig.add_trace(
                go.Line(
                    name=f'EMA{period}',
                    y=exp.values.T[0],
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)

        self.returns = ret
        return ret

    def strategy_rsi(self,
                     min=20,
                     max=80,
                     max_mid=75,
                     min_mid=35,
                     *args,
                     **rsi_kwargs):
        rsi = ta.momentum.rsi(self.df['Close'], **rsi_kwargs)
        ret = []
        flag = EXIT

        for val, diff in zip(rsi.values, rsi.diff().values):
            if val < min and diff > 0 and val is not pd.NA:
                ret.append(BUY)
                flag = BUY
            elif val > max and diff < 0 and val is not pd.NA:
                ret.append(SELL)
                flag = SELL
            elif flag == BUY and val < max_mid:
                flag = EXIT
                ret.append(EXIT)
            elif flag == SELL and val > min_mid:
                flag = EXIT
                ret.append(EXIT)
            else:
                ret.append(flag)

        ret = ret[self.drop:]
        self.returns = ret
        return ret

    def strategy_macd_rsi(self,
                          mac_slow=26,
                          mac_fast=12,
                          rsi_level=50,
                          rsi_kwargs=dict(),
                          *args,
                          **macd_kwargs):
        ret = []
        macd = ta.trend.macd(self.df['Close'], mac_slow, mac_fast,
                             **macd_kwargs)
        rsi = ta.momentum.rsi(self.df['Close'], **rsi_kwargs)
        for MACD, RSI in zip(macd.values, rsi.values):
            if MACD > 0 and RSI > rsi_level:
                ret.append(BUY)
            elif MACD < 0 and RSI < rsi_level:
                ret.append(SELL)
            else:
                ret.append(EXIT)
        ret = ret[self.drop:]
        self.returns = ret
        return ret

    def strategy_parabolic_SAR(self, plot=True, *args, **sar_kwargs):
        ret = []
        sar = ta.trend.PSARIndicator(self.df['High'], self.df['Low'],
                                     self.df['Close'], **sar_kwargs)
        sardown = sar.psar_down()[self.drop:].values
        sarup = sar.psar_up()[self.drop:].values

        if plot:
            for SAR_ in (sarup, sardown):
                self.fig.add_trace(
                    go.Line(
                        name='SAR', y=SAR_, line=dict(width=SUB_LINES_WIDTH)),
                    1, 1)
        for price, up, down in zip(
                list(self.df['Close'].values), list(sarup), list(sardown)):
            numup = np.nan_to_num(up, nan=-9999)
            numdown = np.nan_to_num(down, nan=-9999)
            if numup != -9999:
                ret.append(BUY)
            elif numdown != -9999:
                ret.append(SELL)
            else:
                ret.append(EXIT)
        self.returns = ret
        return ret

    def strategy_macd_histogram_diff(self,
                                     slow=23,
                                     fast=12,
                                     *args,
                                     **macd_kwargs):
        _MACD_ = ta.trend.MACD(self.df['Close'], slow, fast, **macd_kwargs)
        signal_ = _MACD_.macd_signal()
        macd_ = _MACD_.macd()
        histogram = pd.DataFrame(macd_.values - signal_.values)
        ret = digit(histogram.diff().values)
        self.returns = ret
        return ret

    def strategy_regression_model(self, plot=True, *args, **kwargs):
        ret = []
        for i in range(self.inputs - 1):
            ret.append(EXIT)
        data_to_pred = np.array(
            get_window(np.array([self.df['Close'].values]).T, self.inputs))

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
        self.returns = [*ret, *predictions]
        nans = itertools.chain.from_iterable([(np.nan,) * self.inputs])
        filt = (*nans, *frame.T[0])
        if plot:
            self.fig.add_trace(
                go.Line(
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

        self.inputs = inputs
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
                            filter_kwargs=dict(),
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

        if metrics is None:
            metrics = ['acc']
        list_input = []
        list_output = []

        for df in dataframes:
            all_ta = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close',
                                            'Volume', True)
            get_all_out = self._get_this_instance(df=df)
            filter_kwargs['plot'] = False
            output1 = get_all_out.strategy_diff(
                eval('get_all_out.' + filter_ + '(**filter_kwargs)'))

            for output in output1:
                list_output.append(output[0])
            list_input.append(
                pd.DataFrame(
                    self.prepare_scaler(
                        pd.DataFrame(all_ta), regression_net=False)))
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
        import random
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

    def inverse_strategy(self, *args, **kwargs):
        """
        makes signals inverse:

        buy = sell.
        sell = buy.
        exit = exit.

        """

        ret = []
        for signal in self.returns:
            if signal == BUY:
                ret.append(SELL)
            elif signal == SELL:
                ret.append(BUY)
            else:
                ret.append(EXIT)
        self.returns = ret
        return ret

    def basic_backtest(self,
                       deposit=10_000,
                       credit_leverage=1,
                       bet=None,
                       commission=0,
                       stop_loss=None,
                       take_profit=None,
                       plot=True,
                       print_out=True,
                       column='Close',
                       *args,
                       **kwargs):
        """
        testing the strategy.


        deposit:         | int, float. | start deposit.

        credit_leverage: | int, float. | trading leverage. 1 = none.

        bet:             | int, float, | fixed bet to quick_trade. None = all moneys.

        commission:      | int, float. | percentage commission (0 -- 100).

        stop_loss:       | int, float. | stop loss in points.

        take_profit:     | int, float. | take profit in points.

        plot:            |    bool.    | plotting.

        print_out:       |    bool.    | printing.

        column:          |     str     | column of dataframe to becktest.



        returns: pd.DataFrame with data of:
            signals,
            deposit'
            stop loss,
            take profit,
            linear deposit,
            <<column>> price,
            open bet\lot\deal price.


        """

        if stop_loss is None:
            self.stop_loss = np.inf
        else:
            self.stop_loss = stop_loss
        if take_profit is None:
            self.take_profit = np.inf
        else:
            self.take_profit = take_profit

        saved_del = self.returns[len(self.returns) - 1]

        if set_(self.returns)[len(self.returns) - 1] is np.nan:
            self.returns[len(self.returns) - 1] = EXIT

        loc = list(self.df[column].values)

        if plot:
            self.fig.add_candlestick(
                close=self.df['Close'],
                high=self.df['High'],
                low=self.df['Low'],
                open=self.df['Open'],
                row=1,
                col=1,
                name=self.ticker)

        predictions__ = {}
        for e, i in enumerate(set_(self.returns)):
            if i is not np.nan:
                predictions__[e] = i
            # marker's 'y' coordinate on real price of stock/forex
            if plot:
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

        sigs = list(predictions__.values())
        vals = list(predictions__.keys())

        self.moneys = deposit
        leverage = credit_leverage
        rate = bet
        commission = commission

        money_start = self.moneys
        resur = [self.moneys]
        self.losses = 0
        self.trades = 0
        self.open_lot_prices = []
        stop_losses = []
        take_profits = []
        exit = False

        e = 0
        for enum, (val, val2, sig) in enumerate(
                zip(vals[:len(vals) - 1],
                    vals[1:],
                    sigs[:len(sigs) - 1])):
            coefficient = self.moneys / loc[val]
            self.open_price = loc[val]

            _rate = self.moneys if rate is None else rate
            commission_real = _rate * (commission / 100)
            self.moneys -= commission_real
            if _rate > self.moneys:
                _rate = self.moneys
            mons = self.moneys

            for i in (pd.DataFrame(loc).diff().values *
                      coefficient)[val + 1:val2 + 1]:

                min_price = self.df['Low'][e + 1]
                max_price = self.df['High'][e + 1]
                self.open_lot_prices.append(self.open_price)

                take_stop = self.get_stop_take(sig)
                _stop_loss = take_stop['stop']
                take = take_stop['take']

                stop_losses.append(_stop_loss)
                take_profits.append(take)

                def get_condition(value):
                    return min(_stop_loss, take) < value < max(
                        _stop_loss, take)

                cond = get_condition(min_price) and get_condition(max_price)

                if cond and not exit:
                    if self.moneys > 0:
                        if sig == SELL:
                            self.moneys -= i[0] * leverage * (_rate / mons)
                            resur.append(self.moneys)
                        elif sig == BUY:
                            self.moneys += i[0] * leverage * (_rate / mons)
                            resur.append(self.moneys)
                        elif sig == EXIT:
                            resur.append(self.moneys)
                    else:
                        resur.append(0)  # 0 because it's deposit
                else:
                    flag = True
                    if cond and self.moneys > 0:
                        close = self.df['Close'][e + 1]
                        open_ = self.df['Open'][e + 1]
                        if sig == BUY and close < _stop_loss:
                            diff = _stop_loss - open_
                        elif sig == BUY and close > take:
                            diff = take - open_
                        elif sig == SELL and close < take:
                            diff = take - close
                        elif sig == SELL and close > _stop_loss:
                            diff = _stop_loss - open_
                        else:
                            flag = False
                        if flag:
                            if bet is None:
                                _rate = self.moneys
                            commission_real = _rate * (commission / 100)
                            self.moneys -= commission_real
                            if bet is None:
                                _rate = self.moneys
                            if sig == SELL:
                                self.moneys -= diff * coefficient * leverage * (
                                        _rate / mons)
                                resur.append(self.moneys)
                            elif sig == BUY:
                                self.moneys += diff * coefficient * leverage * (
                                        _rate / mons)
                                resur.append(self.moneys)
                    exit = True
                    resur.append(self.moneys)

                e += 1

            self.trades += 1
            if self.moneys < mons:
                self.losses += 1
            if self.moneys < 0:
                self.moneys = 0
            exit = False

        if plot:
            if self.take_profit != np.inf:
                self.fig.add_trace(
                    go.Line(
                        y=take_profits,
                        line=dict(width=TAKE_STOP_OPN_WIDTH, color=G),
                        opacity=STOP_TAKE_OPN_ALPHA,
                        name='take profit'), 1, 1)
            if self.stop_loss != np.inf:
                self.fig.add_trace(
                    go.Line(
                        y=stop_losses,
                        line=dict(width=TAKE_STOP_OPN_WIDTH, color=R),
                        opacity=STOP_TAKE_OPN_ALPHA,
                        name='stop loss'), 1, 1)

            self.fig.add_trace(
                go.Line(
                    y=self.open_lot_prices,
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=B),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='open lot'), 1, 1)

        self.returns[len(self.returns) - 1] = saved_del
        if set_(self.returns)[len(self.returns) - 1] is np.nan:
            stop_losses.append(_stop_loss)
            take_profits.append(take)
            self.open_lot_prices.append(self.open_price)
        else:
            sig = sigs[len(sigs) - 1]
            take_stop = self.get_stop_take(sig)
            _stop_loss = take_stop['stop']
            take = take_stop['take']
            stop_losses.append(_stop_loss)
            take_profits.append(take)
            self.open_lot_prices.append(loc[len(loc) - 1])

        linear_dat = self.linear_(resur)
        if plot:
            self.fig.add_trace(
                go.Line(
                    y=resur,
                    line=dict(color=COLOR_DEPOSIT),
                    name=f'D E P O S I T  (S T A R T: ${money_start})'), 2, 1)
            self.fig.add_trace(go.Line(y=linear_dat, name='L I N E A R'), 2, 1)
            for e, i in enumerate(resur):
                if i < 0:
                    self.fig.add_scatter(
                        y=[i],
                        x=[e],
                        row=2,
                        col=1,
                        line=dict(color=R),
                        marker=dict(
                            symbol='triangle-down',
                            size=SCATTER_SIZE,
                            opacity=SCATTER_DEPO_ALPHA))

        start = linear_dat[0]
        end = linear_dat[len(linear_dat) - 1]
        self.year_profit = (end - start) / start * 100
        self.year_profit /= len(resur) * self.profit_calculate_coef
        self.linear = linear_dat
        self.profits = self.trades - self.losses
        if print_out:
            print(f'L O S S E S: {self.losses}')
            print(f'T R A D E S: {self.trades}')
            print(f'P R O F I T S: {self.profits}')
            print(
                'M E A N   P E R C E N T A G E   Y E A R   P R O F I T: ',
                self.year_profit,
                '%',
                sep='')
        if plot:
            self.fig.show()
        self.stop_losses = stop_losses
        self.take_profits = take_profits
        self.money_list = resur
        self.backtest_out = pd.DataFrame(
            (resur, stop_losses, take_profits, self.returns,
             self.open_lot_prices, loc, self.linear),
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
                   row_heights=[100, 160],
                   **subplot_kwargs):
        self.fig = sub_make(2, 1, row_heights=row_heights, **subplot_kwargs)
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

        standart: nothing.

        example:  Strategies.strategy_macd.

        second_func:     |  trading strategy  |   strategy to combine.

        standart: nothing.

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
        ret = []
        if mode == 'minimalist':
            for ret1, ret2 in zip(first_returns, second_returns):
                if ret1 == ret2:
                    ret.append(ret1)
                else:
                    ret.append(EXIT)
        elif mode == 'maximalist':
            ret = self.__maximalist(first_returns, second_returns)
        elif mode == 'super':
            ret = self.__collide_super(first_returns, second_returns)
        else:
            raise ValueError('I N C O R R E C T   M O D E')
        self.returns = ret
        return ret

    def __maximalist(self, returns1, returns2):
        ret = []
        flag = EXIT
        for a, b in zip(returns1, returns2):
            if a == b:
                ret.append(a)
                flag = a
            else:
                ret.append(flag)
        return ret

    def __collide_super(self, l1, l2):
        ret = []
        for first, sec in zip(set_(l1), set_(l2)):
            if first is not np.nan and sec is not np.nan and first is not sec:
                ret.append(EXIT)
            elif first is sec:
                ret.append(first)
            elif first is np.nan:
                ret.append(sec)
            else:
                ret.append(first)
        return anti_set_(ret)

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
                    if not self.first:
                        self.client.exit_last_order()
                        logger.info('client exit')
                    self.client.new_order_buy(self.ticker, bet, credit_leverage=credit_leverage)
                    logger.info('client buy')
                    self.__exit_order__ = False
                    self.first = False

                if predict == 'Sell':
                    if not self.first:
                        self.client.exit_last_order()
                        logger.info('client exit')
                    if can_sell:
                        self.client.new_order_sell(self.ticker, bet, credit_leverage=credit_leverage)
                        logger.info('client sell')
                    self.first = False
                    self.__exit_order__ = False

                if predict == 'Exit':
                    if not self.first:
                        self.client.exit_last_order()
                        logger.info('client exit')

            predict = predicts[len(predicts) - 1]
            for sig, close_ in zip(self.returns[::-1],
                                   self.df['Close'].values[::-1]):
                if sig != EXIT:
                    self.open_price = close_
                    break
            rets = self.get_stop_take(predict)
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
                         get_data_kwargs=dict(),
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
        global ret
        ret = {}
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
                ret[index] = prediction
                while True:
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
                            ret[index] = prediction
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
                 bet=None,
                 commission=0,
                 stop_loss=None,
                 take_profit=None,
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
            open bet\lot\deal price.

        """

        money_start = deposit
        loc = self.df['Close'].values
        _df_flag = self.df
        self.df = pd.concat(
            [
                self.df['Close'], self.df['Open'], self.df['High'],
                self.df['Low']
            ],
            axis=1)
        __df = self.df
        df = inverse_4_col_df(self.df, ['Close', 'Open', 'High', 'Low'])

        self.df = df
        _returns_flag = self.returns

        self.returns = []
        for pred in _returns_flag:
            for column in range(4):
                self.returns.append(pred)

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
        deposit_df = to_4_col_df(deposit_df, 'deposit Close', 'deposit Open',
                                 'deposit High', 'deposit Low')

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
                    min(self.money_list),
                    max(self.money_list)
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
                go.Line(
                    y=self.take_profits.values.T[0],
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=G),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='take profit'),
                row=1,
                col=1)
            self.fig.add_trace(
                go.Line(
                    y=self.stop_losses.values.T[0],
                    line=dict(width=TAKE_STOP_OPN_WIDTH, color=R),
                    opacity=STOP_TAKE_OPN_ALPHA,
                    name='stop loss'),
                row=1,
                col=1)
            self.fig.add_trace(
                go.Line(
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
                go.Line(y=self.linear.values.T[0], name='L I N E A R'),
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

    def set_client(self, class_client, *client_set_args, **client_set_kwargs):
        self.client = class_client(*client_set_args, **client_set_kwargs)

    def convert_signal(self, old=SELL, new=EXIT):
        for pos, val in enumerate(self.returns):
            if val == old:
                self.returns[pos] = new


class PatternFinder(Strategies):
    """
    algo-trading system.


    ticker:   |     str      |  ticker/symbol of chart

    days_undo:|     int      |  days of chart

    df:       |   dataframe  |  data of chart

    interval: |     str      |  interval of df.
    one of: '1d', '1m' ...


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

    def __init__(self,
                 ticker='AAPL',
                 days_undo=100,
                 df=np.nan,
                 interval='1d',
                 rounding=5,
                 *args,
                 **kwargs):
        df_ = round(df, rounding)
        self.first = True
        self.rounding = rounding
        diff = digit(df_['Close'].diff().values)[1:]
        self.diff = [EXIT, *diff]
        self.df = df_.reset_index(drop=True)
        self.drop = 0
        self.ticker = ticker
        self.days_undo = days_undo
        self.interval = interval
        if interval == '1m':
            self.profit_calculate_coef = 1 / 60 / 24 / 365
        elif interval == '1d':
            self.profit_calculate_coef = 1 / 365
        else:
            raise ValueError('I N C O R R E C T   I N T E R V A L')
        self.inputs = INPUTS
        self.__exit_order__ = False

    def _window_(self, column, n=2):
        return get_window(self.df[column].values, n)

    def find_pip_bar(self, min_diff_coef=2, body_coef=10):
        ret = []
        flag = EXIT
        for e, (high, low, open, close) in enumerate(
                zip(self.df['High'], self.df['Low'], self.df['Open'],
                    self.df['Close']), 1):
            body = abs(open - close)
            shadow_high = high - max(open, close)
            shadow_low = min(open, close) - low
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
        ret = ret
        self.returns = ret
        return ret

    def find_DBLHC_DBHLC(self):
        ret = [EXIT]
        flag = EXIT
        for e, (high, low, open, close) in enumerate(
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
        ret = ret[self.drop:]
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
        ret = ret
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
        ret = ret
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


if __name__ == '__main__':
    # tests
    import yfinance as yf


    def real(df, window_length=221, polyorder=3):

        from tqdm import auto
        """
        for i in auto.tqdm(range(aa, len(df - 1))):
            filtered.append(signal.savgol_filter(
                df[i - aa:i],
                window_length=window_length,
                polyorder=polyorder)[aa - 1])
        """

        aa = 222
        filtered = [EXIT for _ in range(aa)]

        filtered1 = df
        for _ in auto.tqdm(range(len(df))):
            filtered1 = KalmanFilter().filter(filtered1)[0]
            filtered.append(filtered1[len(filtered1) - 1])
        return pd.DataFrame(filtered)


    profits = []
    from itertools import chain
    for ticker in chain.from_iterable([
         [['BTC-USD', '1d', 'max'], ['BTC-USD', '1m', '5d'], ['BTC-USD', '1d', '3y']],
         [['ETH-USD', '1d', 'max'], ['ETH-USD', '1m', '5d'], ['ETH-USD', '1d', '3y']],
         [['USDT-USD', '1d', 'max'], ['USDT-USD', '1m', '5d'], ['USDT-USD', '1d', '3y']],
         [['XRP-USD', '1d', 'max'], ['XRP-USD', '1m', '5d'], ['XRP-USD', '1d', '3y']],
         [['LINK-USD', '1d', 'max'], ['LINK-USD', '1m', '5d'], ['LINK-USD', '1d', '3y']],
         [['BCH-USD', '1d', 'max'], ['BCH-USD', '1m', '5d'], ['BCH-USD', '1d', '3y']],
         [['BNB-USD', '1d', 'max'], ['BNB-USD', '1m', '5d'], ['BNB-USD', '1d', '3y']],
         [['LTC-USD', '1d', 'max'], ['LTC-USD', '1m', '5d'], ['LTC-USD', '1d', '3y']],
         [['EOS-USD', '1d', 'max'], ['EOS-USD', '1m', '5d'], ['EOS-USD', '1d', '3y']],
         [['ADA-USD', '1d', 'max'], ['ADA-USD', '1m', '5d'], ['ADA-USD', '1d', '3y']],
         [['TRX-USD', '1d', 'max'], ['TRX-USD', '1m', '5d'], ['TRX-USD', '1d', '3y']],
         [['XLM-USD', '1d', 'max'], ['XLM-USD', '1m', '5d'], ['XLM-USD', '1d', '3y']],
         [['XMR-USD', '1d', 'max'], ['XMR-USD', '1m', '5d'], ['XMR-USD', '1d', '3y']],
         [['NEO-USD', '1d', 'max'], ['NEO-USD', '1m', '5d'], ['NEO-USD', '1d', '3y']],
         [['XEM-USD', '1d', 'max'], ['XEM-USD', '1m', '5d'], ['XEM-USD', '1d', '3y']],
         [['MIOTA-USD', '1d', 'max'], ['MIOTA-USD', '1m', '5d'], ['MIOTA-USD', '1d', '3y']],
         [['DASH-USD', '1d', 'max'], ['DASH-USD', '1m', '5d'], ['DASH-USD', '1d', '3y']],
         [['VET-USD', '1d', 'max'], ['VET-USD', '1m', '5d'], ['VET-USD', '1d', '3y']],
         [['ZEC-USD', '1d', 'max'], ['ZEC-USD', '1m', '5d'], ['ZEC-USD', '1d', '3y']],
         [['ETC-USD', '1d', 'max'], ['ETC-USD', '1m', '5d'], ['ETC-USD', '1d', '3y']],
         [['OMG-USD', '1d', 'max'], ['OMG-USD', '1m', '5d'], ['OMG-USD', '1d', '3y']],
         [['BAT-USD', '1d', 'max'], ['BAT-USD', '1m', '5d'], ['BAT-USD', '1d', '3y']],
         [['DOGE-USD', '1d', 'max'], ['DOGE-USD', '1m', '5d'], ['DOGE-USD', '1d', '3y']],
         [['ZRX-USD', '1d', 'max'], ['ZRX-USD', '1m', '5d'], ['ZRX-USD', '1d', '3y']],
         [['DGB-USD', '1d', 'max'], ['DGB-USD', '1m', '5d'], ['DGB-USD', '1d', '3y']],
         [['LRC-USD', '1d', 'max'], ['LRC-USD', '1m', '5d'], ['LRC-USD', '1d', '3y']],
         [['WAVES-USD', '1d', 'max'], ['WAVES-USD', '1m', '5d'], ['WAVES-USD', '1d', '3y']],
         [['ICX-USD', '1d', 'max'], ['ICX-USD', '1m', '5d'], ['ICX-USD', '1d', '3y']],
         [['QTUM-USD', '1d', 'max'], ['QTUM-USD', '1m', '5d'], ['QTUM-USD', '1d', '3y']],
         [['KNC-USD', '1d', 'max'], ['KNC-USD', '1m', '5d'], ['KNC-USD', '1d', '3y']],
         [['REPV2-USD', '1d', 'max'], ['REPV2-USD', '1m', '5d'], ['REPV2-USD', '1d', '3y']],
         [['REP-USD', '1d', 'max'], ['REP-USD', '1m', '5d'], ['REP-USD', '1d', '3y']],
         [['LSK-USD', '1d', 'max'], ['LSK-USD', '1m', '5d'], ['LSK-USD', '1d', '3y']],
         [['DCR-USD', '1d', 'max'], ['DCR-USD', '1m', '5d'], ['DCR-USD', '1d', '3y']],
         [['SC-USD', '1d', 'max'], ['SC-USD', '1m', '5d'], ['SC-USD', '1d', '3y']],
         [['ANT-USD', '1d', 'max'], ['ANT-USD', '1m', '5d'], ['ANT-USD', '1d', '3y']],
         [['BTG-USD', '1d', 'max'], ['BTG-USD', '1m', '5d'], ['BTG-USD', '1d', '3y']],
         [['GNT-USD', '1d', 'max'], ['GNT-USD', '1m', '5d'], ['GNT-USD', '1d', '3y']],
         [['NANO-USD', '1d', 'max'], ['NANO-USD', '1m', '5d'], ['NANO-USD', '1d', '3y']],
         [['BTS-USD', '1d', 'max'], ['BTS-USD', '1m', '5d'], ['BTS-USD', '1d', '3y']],
         [['SNT-USD', '1d', 'max'], ['SNT-USD', '1m', '5d'], ['SNT-USD', '1d', '3y']],
         [['STORJ-USD', '1d', 'max'], ['STORJ-USD', '1m', '5d'], ['STORJ-USD', '1d', '3y']],
         [['MONA-USD', '1d', 'max'], ['MONA-USD', '1m', '5d'], ['MONA-USD', '1d', '3y']],
         [['RLC-USD', '1d', 'max'], ['RLC-USD', '1m', '5d'], ['RLC-USD', '1d', '3y']],
         [['BNT-USD', '1d', 'max'], ['BNT-USD', '1m', '5d'], ['BNT-USD', '1d', '3y']],
         [['XVG-USD', '1d', 'max'], ['XVG-USD', '1m', '5d'], ['XVG-USD', '1d', '3y']],
         [['KMD-USD', '1d', 'max'], ['KMD-USD', '1m', '5d'], ['KMD-USD', '1d', '3y']],
         [['GNO-USD', '1d', 'max'], ['GNO-USD', '1m', '5d'], ['GNO-USD', '1d', '3y']],
         [['STEEM-USD', '1d', 'max'], ['STEEM-USD', '1m', '5d'], ['STEEM-USD', '1d', '3y']],
         [['MCO-USD', '1d', 'max'], ['MCO-USD', '1m', '5d'], ['MCO-USD', '1d', '3y']],
         [['ARDR-USD', '1d', 'max'], ['ARDR-USD', '1m', '5d'], ['ARDR-USD', '1d', '3y']],
         [['ZEN-USD', '1d', 'max'], ['ZEN-USD', '1m', '5d'], ['ZEN-USD', '1d', '3y']],
         [['XZC-USD', '1d', 'max'], ['XZC-USD', '1m', '5d'], ['XZC-USD', '1d', '3y']],
         [['MLN-USD', '1d', 'max'], ['MLN-USD', '1m', '5d'], ['MLN-USD', '1d', '3y']],
         [['HC-USD', '1d', 'max'], ['HC-USD', '1m', '5d'], ['HC-USD', '1d', '3y']],
         [['STRAT-USD', '1d', 'max'], ['STRAT-USD', '1m', '5d'], ['STRAT-USD', '1d', '3y']],
         [['ETHOS-USD', '1d', 'max'], ['ETHOS-USD', '1m', '5d'], ['ETHOS-USD', '1d', '3y']],
         [['AE-USD', '1d', 'max'], ['AE-USD', '1m', '5d'], ['AE-USD', '1d', '3y']],
         [['ARK-USD', '1d', 'max'], ['ARK-USD', '1m', '5d'], ['ARK-USD', '1d', '3y']],
         [['MAID-USD', '1d', 'max'], ['MAID-USD', '1m', '5d'], ['MAID-USD', '1d', '3y']],
         [['SYS-USD', '1d', 'max'], ['SYS-USD', '1m', '5d'], ['SYS-USD', '1d', '3y']],
         [['VGX-USD', '1d', 'max'], ['VGX-USD', '1m', '5d'], ['VGX-USD', '1d', '3y']],
         [['WTC-USD', '1d', 'max'], ['WTC-USD', '1m', '5d'], ['WTC-USD', '1d', '3y']],
         [['BCN-USD', '1d', 'max'], ['BCN-USD', '1m', '5d'], ['BCN-USD', '1d', '3y']],
         [['BTCD-USD', '1d', 'max'], ['BTCD-USD', '1m', '5d'], ['BTCD-USD', '1d', '3y']],
         [['FUN-USD', '1d', 'max'], ['FUN-USD', '1m', '5d'], ['FUN-USD', '1d', '3y']],
         [['PIVX-USD', '1d', 'max'], ['PIVX-USD', '1m', '5d'], ['PIVX-USD', '1d', '3y']],
         [['CVC-USD', '1d', 'max'], ['CVC-USD', '1m', '5d'], ['CVC-USD', '1d', '3y']],
         [['MTL-USD', '1d', 'max'], ['MTL-USD', '1m', '5d'], ['MTL-USD', '1d', '3y']],
         [['ICN-USD', '1d', 'max'], ['ICN-USD', '1m', '5d'], ['ICN-USD', '1d', '3y']],
         [['GAS-USD', '1d', 'max'], ['GAS-USD', '1m', '5d'], ['GAS-USD', '1d', '3y']],
         [['GBYTE-USD', '1d', 'max'], ['GBYTE-USD', '1m', '5d'], ['GBYTE-USD', '1d', '3y']],
         [['ADX-USD', '1d', 'max'], ['ADX-USD', '1m', '5d'], ['ADX-USD', '1d', '3y']],
         [['PPT-USD', '1d', 'max'], ['PPT-USD', '1m', '5d'], ['PPT-USD', '1d', '3y']],
         [['KIN-USD', '1d', 'max'], ['KIN-USD', '1m', '5d'], ['KIN-USD', '1d', '3y']],
         [['VTC-USD', '1d', 'max'], ['VTC-USD', '1m', '5d'], ['VTC-USD', '1d', '3y']],
         [['ETP-USD', '1d', 'max'], ['ETP-USD', '1m', '5d'], ['ETP-USD', '1d', '3y']],
         [['FCT-USD', '1d', 'max'], ['FCT-USD', '1m', '5d'], ['FCT-USD', '1d', '3y']],
         [['QASH-USD', '1d', 'max'], ['QASH-USD', '1m', '5d'], ['QASH-USD', '1d', '3y']],
         [['NXS-USD', '1d', 'max'], ['NXS-USD', '1m', '5d'], ['NXS-USD', '1d', '3y']],
         [['NXT-USD', '1d', 'max'], ['NXT-USD', '1m', '5d'], ['NXT-USD', '1d', '3y']],
         [['UBQ-USD', '1d', 'max'], ['UBQ-USD', '1m', '5d'], ['UBQ-USD', '1d', '3y']],
         [['DGD-USD', '1d', 'max'], ['DGD-USD', '1m', '5d'], ['DGD-USD', '1d', '3y']],
         [['PAY-USD', '1d', 'max'], ['PAY-USD', '1m', '5d'], ['PAY-USD', '1d', '3y']],
         [['WINGS-USD', '1d', 'max'], ['WINGS-USD', '1m', '5d'], ['WINGS-USD', '1d', '3y']],
         [['NAV-USD', '1d', 'max'], ['NAV-USD', '1m', '5d'], ['NAV-USD', '1d', '3y']],
         [['NEBL-USD', '1d', 'max'], ['NEBL-USD', '1m', '5d'], ['NEBL-USD', '1d', '3y']],
         [['TAAS-USD', '1d', 'max'], ['TAAS-USD', '1m', '5d'], ['TAAS-USD', '1d', '3y']],
         [['QRL-USD', '1d', 'max'], ['QRL-USD', '1m', '5d'], ['QRL-USD', '1d', '3y']],
         [['DNT-USD', '1d', 'max'], ['DNT-USD', '1m', '5d'], ['DNT-USD', '1d', '3y']],
         [['GAME-USD', '1d', 'max'], ['GAME-USD', '1m', '5d'], ['GAME-USD', '1d', '3y']],
         [['BLOCK-USD', '1d', 'max'], ['BLOCK-USD', '1m', '5d'], ['BLOCK-USD', '1d', '3y']],
         [['SALT-USD', '1d', 'max'], ['SALT-USD', '1m', '5d'], ['SALT-USD', '1d', '3y']],
         [['VERI-USD', '1d', 'max'], ['VERI-USD', '1m', '5d'], ['VERI-USD', '1d', '3y']],
         [['PART-USD', '1d', 'max'], ['PART-USD', '1m', '5d'], ['PART-USD', '1d', '3y']],
         [['DCN-USD', '1d', 'max'], ['DCN-USD', '1m', '5d'], ['DCN-USD', '1d', '3y']],
         [['SMART-USD', '1d', 'max'], ['SMART-USD', '1m', '5d'], ['SMART-USD', '1d', '3y']],
         [['SNGLS-USD', '1d', 'max'], ['SNGLS-USD', '1m', '5d'], ['SNGLS-USD', '1d', '3y']],
         [['SNM-USD', '1d', 'max'], ['SNM-USD', '1m', '5d'], ['SNM-USD', '1d', '3y']],
         [['LKK-USD', '1d', 'max'], ['LKK-USD', '1m', '5d'], ['LKK-USD', '1d', '3y']],
         [['XCP-USD', '1d', 'max'], ['XCP-USD', '1m', '5d'], ['XCP-USD', '1d', '3y']],
         [['NLC2-USD', '1d', 'max'], ['NLC2-USD', '1m', '5d'], ['NLC2-USD', '1d', '3y']],
         [['IOC-USD', '1d', 'max'], ['IOC-USD', '1m', '5d'], ['IOC-USD', '1d', '3y']],
         [['FAIR-USD', '1d', 'max'], ['FAIR-USD', '1m', '5d'], ['FAIR-USD', '1d', '3y']],
         [['MGO-USD', '1d', 'max'], ['MGO-USD', '1m', '5d'], ['MGO-USD', '1d', '3y']],
         [['SUB-USD', '1d', 'max'], ['SUB-USD', '1m', '5d'], ['SUB-USD', '1d', '3y']],
         [['EDG-USD', '1d', 'max'], ['EDG-USD', '1m', '5d'], ['EDG-USD', '1d', '3y']],
         [['BTM1-USD', '1d', 'max'], ['BTM1-USD', '1m', '5d'], ['BTM1-USD', '1d', '3y']],
         [['FRST-USD', '1d', 'max'], ['FRST-USD', '1m', '5d'], ['FRST-USD', '1d', '3y']],
         [['MCAP-USD', '1d', 'max'], ['MCAP-USD', '1m', '5d'], ['MCAP-USD', '1d', '3y']],
         [['ATB-USD', '1d', 'max'], ['ATB-USD', '1m', '5d'], ['ATB-USD', '1d', '3y']],
         [['XUC-USD', '1d', 'max'], ['XUC-USD', '1m', '5d'], ['XUC-USD', '1d', '3y']]
    ]):
        print(ticker)
        TICKER = ticker[0]
        interval = ticker[1]
        df = yf.download(TICKER, interval=interval, period=ticker[2])
        trader = PatternFinder(df=df, interval=interval, ticker=TICKER)
        trader.set_client(TradingClient)
        trader.set_pyplot()
        # trader.get_trained_network([df], network_save_path='qwty')
        trader.load_model('./model_regression')
        trader.prepare_scaler(dataframe=trader.df, regression_net=True)
        trader.strategy_regression_model()
        # trader.strategy_diff(real(df['Close'].values))
        # trader.convert_signal()
        trader.log_deposit()
        trader.backtest(commission=0.075, show=False, plot=False)
        profits.append((trader.year_profit, ticker))
    print(profits)
