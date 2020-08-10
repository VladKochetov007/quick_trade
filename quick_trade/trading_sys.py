"""used ta by Darío López Padial (Bukosabino) https://github.com/bukosabino/ta"""

import copy
import time

import plotly.graph_objects as go
import ta
from plotly.subplots import make_subplots as sub_make
from pykalman import KalmanFilter
from quick_trade.utils import *
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


class Strategies(object):
    """

    basic class for PatternFinder.
    please use PatternFinder class.

    """

    def __init__(self,
                 ticker='AAPL',
                 days_undo=100,
                 df=np.nan,
                 interval='1m',
                 *args,
                 **kwargs):
        self.ticker = ticker
        self.days_undo = days_undo
        self.drop = 0
        self.interval = interval
        if df is np.nan:
            data = get_data(ticker, days_undo)
            self.df = data.reset_index()
        else:
            self.df = df.reset_index()
        self.df = self.df[self.drop:].reset_index()
        if interval == '1m':
            self.profit_calculate_coef = 1 / 60 / 24 / 365
        elif interval == '1d':
            self.profit_calculate_coef = 1 / 365
        else:
            raise ValueError('I N C O R R E C T   I N T E R V A L')

    def __repr__(self):
        return 'trader'

    def __iter__(self):
        try:
            return iter(list(self.backtest_out.values))
        except AttributeError:
            raise ValueError('D O   B A C K T E S T')

    def kalman_filter(self,
                      df='self.df["Close"]',
                      iters=40,
                      plot=True,
                      *args,
                      **kwargs):
        k_filter = KalmanFilter()
        if df == 'self.df["Close"]':
            df = self.df['Close']
        filtered = k_filter.filter(np.array(df))[0]
        for i in range(iters):
            filtered = k_filter.smooth(filtered)[0]
        if plot:
            self.fig.add_trace(
                go.Line(
                    name='kalman filter',
                    y=filtered,
                    line=dict(width=SUB_LINES_WIDTH)), 1, 1)
        return pd.DataFrame(filtered)

    def scipy_filter(self,
                     window_length=101,
                     df='self.df["Close"]',
                     polyorder=3,
                     plot=True,
                     **scipy_savgol_filter_kwargs):
        if df == 'self.df["Close"]':
            df = self.df['Close']
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
        ret = list(np.digitize(frame_to_diff.diff(), bins=[0]))
        ret = ret[self.drop:]
        self.returns = ret
        return ret

    def strategy_2_sma(self, slow=100, fast=30, plot=True, *args, **kwargs):
        ret = []
        SMA1 = ta.trend.sma(self.df['Close'], fast)[self.drop:]
        SMA2 = ta.trend.sma(self.df['Close'], slow)[self.drop:]
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
        SMA1 = ta.trend.sma(self.df['Close'], fast)[self.drop:]
        SMA2 = ta.trend.sma(self.df['Close'], mid)[self.drop:]
        SMA3 = ta.trend.sma(self.df['Close'], slow)[self.drop:]

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

    def strategy_exp_diff(self, period=80, plot=True, *args, **kwargs):
        ret = self.strategy_diff(self.tema(period))
        if plot:
            self.fig.add_trace(
                go.Line(
                    name=f'EMA{period}',
                    y=self.tema(period).values[self.drop:],
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
        ret = digit(histogram.diff().values)[self.drop:]
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
                               network_save_path='quick_trade/model-regression',
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
                            filter_kwargs={},
                            optimizer='adam',
                            loss='mse',
                            metrics=None,
                            network_save_path='quick_trade/model-predicting',
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
            get_all_out = PatternFinder(df=df)
            filter_kwargs['plot'] = False
            output1 = get_all_out.strategy_diff(
                eval('get_all_out.' + filter_ + '(**filter_kwargs)'))

            for output in output1:
                list_output.append(output[0])
            list_input.append(pd.DataFrame(self.prepare_scaler(pd.DataFrame(all_ta), regression_net=False)))
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

    def strategy_with_network(self, rounding=0, *args, **kwargs):
        scaler = MinMaxScaler(feature_range=(0, 1))
        all_ta = ta.add_all_ta_features(self.df, "Open", 'High', 'Low', 'Close', "Volume", True).values
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

        bet:             | int, float, | fixed bet to quick_trade--. None = all moneys.

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

        __predictions = {}
        for e, i in enumerate(set_(self.returns)):
            if i is not np.nan:
                __predictions[e] = i
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

        sigs = list(__predictions.values())
        vals = list(__predictions.keys())

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
                zip(vals[:len(vals) - 1], vals[1:], sigs[:len(sigs) - 1])):
            mons = self.moneys
            coef = self.moneys / loc[val]
            self.open_price = loc[val]

            _rate = self.moneys if rate is None else rate
            if rate is not None:
                if _rate >= rate:
                    _rate = self.moneys
            _rate -= _rate * (commission / 100)

            for i in (
                             pd.DataFrame(loc).diff().values * coef)[val + 1:val2 + 1]:

                min_price = self.df['Low'][e + 1]
                max_price = self.df['High'][e + 1]
                self.open_lot_prices.append(self.open_price)

                take_stop = self.get_stop_take(sig)
                _stop_loss = take_stop['stop']
                take = take_stop['take']

                stop_losses.append(_stop_loss)
                take_profits.append(take)

                def get_condition(value):
                    return min(_stop_loss, take) < value < max(_stop_loss, take)

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
                        resur.append(0)  # 0 but it's deposit
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
                            if sig == SELL:
                                self.moneys -= diff * coef * leverage * (_rate / mons)
                                resur.append(self.moneys)
                            elif sig == BUY:
                                self.moneys += diff * coef * leverage * (_rate / mons)
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
                            *args,
                            **kwargs):
        if take_profit is None:
            self.take_profit = np.inf
        if stop_loss is None:
            self.stop_loss = np.inf
        if inverse:
            self.inverse_strategy()
        predicts = self.returns
        predict = predicts[len(predicts) - 1]
        close = self.df['Close'].values
        if set_(predicts)[len(predicts) - 1] is np.nan:
            opn = self.open_price
        else:
            opn = close[len(close) - 1]
        curr_price = close[len(close) - 1]
        rets = self.get_stop_take(predict)
        if predict == BUY:
            predict = 'Buy'
        elif predict == SELL:
            predict = 'Sell'
        elif predict == EXIT:
            predict = 'Exit'
        return {
            'predict': predict,
            'open lot price': opn,
            'stop loss': rets['stop'],
            'take profit': rets['take'],
            'currency close': curr_price
        }

    def realtime_trading(self,
                         strategy,
                         get_gataframe,
                         get_data_kwargs={},
                         sleeping_time=60,
                         print_out=True,
                         take_profit=None,
                         stop_loss=None,
                         inverse=False,
                         **strategy_kwargs):
        """
        strategy:         |   Strategies.some_strategy  |  trading strategy.

        get_gataframe:    |          function           |  function to getting the data:
            first argument must be a ticker.

        get_data_kwargs:  |             dict            |  named arguments to <<get_gataframe>> WITHOUT TICKER.

        **strategy_kwargs:|             kwargs          |  named arguments to <<strategy>>.

        sleeping_time:    |             int             |  sleeping time.

        print_out:        |             bool            |  printing.

        """

        try:
            ret = {}
            while True:
                self.df = get_gataframe(self.ticker, **get_data_kwargs)
                strategy(**strategy_kwargs)
                prediction = self.get_trading_predict(
                    take_profit, stop_loss, inverse=inverse)
                now = copy.copy(time.ctime())
                ret[now] = prediction
                if print_out:
                    print(now, prediction)
                time.sleep(sleeping_time)
        except KeyboardInterrupt:
            return ret

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

            1: dropped na
            2: no dropped

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
            ret = list(np.array(obj))[::4]
            ret = pd.DataFrame(ret, columns=columns).reset_index()
            del ret['index']
            return ret

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
        self.year_profit = self.mean_diff / self.profit_calculate_coef + money_start
        self.year_profit = ((self.year_profit - money_start) / money_start) * 100
        self.info = (f"L O S S E S: {self.losses}\n"
                     f"T R A D E S: {self.trades}\n"
                     f"P R O F I T S: {self.profits}\n"
                     f"M E A N   Y E A R   P E R C E N T A G E P   R O F I T: {self.year_profit}%\n")
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

        return self.backtest_out, self.backtest_out_no_drop

    def load_model(self, path):
        self.model = load_model(path)


class PatternFinder(Strategies):
    """
    algo-trading system.


    ticker:   |     str      |  ticker/symbol of chart

    days_undo:|     int      |  days of chart

    df:       |   dataframe  |  data of chart

    interval: |     str      |  interval of df.
    one of: '1d', '1m'


    step 1:
    creating a plotter

    step 2:
    using strategy

    step 3:
    backtesting a strategy


    example:


    drop = 0


    trader = Strategies('AAPL', 100, drop)

    trader.set_pyplot()

    trader.strategy_3_sma()

    resur1 = trader.backtest(credit_leverage=40)

    """

    def __getitem__(self, item):
        ret_trader = copy.copy(self)
        ret_trader.df = self.df.T[item].T
        try:
            ret_trader.returns = copy.copy(self.returns[item])
        except AttributeError:
            pass
        try:
            ret_trader.backtest_out = copy.copy(self.backtest_out.T[item].T)
        except AttributeError:
            pass
        try:
            ret_trader.open_lot_prices = copy.copy(self.open_lot_prices[item])
            ret_trader.stop_losses = copy.copy(self.stop_losses[item])
            ret_trader.take_profits = copy.copy(self.take_profits[item])
        except AttributeError:
            pass
        return ret_trader

    def __init__(self,
                 ticker='AAPL',
                 days_undo=100,
                 df=np.nan,
                 interval='1d',
                 rounding=5,
                 *args,
                 **kwargs):
        df_ = round(df, rounding)
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

    def _window_(self, column, n=2):
        return get_window(self.df[column].values, n)

    def find_pip_bar(self, min_diff_co=2):
        ret = []
        flag = EXIT
        for e, (high, low, open, close) in enumerate(
                zip(self.df['High'], self.df['Low'], self.df['Open'],
                    self.df['Close']), 1):
            body = abs(open - close)
            shadow_high = high - max(open, close)
            shadow_low = min(open, close) - low
            if body < (max(shadow_high, shadow_low) * min_diff_co):
                if shadow_low > (shadow_high * min_diff_co):
                    ret.append(BUY)
                    flag = BUY
                elif shadow_high > (shadow_low * min_diff_co):
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
            if low[0] == low[1] and close[1] > close[0]:
                ret.append(BUY)
                flag = BUY
            elif high[0] == high[1] and close[0] > close[1]:
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
            if high[0] == high[1] and self.diff[e -
                                                1] == 1 and self.diff[e] == 0:
                ret.append(BUY)
                flag = BUY
            elif low[0] == low[1] and self.diff[e -
                                                1] == 0 and self.diff[e] == 1:
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
