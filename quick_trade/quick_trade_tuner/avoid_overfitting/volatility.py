import numpy as np
import typing

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation
import ccxt
from quick_trade.quick_trade._saving import Buffer  # "...saving"


class DataFrameHandler:
    def __init__(self, client=None, timeframe: str = '1d'):
        if client is None:
            client = TradingClient(ccxt.binance())
        self._client = client
        self._timeframe = timeframe
        self._buffer = Buffer()

    def __prepare_frame(self):
        self.df = pd.DataFrame(
            {
                'Open': self.df['Open'],
                'High': self.df['High'],
                'Low': self.df['Low'],
                'Close': self.df['Close'],
            }
        )

    def __download(self, ticker: str):
        self.df = self._client.get_data_historical(ticker, self._timeframe)

    def download(self, ticker: str):
        if ticker in self._buffer:
            self.df = self._buffer.read(ticker)
        else:
            self.__download(ticker=ticker)
            self.__prepare_frame()
            self._buffer.write(key=ticker, data=self.df)
        return self.df


class VolatilityHandler:
    def __init__(self, data_handler: DataFrameHandler, span_start: int = 20, span_end: int = 30, span_step: int = 2):
        self._data_handler = data_handler
        self._span = range(span_start, span_end, span_step)

    def historical_volatility(self, ticker: str, period: int) -> pd.Series:
        df = self._data_handler.download(ticker)
        roll = df.rolling(period)
        mini = roll.min()['Low'].values
        maxi = roll.max()['High'].values
        mean = roll.mean()
        mean = mean.mean(axis=1).values

        return pd.Series((maxi - mini) / mean)

    def average_historical_volatility(self, ticker: str, period: int = 20) -> np.float64:
        return self.historical_volatility(ticker=ticker, period=period).mean()

    def period_volatility(self, ticker: str) -> pd.Series:
        volatility = []
        for period in self._span:
            volatility.append(
                self.average_historical_volatility(ticker=ticker, period=period)
            )
        return pd.Series(volatility, index=self._span)

    def multipair_volatility(self, tickers: typing.Iterable[str]):
        analysis = pd.DataFrame
        for ticker in tickers:
            analysis[ticker] = self.period_volatility(ticker)
        return analysis


class VolatilityScaler:
    def __init__(self, volatility: pd.DataFrame):
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._volatility = volatility

    def __fit(self):
        self._scaler.fit(volatility.T.values)

    def scaled_analysis(self):
        return self._scaler.fit_transform(self._volatility.T.values).T






"""def get_volatility(df, period):##################
    roll = df.rolling(period)
    mini = roll.min()['Low'].values
    maxi = roll.max()['High'].values
    mean = roll.mean()
    mean = mean.mean(axis=1).values

def get_mean_volatility(df, period):###################
    return pd.Series(get_volatility(df, period)).mean()

def volatility_by_period(df, start=20, stop=30, step=2):###############
    volatility_correlation = []
    for period in range(start, stop, step):
        volatility_correlation.append(get_mean_volatility(df, period=period))
    return pd.Series(volatility_correlation, index=range(start, stop, step))
def scale_volatility(volatility: pd.DataFrame):################
    scaler = MinMaxScaler()
    scaler.fit(volatility.T.values)
    scaled = scaler.fit_transform(volatility.T.values).T
    return pd.DataFrame(scaled, columns=volatility.columns, index=volatility.index)

"""

def get_scaled_analysis(data_handler: DataFrameHandler, tickers: typing.List[str], start=20, stop=30, step=2):
    volatility = pd.DataFrame()
    for ticker in tickers:
        df = data_handler.download(ticker)
        volatility[ticker] = volatility_by_period(df=df, start=start, stop=stop, step=step)
    return scale_volatility(volatility)

def mean_analysis(scaled):
    return scaled.mean(skipna=True)

def pairs_clustering(analysis: pd.Series, aprop_kwargs=None):
    if aprop_kwargs is None:
        aprop_kwargs = dict(preference=-0.012, random_state=0)
    aprop = AffinityPropagation(**aprop_kwargs)

    tickers = list(analysis.index)
    X = [(0, x) for x in analysis.values]

    fit = aprop.fit(X)
    cluster_centers_indices = fit.cluster_centers_indices_
    labels = fit.labels_
    n_clusters = len(cluster_centers_indices)

    clusters = [[]]*n_clusters
    tickers = analysis.index
    for ticker, cluster in zip(tickers, labels):
        clusters[cluster] = clusters[cluster] + [ticker]

    return clusters


if __name__ == '__main__':
    from binance import Client
    from quick_trade.quick_trade.brokers import TradingClient
    import datetime
    from quick_trade.quick_trade.plots import ValidationAnalysisGraph, make_figure

    class BinanceTradingClient(Client, TradingClient):
        def get_data_historical(self,
                                ticker: str = None,
                                interval: str = '1m',
                                start_type: str = '%d %b %Y',
                                limit: int = 1000, ):
            print(ticker)
            try:
                with open('../../../../dataframes/' + ticker.replace('/', '') + f'{interval}.csv') as file:
                    df = pd.read_csv(file)
            except Exception as e:
                ticker = ticker.replace('/', '')
                start_date = datetime.datetime.strptime('5 Aug 2000', start_type)

                today = datetime.datetime.now()
                frames = self.get_historical_klines(ticker,
                                                    interval,
                                                    start_date.strftime("%d %b %Y %H:%M:%S"),
                                                    today.strftime("%d %b %Y %H:%M:%S"),
                                                    1000)
                data = pd.DataFrame(frames,
                                    columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av',
                                             'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
                df = data.astype(float)
                df.to_csv('../../../../dataframes/' + ticker.replace('/', '_') + f'{interval}.csv')
            return df


    client = BinanceTradingClient()
    frame_handler = DataFrameHandler(client)
    vol_handler = VolatilityHandler(data_handler=frame_handler)
    volatility = vol_handler.period_volatility('BTC/USDT')
    print(volatility, type(volatility))
    exit()

    tickers =['BTC/USDT',
              'MANA/USDT',
              'ETH/USDT',
              'LTC/USDT',
              'LUNA/USDT',
              'GALA/USDT',
              'BNB/USDT',
              'XRP/USDT',
              'ADA/USDT',
              'SOL/USDT',
              'AVAX/USDT',
              'BCH/USDT',
              'XMR/USDT',
              'SLP/USDT',
              'CELO/USDT',
              'BAT/USDT',
              'BAT/BTC',
              'FTM/USDT',
              'MATIC/USDT',
              'ATOM/USDT',
              'TRX/USDT',
              'NEAR/USDT',
              'UNI/USDT',
              'FTT/USDT',
              'BNB/USDT',
              'XLM/USDT',
              'EGLD/USDT',
              'SAND/USDT',
              'VET/USDT',
              'WAVES/USDT',
              'ZEC/USDT',
              'RUNE/USDT',
              'STX/USDT',
              #'NEXO/USDT',
              'DASH/USDT',
              'ANKR/USDT',
              'ZEN/USDT',
              'ICP/USDT',
              'SC/USDT',
              'IOST/USDT',
              'SKL/USDT',
              'SUSHI/USDT',
              'DGB/USDT',
              'LSK/USDT',
              'DOT/USDT',
              'LINK/USDT',
              ]
    volatility = get_scaled_analysis(DataFrameHandler(client), tickers, stop=30, start=20)
    volatility = mean_analysis(volatility)
    g = ValidationAnalysisGraph(make_figure(width=1400, height=700))
    clusters = pairs_clustering(volatility)
    print(clusters)
    colors = [
        'red', 'green', 'blue', 'yellow', '#7800FF', '#F700FF', '#FFFFFF', '#18D1B5', '#3E5EDF'
    ]
    for color, cluster in zip(colors, clusters):
        for ticker in cluster:
            g.plot_line([volatility[ticker]], index=[0], name=ticker, mode='markers', color=color, width=10)
    g.show()
