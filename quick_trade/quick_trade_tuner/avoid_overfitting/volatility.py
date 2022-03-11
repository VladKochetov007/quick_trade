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
        analysis = pd.DataFrame()
        for ticker in tickers:
            analysis[ticker] = self.period_volatility(ticker)
        return analysis


class VolatilityScaler:
    def __init__(self, volatility: pd.DataFrame):
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._volatility = volatility
        self.__fit()

    def __fit(self):
        self._scaler.fit(self._volatility.T.values)

    def scaled_analysis(self):
        scaled = self._scaler.fit_transform(self._volatility.T.values).T
        self.scaled_volatility = pd.DataFrame(scaled,
                                              columns=self._volatility.columns,
                                              index=self._volatility.index)
        return self.scaled_volatility

    def mean_scaled_analysis(self):
        self.mean_scaled_volatility = self.scaled_analysis().mean(skipna=True)
        return self.mean_scaled_volatility

class Clusterizer:
    def __init__(self, afprop_kwargs=None):
        clusterer_kwargs = afprop_kwargs
        if afprop_kwargs is None:
            clusterer_kwargs = dict(preference=-0.012, random_state=0)
        self.__clusterer = AffinityPropagation(**clusterer_kwargs)

    def make_clusters(self, scaled_volatility: pd.DataFrame):
        pairs = [(0, volatility) for volatility in scaled_volatility.values]

        self.__fit = self.__clusterer.fit(pairs)
        self._clusters_centers = self.__fit.cluster_centers_indices_
        labels = self.__fit.labels_
        n_clusters = len(self._clusters_centers)

        self.clusters = [[]] * n_clusters
        tickers = scaled_volatility.index

        for ticker, cluster in zip(tickers, labels):
            self.clusters[cluster] = self.clusters[cluster] + [ticker]

        return self.clusters

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
    data_handler = DataFrameHandler(client)
    vol_handler = VolatilityHandler(data_handler)
    vol_scaler = VolatilityScaler(vol_handler.multipair_volatility(tickers))
    volatility = vol_scaler.mean_scaled_analysis()
    clusterer = Clusterizer()
    clusters = clusterer.make_clusters(volatility)
    print(clusters)
    g = ValidationAnalysisGraph(make_figure(width=1400, height=700))
    colors = [
        'red', 'green', 'blue', 'yellow', '#7800FF', '#F700FF', '#FFFFFF', '#18D1B5', '#3E5EDF'
    ]
    for color, cluster in zip(colors, clusters):
        for ticker in cluster:
            g.plot_line([volatility[ticker]], index=[0], name=ticker, mode='markers', color=color, width=10)
    g.show()
