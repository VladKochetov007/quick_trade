import numpy as np
from typing import Dict, Iterable, Any, List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation
import ccxt
from ..._saving import Buffer
from ...brokers import TradingClient
from copy import deepcopy


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

    def multipair_volatility(self, tickers: Iterable[str]):
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

class Tuner:  # TODO
    def __init__(self,
                 tuner_instance,
                 client: TradingClient,
                 clusters: List[List[str]],
                 intervals: Iterable[str],
                 limits: Iterable,
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None):
        self._tuners = []
        for cluster in clusters:
            if len(cluster) > 1:
                self._tuners.append(
                    tuner_instance(
                        client=deepcopy(client),
                        tickers=cluster,
                        intervals=intervals,
                        limits=limits,
                        strategies_kwargs=strategies_kwargs,
                        multi_backtest=True
                    )
                )

def split_tickers_volatility(tickers: List[str],
                             client=None,
                             timeframe: str = '1d',
                             span_start: int = 20,
                             span_end: int = 30,
                             span_step: int = 2,
                             afprop_kwargs=None):
    df_handler = DataFrameHandler(client=client)
    volatility_handler = VolatilityHandler(data_handler=df_handler,
                                           span_start=span_start,
                                           span_end=span_end,
                                           span_step=span_step)
    scaler = VolatilityScaler(volatility_handler.multipair_volatility(tickers=tickers))
    clusrerizer = Clusterizer(afprop_kwargs=afprop_kwargs)
    return clusrerizer.make_clusters(scaler.mean_scaled_volatility())
