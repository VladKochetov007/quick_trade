from warnings import warn

import numpy as np
from typing import Dict, Iterable, Any, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation
import ccxt
from ..._saving import Buffer, check_make_dir
from ...brokers import TradingClient
from copy import deepcopy
from ..tuner import QuickTradeTuner


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
        self._mean_scaled_volatility = self.scaled_analysis().mean(skipna=True)
        return self._mean_scaled_volatility

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

class Tuner:
    def __init__(self,
                 client: TradingClient,
                 clusters: List[List[str]] = None,
                 intervals: Iterable[str] = None,
                 limits: Iterable = None,
                 tuner_instance=QuickTradeTuner,
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None,
                 initialize_tuners: bool = True):
        self._tuners = []
        self._buffer = Buffer()
        self._tuner_instance_ = tuner_instance
        self.clusters = []
        self.client = deepcopy(client)
        if clusters is None:
            clusters = []
        for cluster in clusters:
            if len(cluster) > 1 and initialize_tuners:
                self.clusters.append(cluster)
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
        self.n_groups = len(self._tuners)

    def __run_task(self, task, kwargs=None, kwargs_dynamic=None):
        if kwargs_dynamic is None:
            kwargs_dynamic = [{}] * self.n_groups
        if kwargs is None:
            kwargs = dict()
        for n_cluster, (tuner, cluster) in enumerate(zip(self._tuners, self.clusters)):
            task(tuner, **kwargs, **kwargs_dynamic[n_cluster])
            self._buffer.write(' '.join(cluster), tuner.result_tunes)
        self._buffer.save_to_json(self.__global_tuner_path)

    def __format_path_dynamic(self, path: str, param: str) -> List[Dict[str, str]]:
        dynamic = []
        for cluster in range(self.n_groups):
            dynamic.append({param: path.format(cluster)})
        return dynamic

    def _update_path(self, filepath):
        self.__dir_path = check_make_dir(filepath)
        self.__global_tuner_path = self.__dir_path + "/volatility_tuner_global.json"

    def __tuners_from_buffer(self):
        buffer_keys = self._buffer.keys()
        self.tickers = ' '.join(buffer_keys).split(' ')
        self.clusters = [cluster.split(' ') for cluster in buffer_keys]
        self.n_groups = len(self.clusters)
        self._tuners = [self._tuner_instance_(deepcopy(self.client), self.tickers, multi_backtest=True) for i in range(self.n_groups)]
        dynamic = [{'data': data} for data in self._buffer.values()]
        self.__run_task(self._tuner_instance_.load_tunes, kwargs_dynamic=dynamic)

    def tune(self,
             trading_class,
             use_tqdm: bool = True,
             update_json: bool = True,
             update_json_path: str = 'volatility_tuner/returns-{}.json',
             **backtest_kwargs):

        dynamic = self.__format_path_dynamic(update_json_path, param='update_json_path')
        self._update_path(update_json_path)
        self.__run_task(self._tuner_instance_.tune,
                        dict(trading_class=trading_class,
                             use_tqdm=use_tqdm,
                             update_json=update_json,
                             **backtest_kwargs),
                        kwargs_dynamic=dynamic)

    def resorting(self, sort_by: str = 'percentage year profit', drop_na: bool = True):
        self.__run_task(self._tuner_instance_.resorting,
                        kwargs=dict(sort_by=sort_by,
                                    drop_na=drop_na))

    def load_tunes(self, path: str = 'volatility_tuner/returns-{}.json', data=None):
        self._update_path(path)
        if data is None:
            self._buffer.load_from_json(self.__global_tuner_path)
        else:
            self._buffer = Buffer(buffer_data=data)
        self.__tuners_from_buffer()

    def sort_tunes(self, sort_by: str = 'percentage year profit', drop_na: bool = True):
        self.__run_task(self._tuner_instance_.sort_tunes,
                        kwargs=dict(sort_by=sort_by,
                                    drop_na=drop_na))

    def save_tunes(self, path: str = 'returns.json'):
        warn('tunes will be saved automatically')

    def resorting(self, sort_by: str = 'percentage year profit', drop_na: bool = True):
        self.__run_task(self._tuner_instance_.resorting,
                        kwargs=dict(sort_by=sort_by,
                                    drop_na=drop_na))

    def get_best(self, num: int = 1) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        self.bests = {}
        for cluster, tuner in zip(self.clusters, self._tuners):
            self.bests[' '.join(cluster)] = tuner.get_best(num=num)
        return self.bests

    def get_worst(self, num: int = 1) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        self.bests = {}
        for cluster, tuner in zip(self.clusters, self._tuners):
            self.bests[' '.join(cluster)] = tuner.get_worst(num=num)
        return self.bests

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
    return clusrerizer.make_clusters(scaler.mean_scaled_analysis())
