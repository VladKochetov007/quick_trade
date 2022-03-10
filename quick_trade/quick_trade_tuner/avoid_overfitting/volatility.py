import pprint
import typing

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation


def prepare_frame(df):
    df = pd.DataFrame(
        {
            'Open': df['Open'],
            'High': df['High'],
            'Low': df['Low'],
            'Close': df['Close'],
        }
    )
    return df

def get_daily(client, ticker):
    df = client.get_data_historical(ticker=ticker, interval='1d')
    return prepare_frame(df)

def get_volatility(df, period):
    roll = df.rolling(period)
    mini = roll.min()['Low'].values
    maxi = roll.max()['High'].values
    mean = (roll.mean())
    mean = mean.mean(axis=1).values

    return pd.Series((maxi - mini) / mean)

def get_mean_volatility(df, period):
    return pd.Series(get_volatility(df, period)).mean()

def volatility_by_period(df, start=20, stop=30, step=2):
    volatility_correlation = []
    for period in range(start, stop, step):
        volatility_correlation.append(get_mean_volatility(df, period=period))
    return pd.Series(volatility_correlation, index=range(start, stop, step))

def scale_volatility(volatility: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit(volatility.T.values)
    scaled = scaler.fit_transform(volatility.T.values).T
    return pd.DataFrame(scaled, columns=volatility.columns, index=volatility.index)

def get_scaled_analysis(client, tickers, start=20, stop=30, step=2):
    volatility = pd.DataFrame()
    for ticker in tickers:
        df = get_daily(client, ticker)
        df = prepare_frame(df)
        volatility[ticker] = volatility_by_period(df=df, start=start, stop=stop, step=step)
    return scale_volatility(volatility)

def mean_analysis(scaled):
    return scaled.mean(skipna=True)

def pairs_clustering(analysis: pd.Series, aprop_kwargs=None):
    if aprop_kwargs is None:
        aprop_kwargs = dict(preference=-0.02, random_state=0)
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
            try:
                with open('../../../../dataframes/' + ticker.replace('/', '') + f'{interval}.csv') as file:
                    df = pd.read_csv(file)
                    # print('in dir')
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
    tickers =['BTC/USDT', 'MANA/USDT', 'ETH/USDT', 'LTC/USDT', 'LUNA/USDT', 'GALA/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT', 'BCH/USDT', 'EUR/USDT', 'USDT/UAH', 'DOGE/USDT', 'XMR/USDT', 'SLP/USDT', 'CELO/USDT', 'BAT/USDT', 'BAT/BTC', 'FTM/USDT', 'SHIB/USDT']
    volatility = get_scaled_analysis(client, tickers, stop=30, start=20)
    volatility = mean_analysis(volatility)
    g = ValidationAnalysisGraph(make_figure(width=1400, height=700))
    pprint.pprint(pairs_clustering(volatility))
    for ticker_vol in volatility.items():
        g.plot_line([ticker_vol[1]], name=ticker_vol[0], width=20, index=[0], mode='markers')
        #print(ticker_vol[1], ', #', ticker_vol[0])
    g.show()
