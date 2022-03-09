import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_volatility(df, period):
    roll = df.rolling(period)
    mini = roll.min()['Low'].values
    maxi = roll.max()['High'].values
    mean = (roll.mean())
    mean = mean.mean(axis=1).values

    return pd.Series((maxi - mini) / mean)

def get_mean_volatility(df, period):
    return pd.Series(get_volatility(df, period)).mean()

def volatility_by_period(df, start=1, stop=100, step=5):
    volatility_correlation = []
    for period in range(start, stop, step):
        volatility_correlation.append(get_mean_volatility(df, period=period))
    return pd.Series(volatility_correlation, index=range(start, stop, step))

def scale_volatility(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit(df.T.values)
    scaled = scaler.fit_transform(df.T.values).T
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


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


    def get_df(ticker, timeframe, client):
        df = client.get_data_historical(ticker, timeframe)
        df = pd.DataFrame(
            {
                'Open': df['Open'],
                'High': df['High'],
                'Low': df['Low'],
                'Close': df['Close'],
            }
        )
        return df

    client = BinanceTradingClient()

    def add_volatility(ticker, key=55, result=pd.DataFrame()):
        frame = get_df(ticker, '1d', client)
        result[ticker] = volatility_by_period(frame, key, key+1)
        return result

    for ticker in ['BTC/USDT', 'MANA/USDT', 'ETH/USDT', 'LTC/USDT', 'LUNA/USDT', 'GALA/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT', 'BCH/USDT', 'EUR/USDT', 'USDT/UAH', 'DOGE/USDT', 'XMR/USDT', 'SLP/USDT', 'CELO/USDT', 'BAT/USDT', 'BAT/BTC', 'FTM/USDT']:
        volatility = add_volatility(ticker)

    g = ValidationAnalysisGraph(make_figure(width=1400, height=700))
    volatility = scale_volatility(volatility)
    for ticker_vol in volatility.items():
        g.plot_line(ticker_vol[1], name=ticker_vol[0], width=20, index=ticker_vol[1].index, mode='markers')
    g.show()
