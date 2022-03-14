from binance.client import Client  # pip3 install python-binance
from quick_trade import utils
from quick_trade.brokers import TradingClient
import pandas as pd
import datetime

class BinanceTradingClient(Client, TradingClient):
    @utils.wait_success
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1m',
                            start_type: str = '%d %b %Y'):
        try:
            with open('dataframes/' + ticker.replace('/', '') + f'{interval}.csv') as file:
                df = pd.read_csv(file)
        except:
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
            df.to_csv('dataframes/' + ticker.replace('/', '_') + f'{interval}.csv')
        return df
