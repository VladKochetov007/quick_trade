import datetime
import time
import typing
from typing import Dict

import ftx
import pandas as pd
import requests
from binance.client import Client
from quick_trade import utils


class TradingClient(object):
    ordered: bool = False
    __side__: str
    quantity: float
    ticker: str
    order: Dict[str, typing.Any]

    def order_create(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def get_ticker_price(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    @staticmethod
    def get_data(*args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def new_order_buy(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def new_order_sell(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def get_data_historical(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def exit_last_order(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')

    def get_balance_ticker(self, *args, **kwargs):
        utils.logger.fatal('non set trading client method')


class BinanceTradingClient(Client, TradingClient):
    def order_create(self,
                     side: str,
                     ticker: str = 'None',
                     quantity: float = 0.0,
                     credit_leverage: float = 1.0,
                     *args,
                     **kwargs):
        if '_moneys_' in kwargs:
            if quantity > kwargs['_moneys_']:
                quantity -= utils.min_admit(kwargs['rounding_bet'])
            quantity -= utils.min_admit(kwargs['rounding_bet'])
            quantity = round(quantity, kwargs['rounding_bet'])
            utils.logger.info(f'client: quantity: {quantity}, moneys: {kwargs["_moneys_"]}, side: {side}')
        else:
            utils.logger.info(f'quantity: {quantity}, side: {side}')
        if side == 'Buy':
            self.order = self.order_market_buy(symbol=ticker, quantity=quantity)
        elif side == 'Sell':
            self.order = self.order_market_sell(symbol=ticker, quantity=quantity)
        self.order_id = self.order['orderId']
        self.quantity = quantity
        self.__side__ = side
        self.ticker = ticker
        self.ordered = True

    def get_ticker_price(self,
                         ticker: str) -> float:
        return float(self.get_symbol_ticker(symbol=ticker)['price'])

    @staticmethod
    def get_data(ticker: str = None,
                 interval: str = None,
                 **get_kw) -> pd.DataFrame:
        return utils.get_binance_data(ticker, interval, **get_kw)

    def new_order_buy(self,
                      ticker: str = None,
                      quantity: float = 0.0,
                      credit_leverage: float = 1.0,
                      logging=True,
                      *args,
                      **kwargs):
        self.order_create('Buy',
                          ticker=ticker,
                          quantity=quantity,
                          credit_leverage=credit_leverage,
                          *args,
                          **kwargs)
        if logging:
            utils.logger.info('client buy')

    def new_order_sell(self,
                       ticker: str = None,
                       quantity: float = 0.0,
                       credit_leverage: float = 1.0,
                       logging=True,
                       *args,
                       **kwargs):
        self.order_create('Sell',
                          ticker=ticker,
                          quantity=quantity,
                          credit_leverage=credit_leverage,
                          *args,
                          **kwargs)
        if logging:
            utils.logger.info('client sell')

    def get_data_historical(self,
                            ticker: str = None,
                            start: str = '25 Dec 2020',
                            interval: str = '1m',
                            limit: int = 1000,
                            start_type: str = '%d %b %Y'):
        start_date = datetime.datetime.strptime(start, start_type)
        today = datetime.datetime.now()

        frames = self.get_historical_klines(ticker,
                                            interval,
                                            start_date.strftime("%d %b %Y %H:%M:%S"),
                                            today.strftime("%d %b %Y %H:%M:%S"),
                                            limit)
        data = pd.DataFrame(frames,
                            columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        data.set_index('timestamp', inplace=True)
        return data.astype(float)

    def exit_last_order(self):
        if self.ordered:
            if self.__side__ == 'Sell':
                self.new_order_buy(self.ticker, self.quantity, logging=False)
            elif self.__side__ == 'Buy':
                self.new_order_sell(self.ticker, self.quantity, logging=False)
            self.__side__ = 'Exit'
            self.ordered = False
            utils.logger.info('client exit')

    def get_balance_ticker(self, ticker: str) -> float:
        for asset in self.get_account()['balances']:
            if asset['asset'] == ticker:
                utils.logger.debug(f'client balance {asset["free"]} {ticker}')
                return float(asset['free'])


class FTXTradingClient(TradingClient, ftx.FtxClient):
    _session: requests.Session
    _api_key: str
    _base_url: str
    _api_secret: str
    _subaccount_name: str

    def __init__(self, api_key: str, api_secret: str, subaccount_name: str = None):
        ftx.FtxClient.__init__(self=self,
                               base_url='https://ftx.com/api/',
                               api_key=api_key,
                               api_secret=api_secret,
                               subaccount_name=subaccount_name)

    def order_create(self,
                     side: str,
                     ticker: str = None,
                     quantity: float = 0.0,
                     credit_leverage: float = 1.0,
                     *args,
                     **kwargs):
        if '_moneys_' in kwargs:
            if quantity > kwargs['_moneys_']:
                quantity -= utils.min_admit(kwargs['rounding_bet'])
            quantity -= utils.min_admit(kwargs['rounding_bet'])
            quantity = round(quantity, kwargs['rounding_bet'])
            utils.logger.info(f'client: quantity: {quantity}, moneys: {kwargs["_moneys_"]}, side: {side}')
        else:
            utils.logger.info(f'quantity: {quantity}, side: {side}')
        if side == 'Buy':
            self.order = self.place_order(market=ticker, type='market', price=None, side='buy', size=quantity)
        elif side == 'Sell':
            self.order = self.place_order(market=ticker, type='market', price=None, side='sell', size=quantity)
        self.quantity = quantity
        self.__side__ = side
        self.ticker = ticker
        self.ordered = True

    def get_ticker_price(self, ticker: str = None, *args, **kwargs):
        return self.get_data_historical(ticker=ticker, limit=1)['Close'].values[0]

    def get_data(self,
                 ticker: str = 'None',
                 start: int = None,
                 interval: str = '1m',
                 limit: int = 5000,
                 *args,
                 **kwargs):
        return self.get_data_historical(ticker=ticker,
                                        start=start,
                                        interval=interval,
                                        limit=limit)

    def new_order_buy(self,
                      ticker: str = None,
                      quantity: float = 0.0,
                      credit_leverage: float = 1.0,
                      logging=True,
                      *args,
                      **kwargs):
        self.order_create('Buy',
                          ticker=ticker,
                          quantity=quantity * self.get_ticker_price(ticker=ticker),
                          # At the FTX exchange, when buying, the price is in currency № 2, and when selling in currency № 1.
                          credit_leverage=credit_leverage,
                          *args,
                          **kwargs)
        if logging:
            utils.logger.info('client buy')

    def new_order_sell(self,
                       ticker: str = None,
                       quantity: float = 0.0,
                       credit_leverage: float = 1.0,
                       logging=True,
                       *args,
                       **kwargs):
        self.order_create('Sell',
                          ticker=ticker,
                          quantity=quantity,
                          credit_leverage=credit_leverage,
                          *args,
                          **kwargs)
        if logging:
            utils.logger.info('client sell')

    def get_data_historical(self,
                            ticker: str = None,
                            start: int = None,
                            interval: str = '1m',
                            limit: int = 5000,
                            *args,
                            **kwargs):
        if interval == '15s':
            interval = 15
        elif interval == '1m':
            interval = 60
        elif interval == '5m':
            interval = 300
        elif interval == '15m':
            interval = 900
        elif interval == '1h':
            interval = 3600
        elif interval == '4h':
            interval = 14400
        elif interval == '1d':
            interval = 86400
        else:
            raise ValueError('I N C O R R E C T   I N T E R V A L')
        ret = pd.DataFrame(self.get_historical_data(market_name=ticker,
                                                    resolution=interval,
                                                    limit=limit,
                                                    end_time=int(time.time()),
                                                    start_time=start))
        utils.logger.debug(f'ret in get_data_historical: \n {ret}')
        ret.columns = ['Close', 'High', 'Low', 'Open', 'startTime', 'time', 'Volume']
        return ret

    def exit_last_order(self, *args, **kwargs):
        if self.ordered:
            if self.__side__ == 'Sell':
                self.new_order_buy(self.ticker, self.quantity, logging=False)
            elif self.__side__ == 'Buy':
                self.new_order_sell(self.ticker, self.quantity, logging=False)
            self.__side__ = 'Exit'
            self.ordered = False
            utils.logger.info('client exit')

    def get_balance_ticker(self, ticker: str) -> float:
        for request in self.get_balances():
            if request['coin'] == ticker:
                return float(request['free'])
