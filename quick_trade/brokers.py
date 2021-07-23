from typing import Dict

from ccxt import Exchange
from pandas import DataFrame

from . import utils


class TradingClient(object):
    ordered: bool = False
    __side__: str
    ticker: str
    cls_open_orders: int = 0
    start_balance: Dict[str, float]
    base: str
    quote: str

    def __init__(self, client: Exchange):
        self.client = client
        self._update_balances()

    @utils.wait_success
    def order_create(self,
                     side: str,
                     ticker: str = 'None',
                     quantity: float = 0.0):
        self._update_balances()
        utils.logger.info('quantity: %f, side: %s, ticker: %s', quantity, side,
                          ticker)
        if side == 'Buy':
            self.client.create_market_buy_order(symbol=ticker, amount=quantity)
        elif side == 'Sell':
            self.client.create_market_sell_order(symbol=ticker, amount=quantity)
        self.__side__ = side
        self.ticker = ticker
        self.base = ticker.split('/')[0]
        self.quote = ticker.split('/')[1]
        self.ordered = True
        self._add_order_count()

    @utils.wait_success
    def get_ticker_price(self,
                         ticker: str) -> float:
        return float(self.client.fetch_ticker(symbol=ticker)['close'])

    def new_order_buy(self,
                      ticker: str = None,
                      quantity: float = 0.0,
                      credit_leverage: float = 1.0):
        self.order_create('Buy',
                          ticker=ticker,
                          quantity=quantity * credit_leverage)

    def new_order_sell(self,
                       ticker: str = None,
                       quantity: float = 0.0,
                       credit_leverage: float = 1.0):
        self.order_create('Sell',
                          ticker=ticker,
                          quantity=quantity * credit_leverage)

    @utils.wait_success
    def get_data_historical(self,
                            ticker: str = None,
                            interval: str = '1m',
                            limit: int = 1000):

        frames = self.client.fetch_ohlcv(ticker,
                                         interval,
                                         limit=limit)
        data = DataFrame(frames,
                         columns=['time', 'Open', 'High', 'Low', 'Close',
                                  'Volume'])
        return data.astype(float)

    def exit_last_order(self):
        if self.ordered:
            utils.logger.info('client exit')
            base_balance = self.get_balance_ticker(self.base)
            bet = base_balance - self.start_balance[self.base]
            if self.__side__ == 'Sell':
                self.new_order_buy(self.ticker,
                                   bet)
            elif self.__side__ == 'Buy':
                self.new_order_sell(self.ticker,
                                    bet)
            self.__side__ = 'Exit'
            self.ordered = False
            self._sub_order_count()

    @utils.wait_success
    def get_balance_ticker(self, ticker: str) -> float:
        return self.client.fetch_free_balance()[ticker]

    @classmethod
    def _add_order_count(cls):
        cls.cls_open_orders += 1
        utils.logger.info('new order')

    @classmethod
    def _sub_order_count(cls):
        cls.cls_open_orders -= 1
        utils.logger.info('order closed')

    @utils.wait_success
    def _update_balances(self):
        if len(self.client.apiKey):
            self.start_balance = self.client.fetch_free_balance()
        else:
            utils.logger.warning('Client doesn\'t have an API key')
