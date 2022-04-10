from quick_trade.trading_sys import ExampleStrategies
from custom_client import BinanceTradingClient

class Client(BinanceTradingClient):
    def get_data_historical(self,
                            ticker: str = None,
                            limit: int = 1000,
                            interval: str = '1m',
                            start_type: str = '%d %b %Y'):
        return super().get_data_historical(ticker=ticker,
                                           interval=interval,
                                           start_type=start_type)[-limit:]


tickers = ['BTC/USDT',
           'ETH/USDT',
           'BAT/USDT',
           'SOL/USDT',
           'MANA/USDT']
timeframe = '1m'
config = [{'strategy_pump_detector': dict(period=60,
                                         take_profit=300,
                                         stop_loss=200,
                                         points=500)}]

client = Client()
trader = ExampleStrategies(interval=timeframe,)
trader.connect_graph()
trader.set_client(client)

trader.multi_backtest(
    test_config={ticker: config for ticker in tickers},
    commission=0.1,
    limit=100_000
)
