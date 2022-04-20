from quick_trade.trading_sys import ExampleStrategies
from quick_trade.plots import make_trader_figure, TraderGraph
from custom_client import BinanceTradingClient

ticker = 'SOL/BTC'
timeframe = '1h'

figure = make_trader_figure(height=600, width=900)
graph = TraderGraph(figure=figure)
client = BinanceTradingClient()

df = client.get_data_historical(ticker=ticker, interval=timeframe)

trader = ExampleStrategies(ticker=ticker, interval=timeframe, df=df)

trader.connect_graph(graph)

trader.strategy_price_channel()
trader.inverse_strategy()  # trend strategy
trader.backtest(deposit=300, commission=0.075)
