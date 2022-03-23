from quick_trade.tuner.avoid_overfitting.volatility import split_tickers_volatility, Tuner
from quick_trade.tuner.tuner import QuickTradeTuner, Arange, Choise
from quick_trade.trading_sys import ExampleStrategies
from custom_client import BinanceTradingClient


tickers = ['BTC/USDT', 'MANA/USDT', 'ETH/USDT', 'LTC/USDT', 'LUNA/USDT', 'GALA/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT', 'BCH/USDT', 'XMR/USDT', 'SLP/USDT', 'CELO/USDT', 'BAT/USDT', 'FTM/USDT']


groups = split_tickers_volatility(tickers)

params = {
    'strategy_bollinger_breakout':
        [
            {
                'window': Arange(10, 200, 5),
                'window_dev': Choise([0.5, 1, 1.5]),
                'plot': False
            }
        ],
}

tuner = Tuner(client=BinanceTradingClient(),
              clusters=groups,
              intervals=['1h'],
              limits=[1000],
              tuner_instance=QuickTradeTuner,
              strategies_kwargs=params)
tuner.tune(ExampleStrategies,
           commission=0.075,
           update_json_path="volatility_tuner_all_binance_history/returns-{}.json")
tuner.sort_tunes()
