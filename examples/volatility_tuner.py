from quick_trade.quick_trade_tuner.avoid_overfitting.volatility import split_tickers_volatility, Tuner
from ccxt import binance
from quick_trade.brokers import TradingClient
from quick_trade.quick_trade_tuner.tuner import QuickTradeTuner, GeometricProgression
from quick_trade.trading_sys import ExampleStrategies


tickers = ['BTC/USDT',
           'MANA/USDT',
           'ETH/USDT',
           'LTC/USDT',
           'LUNA/USDT',
           'GALA/USDT',
           'BNB/USDT',
           'XRP/USDT',
           'ADA/USDT',
           'SOL/USDT',
           'AVAX/USDT',
           'BCH/USDT',
           'XMR/USDT',
           'SLP/USDT',
           'CELO/USDT',
           'BAT/USDT']

groups = split_tickers_volatility(tickers)

params = {
    'strategy_parabolic_SAR':
        [
            {
                'step': GeometricProgression(0.01, 0.2, 1.1),
                'max_step': GeometricProgression(0.1, 2.0, 1.1),
                'plot': False
            }
        ],
}

tuner = Tuner(client=TradingClient(binance()),
              clusters=groups,
              intervals=['1h'],
              limits=[1000],
              tuner_instance=QuickTradeTuner,
              strategies_kwargs=params)
print(tuner._tuners)
tuner.tune(ExampleStrategies,
           commission=0.075)
