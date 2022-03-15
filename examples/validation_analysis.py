from quick_trade.quick_trade_tuner.avoid_overfitting.validation_analysis import ValidationTuner, Analyzer
from quick_trade.quick_trade_tuner.tuner import Arange
from custom_client import BinanceTradingClient
from quick_trade.trading_sys import ExampleStrategies
from quick_trade import strategy
from quick_trade.plots import ValidationAnalysisGraph


class MyTrader(ExampleStrategies):
    @strategy
    def strategy_bollinger_breakout(self,
                                    to_mid: bool = False,
                                    to_opposite: bool = False,
                                    window=80,
                                    window_dev=1):
        super().strategy_bollinger_breakout(plot=False,
                                            to_opposite=to_opposite,
                                            to_mid=to_mid,
                                            window=window,
                                            window_dev=window_dev)
        return self.returns


params = {
    'strategy_bollinger_breakout':
        [
            dict(
                window=Arange(1, 10, 1),
                window_dev=1
            )
        ],
}

t = ValidationTuner(BinanceTradingClient(),
                    tickers=['BTC/USDT', 'ETH/USDT', 'ETC/USDT', 'LTC/USDT'],
                    intervals=['1h'],
                    limits=[1000],
                    strategies_kwargs=params,
                    multi_backtest=True,
                    validation_split=1/3)
t.tune(MyTrader, commission=0.075, val_json_path='val.json', train_json_path='train.json')
validator = Analyzer(train='train.json',
                     val='val.json',
                     sort_by='calmar ratio')
fig = ValidationAnalysisGraph()
fig.connect_analyzer(validator)
validator.generate_frame()
validator.plot_frame()
validator.fig.show()
