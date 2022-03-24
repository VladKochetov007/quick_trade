import numpy as np

from ...tuner import bests_to_config
from ..tuner import QuickTradeTuner
from ...trading_sys import Trader

class InSample:
    def __init__(self, tuner: QuickTradeTuner):
        self._tuner = tuner

    def run(self, trading_class):
        self._tuner.tune(trading_class=trading_class,
                         use_tqdm=False,
                         update_json=False)

    def get_settings(self, sort_by: str = 'percentage year profit'):
        self._tuner.sort_tunes(sort_by=sort_by)
        return bests_to_config(self._tuner.get_best())

class OutOfSample:
    def __init__(self, trader: Trader):
        self._trader = trader

    def run(self, config, bet=np.inf, commission=0):
        self._trader.multi_backtest(commission=commission,
                                    plot=False,
                                    print_out=False,
                                    show=False,
                                    test_config=config)

    def equity(self):
        return self._trader.deposit_history

class WalkForward:
    def __init__(self,
                 trader: Trader,
                 ticker: str,
                 timeframe: str,
                 total_chunks: int = 10,
                 insample_chunks: int = 3,
                 outofsample_chunks: int = 1):
        assert not (total_chunks - insample_chunks) % outofsample_chunks
        history_length = trader.client.get_data_historical(ticker=ticker,
                                                           interval=timeframe)
        chunk_length = history_length // total_chunks


    def run_analysis(self, tuner: QuickTradeTuner):
        pass # TODO
