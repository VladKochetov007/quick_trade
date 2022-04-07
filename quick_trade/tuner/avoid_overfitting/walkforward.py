import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...tuner import bests_to_config
from ..tuner import QuickTradeTuner
from ...trading_sys import Trader
from ...brokers import TradingClient
from ... import utils


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


def _static_data_historical_client(instance, df):
    class _Client(instance):
        def __init__(self, *args, **kwargs): pass

        @staticmethod
        def get_data_historical(*args, **kwargs):
            return df
    return _Client()


class WalkForward:
    _df: pd.DataFrame
    year_profit: float
    mean_deviation: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    profit_deviation_ratio: float
    average_growth: np.ndarray

    def __load_df(self, ticker, timeframe):
        self._df = self._client.get_data_historical(ticker=ticker,
                                                    interval=timeframe)
        self._df = self._df.reset_index()

    def __total_chs_by_ch_length(self):
        if not self.chunk_length:
            self.chunk_length = len(self._df) // self._total_chunks
        else:
            self._total_chunks = len(self._df) // self.chunk_length

    def __prepare_df(self):
        self.__total_chs_by_ch_length()
        total_length = self.chunk_length * self._total_chunks
        self._df = self._df[-total_length:]

    def _make_samples(self):
        IS_dataframes = []
        OOS_dataframes = []

        for IS, OOS in zip(range(0,
                                 self._total_chunks - self._outofsample_chunks,
                                 self._outofsample_chunks),
                           range(self._insample_chunks,
                                 self._total_chunks,
                                 self._outofsample_chunks)):
            IS_start = IS * self.chunk_length
            IS_end = (IS + self._insample_chunks) * self.chunk_length

            OOS_start = (OOS - self._indent_chunks) * self.chunk_length
            OOS_end = (OOS + self._outofsample_chunks) * self.chunk_length

            IS_dataframes.append(
                self._df[IS_start:IS_end])
            OOS_dataframes.append(
                self._df[OOS_start:OOS_end]
            )

        return IS_dataframes, OOS_dataframes

    def __init__(self,
                 client: TradingClient,
                 total_chunks: int = 10,
                 insample_chunks: int = 3,
                 outofsample_chunks: int = 1,
                 testing_indent_chunks: int = 1,
                 chunk_length=None):
        assert not (total_chunks - insample_chunks) % outofsample_chunks or chunk_length

        self._total_chunks = total_chunks
        self._insample_chunks = insample_chunks
        self._outofsample_chunks = outofsample_chunks
        self._indent_chunks = testing_indent_chunks
        self.chunk_length = chunk_length

        self._client = client

        self.total_equity = []

    def run_analysis(self,
                     ticker: str,
                     timeframe: str,
                     config=None,
                     tuner_instance=QuickTradeTuner,
                     trader_instance=Trader,
                     sort_by: str = 'percentage year profit',
                     commission=0,
                     bet=np.inf,
                     use_tqdm: bool = True):
        self.ticker = ticker
        self.timeframe = timeframe
        self.__load_df(ticker=self.ticker,
                       timeframe=self.timeframe)
        self.__prepare_df()

        self.total_equity = []

        samples = zip(*self._make_samples())
        if use_tqdm:
            IS_length = len(self._make_samples()[0])
            bar = tqdm(total=IS_length)

        for IS_data, OOS_data in samples:
            IS_client = _static_data_historical_client(self._client.__class__, IS_data)
            OOS_client = _static_data_historical_client(self._client.__class__, OOS_data)

            tuner = tuner_instance(client=IS_client,
                                   tickers=[self.ticker],
                                   intervals=[self.timeframe],
                                   strategies_kwargs=config,
                                   multi_backtest=False)

            trader = trader_instance(ticker=ticker,
                                     interval=timeframe)
            trader.set_client(OOS_client)

            IS = InSample(tuner=tuner)
            OOS = OutOfSample(trader=trader)

            IS.run(trading_class=trader_instance)
            OOS.run(config=IS.get_settings(sort_by=sort_by),
                    commission=commission,
                    bet=bet)

            oos_equity = OOS.equity()[self._indent_chunks*self.chunk_length:]
            multipliers = utils.get_multipliers(pd.Series(oos_equity))
            self.total_equity.extend(multipliers)
            if use_tqdm:
                bar.update(1)
        self.total_equity = list(np.cumprod(self.total_equity))
        self._update_info()

    def equity(self):
        return self.total_equity

    def _update_info(self):
        stats = utils.strategy_characteristics(equity=self.equity(),
                                               timeframe=self.timeframe)
        for name, param in {**utils.TUNER_CODECONF,
                            **utils.ADDITIONAL_TRADER_ATTRIBUTES}.items():
            try:
                setattr(self, param, stats[name])
            except Exception as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise exc

    def info(self) -> str:
        return utils.INFO_TEXT.format(
            None,
            None,
            None,
            self.year_profit,
            None,
            self.mean_deviation,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.max_drawdown,
            self.profit_deviation_ratio
        )
