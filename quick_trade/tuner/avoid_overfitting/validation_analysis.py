import pandas as pd
from ..tuner import QuickTradeTuner
from typing import Union
from ..._saving import read_json
from ..core import resort_tunes
from ...plots import ValidationAnalysisGraph
from typing import Dict, Iterable, List, Any
from ...brokers import TradingClient
from copy import deepcopy


class Analyzer(object):
    train_tunes: dict
    fig: ValidationAnalysisGraph
    frame: pd.DataFrame

    def __init__(self,
                 train: Union[dict, str, QuickTradeTuner],
                 val: Union[dict, str, QuickTradeTuner],
                 sort_by: str = 'percentage year profit'):
        if isinstance(train, QuickTradeTuner):
            self.train_tunes = train.result_tunes
        elif isinstance(train, str):
            self.train_tunes = read_json(train)
        elif isinstance(train, dict):
            self.train_tunes = train

        if isinstance(val, QuickTradeTuner):
            self.validation_tunes = val.result_tunes
        elif isinstance(val, str):
            self.validation_tunes = read_json(val)
        elif isinstance(val, dict):
            self.validation_tunes = val

        self.resort(sort_by=sort_by)

    def resort(self, sort_by: str = 'profit/deviation ratio', drop_na: bool = True):
        self.train_tunes = resort_tunes(self.train_tunes, sort_by=sort_by, drop_na=drop_na)
        self.profit_keys = list(self.train_tunes.keys())[::-1]
        self.sorted_by = sort_by

    def generate_frame(self):
        self.frame = pd.DataFrame(index=self.profit_keys)
        self.frame['train'] = [self.train_tunes[key][self.sorted_by] for key in self.profit_keys]
        self.frame['validation'] = [self.validation_tunes[key][self.sorted_by] for key in self.profit_keys]

    def connect_graph(self, figure: ValidationAnalysisGraph):
        self.fig.connect_analyzer(self)

    def plot_frame(self):
        self.fig.plot_frame()

    def strategy_by_number(self, num: int):
        return self.profit_keys[num]

def slice_frame(df: pd.DataFrame, validation_split: float = 0.3) -> Dict[str, pd.DataFrame]:
    train_val_limit = round(len(df)*(1-validation_split))
    return {'train': df[:train_val_limit],
            'val': df[train_val_limit:]}


class ValidationTuner:
    sort_by: str
    analyzer: Analyzer

    def __init__(self,
                 client: TradingClient,
                 tickers: Iterable[str],
                 intervals: Iterable[str],
                 limits: Iterable,
                 strategies_kwargs: Dict[str, List[Dict[str, Any]]] = None,
                 multi_backtest: bool = True,
                 validation_split: float = 0.3,
                 tuner_instance=QuickTradeTuner):
        class _Tuner(tuner_instance):
            type: str
            validation_split: float

            def config(self, type: str, validation_split: float = 0.3):
                self.type = type
                self.validation_split = validation_split

            def _get_df(self, ticker: str, interval: str, limit):
                frame = super(_Tuner, self)._get_df(ticker=ticker, interval=interval, limit=limit)
                return slice_frame(df=frame, validation_split=self.validation_split)[self.type]

        self.train_tuner = _Tuner(client=client,
                                  tickers=tickers,
                                  intervals=intervals,
                                  limits=limits,
                                  strategies_kwargs=strategies_kwargs,
                                  multi_backtest=multi_backtest)
        self.val_tuner = deepcopy(self.train_tuner)
        self.train_tuner.config(type='train', validation_split=validation_split)
        self.val_tuner.config(type='val', validation_split=validation_split)

    def tune(self,
             trading_class,
             use_tqdm: bool = True,
             update_json: bool = True,
             val_json_path: str = 'val_returns.json',
             train_json_path: str = 'train_returns.json',
             sort_by: str = 'profit/deviation ratio',
             **backtest_kwargs):
        self.sort_by = sort_by
        tune_kwargs_train = dict(trading_class=trading_class,
                                 use_tqdm=use_tqdm,
                                 update_json=update_json,
                                 update_json_path=train_json_path,
                                 **backtest_kwargs)

        tune_kwargs_val = tune_kwargs_train.copy()
        tune_kwargs_val['update_json_path'] = val_json_path

        self.train_tuner.tune(**tune_kwargs_train)
        self.val_tuner.tune(**tune_kwargs_val)

        self.train_tuner.sort_tunes(sort_by=sort_by)
        self.val_tuner.sort_tunes(sort_by=sort_by)

        self.val_tuner.save_tunes(val_json_path)
        self.train_tuner.save_tunes(train_json_path)

    def make_analyzer(self) -> Analyzer:
        self.analyzer = Analyzer(train=self.train_tuner,
                                 val=self.val_tuner,
                                 sort_by=self.sort_by)
        return self.analyzer
