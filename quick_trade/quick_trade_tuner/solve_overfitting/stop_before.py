import pandas as pd
from ..tuner import QuickTradeTuner
from typing import Union
from ..._saving import read_json
from ..core import resort_tunes
from ...plots import StopBeforeGraph

class Analyzer(object):
    train_tunes: dict
    fig: StopBeforeGraph
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

        if isinstance(train, QuickTradeTuner):
            self.validation_tunes = val.result_tunes
        elif isinstance(train, str):
            self.validation_tunes = read_json(val)
        elif isinstance(train, dict):
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

    def connect_graph(self, figure: StopBeforeGraph):
        self.fig = figure
        self.fig.connect_analyzer(self)
