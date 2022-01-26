from .utils import BUFFER_PATH, BUFFER_INDENT, recursive_dict
from json import dump, load
from os.path import isfile
from collections import defaultdict


class JSON(object):
    path: str

    def __init__(self, filepath: str):
        self.path = filepath
        if not isfile(path=filepath):
            self.write(data={})

    def read(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            data = load(file)
        return data

    def write(self, data, indent: int = 2):
        with open(self.path, 'w', encoding='utf-8') as file:
            dump(data, file, indent=indent, ensure_ascii=False)

def read_json(path: str):
    return JSON(filepath=path).read()

def write_json(path: str, data, indent: int = 2):
    return JSON(filepath=path).write(data=data, indent=indent)

def save_trader(trader):
    file = JSON(filepath=BUFFER_PATH)
    data: defaultdict = recursive_dict(base=file.read())
    data[trader.ticker][trader.interval][trader.identifier][trader._registered_strategy] = {
        'deposit_history': trader.deposit_history,
        'winrate': trader.winrate,
        'trades': trader.trades,
        'losses': trader.losses,
        'profits': trader.profits,
        'year_profit': trader.year_profit,
        'mean_deviation': trader.mean_deviation,
        'sharpe_ratio': trader.sharpe_ratio,
        'sortino_ratio': trader.sortino_ratio,
        'calmar_ratio': trader.calmar_ratio,
        'max_drawdown': trader.max_drawdown,
        'profit_deviation_ratio': trader.profit_deviation_ratio,
    }
    file.write(data, indent=BUFFER_INDENT)
