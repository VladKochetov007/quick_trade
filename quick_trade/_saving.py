from .utils import BUFFER_PATH, BUFFER_INDENT
from json import dump, load
from os.path import isfile

class JSON(object):
    path: str

    def __init__(self, filepath: str):
        self.path = filepath
        if not isfile(path=filepath):
            self.write(data={})

    def read(self):

        with open(self.path, 'r') as file:
            data = load(file)
        return data

    def write(self, data, indent: int = 2):
        with open(self.path, 'w') as file:
            dump(data, file, indent=indent)

def read_json(path: str):
    return JSON(filepath=path).read()

def write_json(path: str, data, indent: int = 2):
    return JSON(filepath=path).write(data=data, indent=indent)


class SaveTrader(object):
    def __init__(self):
        pass

def append(trader):
    file = JSON(filepath=BUFFER_PATH)
    trader_short: dict = trader.to_dict()
    trader_key = list(trader_short.keys())[0]
    trader_val = list(trader_short.values())[0]
    data: dict = file.read()
    data[trader_key] = trader_val
    file.write(data, indent=BUFFER_INDENT)