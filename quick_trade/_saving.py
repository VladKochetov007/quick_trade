from utils import BUFFER_PATH
from json import dump, load

class JSON(object):
    path: str

    def __init__(self, filepath: str):
        self.path = filepath

    def read(self):
        with open(self.path, 'r') as file:
            data = load(file)
        return data

    def write(self, data):
        with open(self.path, 'w') as file:
            dump(data, file)

def read_json(path: str):
    return JSON(filepath=path).read()

def write_json(path: str, data):
    return JSON(filepath=path).write(data=data)


class SaveTrader(object)
    pass # TODO: .
