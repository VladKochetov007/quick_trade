from json import dump, load
from os.path import isfile


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

class Buffer:
    def __init__(self, buffer_data=None):
        if buffer_data is None:
            self._buffer = {}
        else:
            self._buffer = buffer_data

    def write(self, key, data):
        self._buffer[key] = data

    def __contains__(self, item):
        return item in self._buffer.keys()

    def read(self, key):
        return self._buffer[key]

def read_json(path: str):
    return JSON(filepath=path).read()

def write_json(path: str, data, indent: int = 2):
    return JSON(filepath=path).write(data=data, indent=indent)
