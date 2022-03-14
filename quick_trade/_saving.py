from json import dump, load
import os
import re


class JSON(object):
    path: str

    def __init__(self, filepath: str):
        self.path = filepath
        if not os.path.isfile(path=filepath):
            self.write(data={})

    def read(self):
        with open(self.path, 'r', encoding='utf-8') as file:
            data = load(file)
        return data

    def write(self, data, indent: int = 2):
        with open(self.path, 'w', encoding='utf-8') as file:
            dump(data, file, indent=indent, ensure_ascii=False)

class Buffer:
    _buffer: dict
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

    def load_from_json(self, path):
        self._buffer = read_json(path=path)

    def save_to_json(self, path):
        write_json(path=path,
                   data=self._buffer)

    def __len__(self):
        return len(self._buffer)

    def keys(self):
        return list(self._buffer.keys())

    def values(self):
        return list(self._buffer.values())

def read_json(path: str):
    return JSON(filepath=path).read()

def write_json(path: str, data, indent: int = 2):
    return JSON(filepath=path).write(data=data, indent=indent)

def check_make_dir(filepath, split_pattern=r"/[^\s/]*$"):
    filename = str(re.findall(split_pattern, filepath)[-1])
    dirpath = filepath[:-len(filename)]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath
