import pandas as pd
from .utils import BUFFER_PRECISION_POINTER, INT_ALPHABET, IDENTIFIER_INT_BASE


def convert_base(num, to_base=128, from_base=10):
    n = 0
    for e, char in enumerate(num[::-1]):
        idx = INT_ALPHABET.index(char)
        n += idx * from_base ** e
    res = ""
    while n > 0:
        n, m = divmod(n, to_base)
        res += INT_ALPHABET[m]
    return res[::-1]

class Identifier(object):
    df: pd.DataFrame
    identifier: str

    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame({'Open': df['Open'],
                                'High': df['High'],
                                'Low': df['Low'],
                                'Close': df['Close']}).dropna()
        min_df = self.df.min(axis=0)
        self.df = self.df * BUFFER_PRECISION_POINTER / min_df

    def _format_candle(self, num: int = 0) -> str:
        candle = self.df.T[num].T
        candle = candle.values.astype(int)
        return f'{candle[0]}{candle[1]}{candle[2]}{candle[3]}'

    def get(self) -> str:
        length: int = len(self.df)
        first: str = self._format_candle(0)
        mid: str = self._format_candle(length//2)
        last: str = self._format_candle(length-1)
        identifier10: str = first+mid+last+str(length)
        self.identifier = convert_base(identifier10, to_base=IDENTIFIER_INT_BASE, from_base=10)
        return self.identifier

    def __repr__(self):
        return self.identifier

    def __str__(self):
        return self.identifier

def get_identifier(df: pd.DataFrame):
    identifier = Identifier(df)
    return identifier.get()
