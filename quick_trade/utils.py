import datetime as dt
import itertools
import json
import logging
import os

import numpy as np
import pandas as pd
from binance.client import Client
from iexfinance.stocks import get_historical_intraday

R = '#ff0000'
G = '#55ff00'
B = '#0015ff'
C = 'cyan'
INPUTS = 60
BUY = 1
SELL = 0
EXIT = 2

TOKEN = 'Tpk_a4bc3e95d4c94810a3b2d4138dc81c5d'

os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = TOKEN
__author__ = 'Vlad Kochetov'
__credits__ = ["Hemerson Tacon -- Stack overflow",
               "hpaulj -- Stack overflow"]
__version__ = "2.1.9"
__install_requires = [
    'iexfinance==0.4.3',
    'plotly==4.9.0',
    'ta==0.5.25',
    'scikit-learn==0.23.1',
    'tensorflow==2.2.0',
    'pykalman==0.9.5',
    'scipy==1.4.1',
    'tqdm==4.48.0',
    'numpy==1.18.5',
    'pandas==1.0.5',
]

TICKER = '^DJI'
SCATTER_SIZE = 12
SCATTER_ALPHA = 1
TAKE_STOP_OPN_WIDTH = 1
TEXT_COLOR = 'white'
SUB_LINES_WIDTH = 3
STOP_TAKE_OPN_ALPHA = 1
COLOR_DEPOSIT = 'white'
SCATTER_DEPO_ALPHA = SCATTER_ALPHA
DEPO_COLOR_UP = 'green'
DEPO_COLOR_DOWN = 'red'
logger = logging.getLogger()
logger.setLevel(50)
logging.basicConfig(level=20, filename='trading.log')


def set_(data):
    ret = list(data.copy())
    for e, i in enumerate(data[1:]):
        if i == data[e]:
            ret[e + 1] = np.nan
    return ret


def to_4_col_df(data, *columns):
    """
    data:  |  array-like  |  data to converting

    returns:
    pd.DataFrame with 4 your columns

    """
    predict = [[] for _ in range(4)]
    for it in range(4):
        for i in range(it, len(data), 4):
            predict[it].append(data[i])
    return pd.DataFrame(predict, index=columns).T


def inverse_4_col_df(df, columns):
    df = pd.DataFrame(df)
    ret = df.values
    rets = []
    for column in columns:
        rets.append(
            pd.DataFrame(itertools.chain.from_iterable(ret), columns=[column]))
    return pd.concat(rets, axis=1)


def get_data(ticker, undo_days):
    tuples = []
    end = dt.datetime.today()
    date_list = [end - dt.timedelta(days=d) for d in range(undo_days)]
    for i in range(len(date_list)):
        ret = str(np.array(date_list[::-1])[i])[:10]
        ret = ret.split('-')
        for e, dat in enumerate(ret):
            ret[e] = int(dat)
        tuples.append(tuple(ret))

    dataframes = []
    for date_ in tuples:
        dat = dt.datetime(*date_)
        df_ = get_historical_intraday(ticker, date=dat, output_format='pandas')
        dataframes.append(df_)
    returns = pd.concat(dataframes, axis=0)
    returns = pd.DataFrame({
        'High': returns['high'].values,
        'Low': returns['low'].values,
        'Open': returns['open'].values,
        'Close': returns['close'].values,
        'Volume': returns['volume'].values
    })
    return returns.dropna()


def anti_set_(seted):
    ret = [seted[0]]
    flag = seted[0]
    for e, i in enumerate(seted[1:]):
        if i is np.nan:
            ret.append(flag)
        else:
            ret.append(i)
            flag = i
    return ret


def move_stop_to_breakeven(stop, open_, sig, price, diff, *args, **kwargs):
    if sig == BUY and price > open_ and diff > 0:
        stop_loss = (price * 2 + open_) / 3
    elif sig == SELL and price < open_ and diff < 0:
        stop_loss = (price * 2 + open_) / 3
    else:
        stop_loss = stop
    return stop_loss


def digit(data):
    data = list(data)
    ret = []
    for element in list(data):
        if element == 0:
            ret.append(EXIT)
        elif element > 0:
            ret.append(BUY)
        else:
            ret.append(SELL)
    return ret


def get_window(values, window):
    ret = []
    for e, i in enumerate(values[:len(values) - window + 1]):
        ret.append(values[e:e + window])
    return ret


def nothing(ret):
    return ret


def get_binance_data(ticker="BNBBTC", interval="1m", date_index=False, start_str='Tue Jun 10 07:20:33 2020'):
    data = Client().get_historical_klines(symbol=ticker, interval=interval, start_str=start_str)
    df = pd.DataFrame(data)
    df.columns = ["open_time",
                  "Open", "High", "Low", 'Close', 'Volume',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    for column in ["Open", "High", "Low", 'Close', 'Volume']:
        df[column] = df[column].astype(float)
    if date_index:
        df.index = [dt.datetime.fromtimestamp(i / 1000) for i in df.close_time]
    return df
