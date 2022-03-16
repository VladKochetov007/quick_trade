from .tuner import *
import re
from ast import literal_eval


def __bests_2_cfg_single(bests):
    config = {}
    for i, (strategy_meta, _) in enumerate(bests):
        strategy_with_kw = re.search(r":: [\S ]+$", strategy_meta).group()[3:]
        tickers = re.search(r"ticker: (([A-Z0-9]+/[A-Z0-9]+)| )+, ", strategy_meta).group()[8:-2]

        tickers = tickers.split(' ')
        strategy_name = re.search(r"^.+\(", strategy_with_kw).group()[:-1]
        strategy_kwargs = strategy_with_kw[len(strategy_name):]

        replace_config = {'(': '{"',
                          ')': '}',
                          '=': '": ',
                          ', ': ', "'}
        for before, after in replace_config.items():
            strategy_kwargs = strategy_kwargs.replace(before, after)

        strategy_kwargs = literal_eval(strategy_kwargs)

        settings = {strategy_name: strategy_kwargs}
        for ticker in tickers:
            if i == 0:
                config[ticker] = [settings]
            else:
                config[ticker].append(settings)
    return config


def bests_to_config(bests):  # automatic config generation for the tester from bests result of tuner
    if isinstance(bests, list):
        return __bests_2_cfg_single(bests=bests)
    config = {}
    for tickers, single_bests in bests.items():
        config = {**config, **__bests_2_cfg_single(single_bests)}
    return config
