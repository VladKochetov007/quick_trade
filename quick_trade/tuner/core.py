from typing import Any
from typing import Dict
from typing import List
from numpy import isnan
from ..utils import logger


def transform_tunable_param(param_and_value: Dict[str, Any]):
    values = tuple(param_and_value.values())
    keys = tuple(param_and_value.keys())
    for e, (key, value) in enumerate(zip(keys, values)):
        if isinstance(value, TunableValue):
            valkeys = dict(zip(keys[e + 1:], values[e + 1:]))
            valkeys2 = dict(zip(keys[:e], values[:e]))  # without this element
            return [{key: val,
                     **valkeys,
                     **valkeys2} for val in set(value.values)]
    else:
        return [param_and_value]


def transform_tunable_params(strategies_kwargs: List[Dict[str, Any]]):
    list_params = []
    for param in strategies_kwargs:
        list_params.extend(transform_tunable_param(param))
    return list_params


def transform_all_tunable_values(strategies_kwargs: Dict[str, List[Dict[str, Any]]]):
    for strategy_name, strategy in zip(strategies_kwargs.keys(),
                                       strategies_kwargs.values()):
        strategies_kwargs[strategy_name] = transform_tunable_params(strategy)
    for cheker in strategies_kwargs.values():
        for param in cheker:
            for value in param.values():
                if isinstance(value, TunableValue):
                    return transform_all_tunable_values(strategies_kwargs)
    return strategies_kwargs

def resort_tunes(tunes: dict, sort_by: str = 'percentage year profit', drop_na: bool = True):
    if drop_na:
        for key, data in tunes.copy().items():
            if isnan(data[sort_by]):
                del tunes[key]
    tunes = {k: v for k, v in sorted(tunes.items(), key=lambda x: -x[1][sort_by])}
    logger.debug('tunes are sorted')
    return tunes


class TunableValue(object):
    def __init__(self, values):
        self.values = values
