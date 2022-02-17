from ._code_inspect import format_arguments
from functools import wraps
from numpy import inf


def strategy(strat):
    @wraps(strat)
    def wrapped(self, *args, **kwargs):
        self._registered_strategy = format_arguments(func=strat, args=args, kwargs=kwargs)
        self.returns = []
        self._converted = []
        self.deposit_history = []
        self.stop_losses = []
        self.take_profits = []
        self.open_lot_prices = []
        if self._use_buffer:
            self.load()
        if not self._successful_load:
            strategy_output = strat(self, *args, **kwargs)
            self.returns_update()
            if not len(self.stop_losses):
                self.set_open_stop_and_take(set_take=False)
            if not len(self.take_profits):
                self.set_open_stop_and_take(set_stop=False)
            self.correct_sl_tp(sl_correction=inf,
                               tp_correction=inf)
            return strategy_output

    return wrapped
