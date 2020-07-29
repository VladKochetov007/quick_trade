# quick_trade

algo-trading system.
trading with python.


customize your strategy!

```
from quick_trade.trading_sys import PatternFinder
import yfinance as yf

class my_trader(PatternFinder):
    def strategy_buy_and_hold(
            self):
        ret = []
        for i in self.df['Close'].values:
            ret.append(1)
        self.returns = ret
        return ret


a = my_trader('MSFT', df=yf.download('MSFT', start='2019-01-01'))
a.set_pyplot()
a.strategy_buy_and_hold()
a.backtest()
```