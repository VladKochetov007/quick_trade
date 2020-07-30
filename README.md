# quick_trade
### used ta by Darío López Padial (Bukosabino   https://github.com/bukosabino/ta) and tensorflow (https://github.com/tensorflow/tensorflow)

[![button](https://i.ibb.co/MgWmjsY/imgonline-com-ua-Resize-y-Wu-Bc-Rv7-KGALSc-Iw.jpg)](https://www.donationalerts.com/r/vladkochetov007)


algo-trading system.
trading with python.


## customize your strategy!

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
## installing:
```
$ git clone https://github.com/VladKochetov007/quick_trade.git
```

## your project tree:
```
project
 ├── quick_trade
 │    ├── quick_trade
 │    │    ├── __init__.py
 │    │    ├── trading_sys.py
 │    │    └── utils.py
 │    ├── LICENSE.txt
 │    ├── README.md
 │    └── setup.py
 └── user_code.py
```

#### user code example:
```
import quick_trade.trading_sys as qt
import yfinance as yf
a = qt.PatternFinder('AAPL', df=yf.download('AAPL', period='1y'))
a.set_pyplot()
a.strategy_macd()
a.backtest()
```

## output plot:
<div align="left">
  <img src="https://i.ibb.co/ThYVwpq/imgonline-com-ua-Big-Picture-afe-Xd-HJoldw-Tp.jpg" width=900">
</div>
