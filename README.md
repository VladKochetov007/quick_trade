# quick_trade

VIEW DOCUMENTATION: https://vladkochetov007.github.io/quick_trade.github.io

![image](logo_with_slogan.PNG)

```
used:
 ├──ta by Darío López Padial (Bukosabino   https://github.com/bukosabino/ta)
 ├──tensorflow==2.2 (https://github.com/tensorflow/tensorflow)
 ├──pykalman (https://github.com/pykalman/pykalman)
 ├──plotly (https://github.com/plotly/plotly.py)
 ├──scipy (https://github.com/scipy/scipy)
 ├──logging
 ├──pandas (https://github.com/pandas-dev/pandas)
 ├──numpy (https://github.com/numpy/numpy)
 ├──itertools
 ├──datetime
 ├──os
 ├──scipy
 ├──python-binance
 └──iexfinance (https://github.com/addisonlynch/iexfinance)
```

Algo-trading system with python.

## customize your strategy!

```
import quick_trade.quick_trade.trading_sys as qtr
import quick_trade.quick_trade.utils. as qtrut
import yfinance as yf

class my_trader(qtr.Trader):
    def strategy_sell_and_hold(self):
        ret = []
        for i in self.df['Close'].values:
            ret.append(qtrut.SELL)
        self.returns = ret
        self.set_credit_leverages(1.0)
        self.set_open_stop_and_take()
        return ret


a = my_trader('MSFT', df=yf.download('MSFT', start='2019-01-01'))
a.set_pyplot()
a.set_client(qtrut.TradingClient())
a.strategy_sell_and_hold()
a.backtest()
```

*

1 -- Buy

2 -- Exit

0 -- Sell

*

## installing:

```
$ git clone https://github.com/VladKochetov007/quick_trade.git
$ pip3 install -r quick_trade/requirements.txt
```

or

```
$ pip3 install quick-trade
```

#### user_code example:

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

![image](https://i.ibb.co/mFLDJsX/IMG-5613.png)

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0"
src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br/>
<span  property="dct:title">quick_trade</span> by 
<a 
href="https://github.com/VladKochetov007" property="cc:attributionName"
rel="cc:attributionURL">Vlad Kochetov</a> is licensed under a <a rel="license"
href="http://creativecommons.org/licenses/by-sa/4.0/">
Creative Commons Attribution-ShareAlike 4.0 International License</a>.
<br />Based on a work at 
<a 
href="https://github.com/VladKochetov007/quick_trade"
rel="dct:source">https://github.com/VladKochetov007/quick_trade</a>.
<br />Permissions beyond the scope of this license may be available at
<a
        href="vladyslavdrrragonkoch@gmail.com"
        rel="cc:morePermissions">vladyslavdrrragonkoch@gmail.com</a>.
