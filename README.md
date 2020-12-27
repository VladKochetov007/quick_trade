# quick_trade

VIEW DOCUMENTATION: https://vladkochetov007.github.io/quick_trade.github.io

P.S.: documentation now in v3.0

![image](https://github.com/VladKochetov007/quick_trade/blob/master/logo_with_slogan.PNG?raw=true)

```
used:
 ├──ta by Darío López Padial (Bukosabino   https://github.com/bukosabino/ta)
 ├──tensorflow (https://github.com/tensorflow/tensorflow)
 ├──pykalman (https://github.com/pykalman/pykalman)
 ├──plotly (https://github.com/plotly/plotly.py)
 ├──scipy (https://github.com/scipy/scipy)
 ├──pandas (https://github.com/pandas-dev/pandas)
 ├──numpy (https://github.com/numpy/numpy)
 ├──itertools
 ├──datetime
 ├──python-binance
 └──iexfinance (https://github.com/addisonlynch/iexfinance)
```

Algo-trading system with python.

## customize your strategy!

```
import quick_trade.quick_trade.trading_sys as qtr
import yfinance as yf

class My_trader(qtr.Trader):
    def strategy_sell_and_hold(self):
        ret = []
        for i in self.df['Close'].values:
            ret.append(qtr.utils.SELL)
        self.returns = ret
        self.set_credit_leverages(1.0)
        self.set_open_stop_and_take()
        return ret


a = My_trader('MSFT', df=yf.download('MSFT', start='2019-01-01'))
a.set_pyplot()
a.set_client(qtr.BinanceTradingClient())
a.strategy_sell_and_hold()
a.backtest()
```

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
import quick_trade.trading_sys as qtr
import yfinance as yf


df = yf.download(tickers='XRP-USD', period='5d', interval='1m')
trader = qtr.Trader('XRP-USD', df=df, interval='1m')
trader.set_client(qtr.TradingClient())
trader.set_pyplot()
trader.strategy_parabolic_SAR()
trader.set_credit_leverages(1)
trader.backtest(commission=0.075)
```

## output plotly chart:

![image](https://i.ibb.co/NyxbsV2/Unknown-2.png)

## output print

```
losses: 226
trades: 460
profits: 232
mean year percentage profit: 9075.656014641549%
winrate: 50.43478260869565%
```

![image](https://i.ibb.co/mFLDJsX/IMG-5613.png)

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" 
src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br/>
<span  property="dct:title">quick_trade</span> by 
<a 
href="https://github.com/VladKochetov007" 
rel="cc:attributionURL">Vlad Kochetov</a> is licensed under a <a 
href="http://creativecommons.org/licenses/by-sa/4.0/">
Creative Commons Attribution-ShareAlike 4.0 International License</a>.
<br />Based on a work at 
<a 
rel="dct:source">https://github.com/VladKochetov007/quick_trade</a>.
<br />Permissions beyond the scope of this license may be available at
<a
        href="vladyslavdrrragonkoch@gmail.com"
        rel="cc:morePermissions">vladyslavdrrragonkoch@gmail.com</a>.
