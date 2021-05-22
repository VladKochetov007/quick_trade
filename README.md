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
 └──iexfinance (https://github.com/addisonlynch/iexfinance)
```

Algo-trading system with python.

## customize your strategy!

```
import quick_trade.trading_sys as qtr
from quick_trade import brokers
import yfinance as yf
import ccxt

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
a.set_client(brokers.TradingClient(ccxt.binance()))
a.strategy_sell_and_hold()
a.backtest()
```

## find the best strategy!
```
import quick_trade.trading_sys as qtr
import ta.trend
import ta.momentum
import ta.volume
import ta.others
import ta.volatility
import ccxt
from quick_trade.quick_trade_tuner.tuner import *
class Test(qtr.Trader):
    def strategy_bollinger_break(self, **kwargs):
        self.strategy_bollinger(plot=True, **kwargs)
        self.inverse_strategy(swap_tpop_take=False)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        self.convert_signal()
        return self.returns

    def bb(self, **kwargs):
        self.strategy_bollinger(plot=False, **kwargs)
        self.set_open_stop_and_take()
        self.set_credit_leverages()
        self.convert_signal()
        return self.returns

    def strategy_supertrend1(self, plot: bool = False, *st_args, **st_kwargs):
        self.strategy_supertrend(plot=plot, *st_args, **st_kwargs)
        self.set_credit_leverages()
        self.convert_signal()
        return self.returns

    def macd(self, histogram=False, **kwargs):
        if not histogram:
            self.strategy_macd(**kwargs)
        else:
            self.strategy_macd_histogram_diff(**kwargs)
        self.set_credit_leverages()
        self.convert_signal()
        return self.returns

    def psar(self, **kwargs):
        self.strategy_parabolic_SAR(plot=False, **kwargs)
        self.set_credit_leverages()
        self.convert_signal()
        return self.returns

    def strategy_200ema(self):
        values = self.df['Close']
        self.returns = []
        ema = ta.trend.ema_indicator(values, 200)
        for close, ma in zip(values.values, ema):
            if close>ma:
                self.returns.append(qtr.utils.BUY)
            else:
                self.returns.append(qtr.utils.SELL)
        return self.returns

    def strategy_rsi(self, len=7, **kwargs):
        rsi = ta.momentum.rsi(self.df['Close'], len)
        self.returns = []
        for rsi_ in rsi:
            if rsi_ > 50:
                self.returns.append(qtr.utils.BUY)
            elif rsi_ < 50:
                self.returns.append(qtr.utils.SELL)
        self.set_open_stop_and_take()
        self.set_credit_leverages(1)
        return self.returns

params = {
    'strategy_supertrend1':
        [
            {
                'multiplier': Linspace(0.5, 22, 5)
            }
        ],
    'strategy_bollinger_break':
        [
            {
                'to_mid': Choise([False, True]),
                'window': Arange(10, 350, 30)
            }
        ],
    'bb':
        [
            {
                'to_mid': Choise([False, True]),
                'window': Arange(10, 350, 30)
            }
        ],
    'macd':
        [
            {
                'slow': Linspace(10, 100, 3),
                'fast': Linspace(3, 60, 3),
                'histogram': Choise([False, True])
            }
        ],
    'psar':
        [
            {
                'step': 0.01,
                'max_step': 0.1
            },
            {
                'step': 0.02,
                'max_step': 0.2
            },
            {
                'step': 0.03,
                'max_step': 0.3
            },
            {
                'step': 0.04,
                'max_step': 0.4
            },
            {
                'step': 0.05,
                'max_step': 0.5
            },
            {
                'step': 0.08,
                'max_step': 0.8
            },
            {
                'step': 0.12,
                'max_step': 1.2
            }
        ]

}

tuner = QuickTradeTuner(
        TradingClient(ccxt.binance()),
        ['BTCUSDT', 'OMGUSDT', 'XRPUSDT'],
        ['15m', '5m],
        [1000, 700],
        params
    )

tuner.tune(Test)
print(tuner.sort_tunes()) # you can save it as json
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

# Help the project

BTC: ```15TC5wjgxTj5btKr75DBswsdafUqXmkAjp```

ETH ERC20: ```0xbc7d8d7fb6ccc0963a7bf5eb41faf8e4bb546740```

USDT TRC20: ```TCb9nWdApXmrfQuPcChdP9FpBXjXuFDFNX```

# License

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
