# quick_trade

View documentation: 🚧 https://vladkochetov007.github.io/quick_trade/#/ 🚧 in process

old documentation (V3 doc): https://vladkochetov007.github.io/quick_trade.github.io

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/logo_with_slogan_v2_fixed.PNG?raw=true)

```
used:
 ├──ta by Darío López Padial (Bukosabino   https://github.com/bukosabino/ta)
 ├──plotly (https://github.com/plotly/plotly.py)
 ├──pandas (https://github.com/pandas-dev/pandas)
 ├──numpy (https://github.com/numpy/numpy)
 └──ccxt (https://github.com/ccxt/ccxt)
```

Algo-trading system with python.

## Customize your strategy!

```python
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


a = My_trader('MSFT/USD', df=yf.download('MSFT', start='2019-01-01'))
a.set_pyplot()
a.set_client(brokers.TradingClient(ccxt.ftx()))
a.strategy_sell_and_hold()
a.backtest()
```

## Find the best strategy!

```python
import quick_trade.trading_sys as qtr
import ccxt
from quick_trade.quick_trade_tuner.tuner import *


class Test(qtr.Trader):
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


params = {
    'strategy_supertrend1':
        [
            {
                'multiplier': Linspace(0.5, 22, 5)
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
            }
        ]

}

tuner = QuickTradeTuner(
    TradingClient(ccxt.binance()),
    ['BTC/USDT', 'OMG/USDT', 'XRP/USDT'],
    ['15m', '5m'],
    [1000, 700, 800, 500],
    params
)

tuner.tune(Test)
print(tuner.sort_tunes())  # you can save it as json
```

## Installing:

```commandline
$ git clone https://github.com/VladKochetov007/quick_trade.git
$ pip3 install -r quick_trade/requirements.txt
```

or

```commandline
$ pip3 install quick-trade
```

## User's code example (backtest)

```python
import quick_trade.trading_sys as qtr
import ccxt
from quick_trade import brokers

client = brokers.TradingClient(ccxt.binance())
df = client.get_data_historical('BTC/USDT', '15m', 1000)
trader = qtr.Trader('BTC/USDT', df=df, interval='15m')
trader.set_client(client)
trader.set_pyplot(height=731, width=1440, row_heights=[10, 5, 2])
trader.strategy_2_sma(55, 21)
trader.backtest(deposit=1000, commission=0.075, bet=qtr.utils.np.inf)  # backtest on one pare
```

## Output plotly chart:

![image](https://raw.githubusercontent.com/VladKochetov007/quick_trade/master/img/plot.png)

## Output print

```
losses: 7
trades: 16
profits: 9
mean year percentage profit: 541.9299012354617%
winrate: 56.25%
```

## Run strategy

Use strategy on real moneys. YES, IT'S FULLY AUTOMATED!

```python
import datetime
from quick_trade.trading_sys import Trader
from quick_trade.brokers import TradingClient
import ccxt

ticker = 'MATIC/USDT'

start_time = datetime.datetime(2021,  # year
                               6,  # month
                               16,  # day

                               15,  # hour
                               59,  # minute
                               55)  # second (Leave a few seconds to download data from the exchange and strategy.)


class MyTrade(Trader):
    def strategy(self):
        self.strategy_supertrend(multiplier=2, length=1, plot=False)
        self.convert_signal()
        self.set_credit_leverages(1)
        self.sl_tp_adder(10)
        return self.returns


keys = {'apiKey': 'your binance api key',
        'secret': 'your binance secret key'}  # or any other exchange
client = TradingClient(ccxt.binance(config=keys))

trader = MyTrade(ticker=ticker,
                 interval='5m',
                 df=client.get_data_historical(ticker, limit=10),
                 trading_on_client=True)
trader.set_pyplot()
trader.set_client(client)
while True:
    if datetime.datetime.utcnow() >= start_time:
        break
trader.realtime_trading(
    strategy=trader.strategy,
    ticker=ticker,
    coin_lotsize_division=True,
    limit=2,
    ignore_exceptions=False,
    wait_sl_tp_checking=5
)

```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/realtime_example.png?raw=true)


# Help the project

BTC: ```15TC5wjgxTj5btKr75DBswsdafUqXmkAjp```

ETH ERC20:  ```0xbc7d8d7fb6ccc0963a7bf5eb41faf8e4bb546740```

USDT ERC20: ```0xbc7d8d7fb6ccc0963a7bf5eb41faf8e4bb546740```

USDT TRC20: ```TCb9nWdApXmrfQuPcChdP9FpBXjXuFDFNX```

### My telegram

```
@VladKochetov07
```

## Donations:

[0xbc7d8d7fb6ccc0963a7bf5eb41faf8e4bb546740](https://etherscan.io/address/0xbc7d8d7fb6ccc0963a7bf5eb41faf8e4bb546740)

# License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">
quick_trade</span>
by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/VladKochetov007" property="cc:attributionName" rel="cc:attributionURL">
Vladyslav Kochetov</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
Creative Commons Attribution-ShareAlike 4.0 International License</a>.