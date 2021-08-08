# quick_trade
[![Downloads](https://static.pepy.tech/personalized-badge/quick-trade?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=PyPi%20downloads)](https://pepy.tech/project/quick-trade)
[![Downloads](https://static.pepy.tech/personalized-badge/quick-trade?period=month&units=none&left_color=grey&right_color=brightgreen&left_text=PyPi%20downloads%20(month))](https://pepy.tech/project/quick-trade)

View documentation: 🚧 https://vladkochetov007.github.io/quick_trade/#/ 🚧 in process

old documentation (V3 doc): https://vladkochetov007.github.io/quick_trade.github.io

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/logo_with_slogan_v2_fixed.PNG?raw=true)

```
used:
 ├──ta by Darío López Padial (Bukosabino   https://github.com/bukosabino/ta)
 ├──plotly (https://github.com/plotly/plotly.py)
 ├──pandas (https://github.com/pandas-dev/pandas)
 ├──numpy (https://github.com/numpy/numpy)
 ├──tqdm (https://github.com/tqdm/tqdm)
 └──ccxt (https://github.com/ccxt/ccxt)
```

Algo-trading system with python.

## Customize your strategy!

```python
import quick_trade.trading_sys as qtr
from quick_trade import brokers
from quick_trade.plots import QuickTradeGraph, make_figure
import yfinance as yf
import ccxt


class MyTrader(qtr.Trader):
    def strategy_sell_and_hold(self):
        ret = []
        for i in self.df['Close'].values:
            ret.append(qtr.utils.SELL)
        self.returns = ret
        self.set_credit_leverages(1.0)
        self.set_open_stop_and_take()
        return ret


a = MyTrader('MSFT/USD', df=yf.download('MSFT', start='2019-01-01'))
a.connect_graph(QuickTradeGraph(make_figure()))
a.set_client(brokers.TradingClient(ccxt.ftx()))
a.strategy_sell_and_hold()
a.backtest()
```

## Find the best strategy!

```python
import quick_trade.trading_sys as qtr
import ccxt
from quick_trade.quick_trade_tuner import *
from quick_trade.brokers import TradingClient


class Test(qtr.ExampleStrategies):  # examples of strategies
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
print(tuner.sort_tunes())
tuner.save_tunes('quick-trade-tunes.json')  # save tunes as JSON

```
You can also set rules for arranging arguments for each strategy by using `_RULES_` and `kwargs` to access the values of the arguments:

```python
params = {
    'strategy_3_sma':
        [
            dict(
                plot=False,
                slow=Choise([2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]),
                fast=Choise([2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]),
                mid=Choise([2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]),
                _RULES_='kwargs["slow"] > kwargs["mid"] > kwargs["fast"]'
            )
        ],
}
```

## Installing:

```commandline
$ git clone https://github.com/VladKochetov007/quick_trade.git
$ pip3 install -r quick_trade/requirements.txt
$ cd quick_trade
$ python3 setup.py install
$ cd ..
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
from quick_trade.plots import make_figure, QuickTradeGraph

client = brokers.TradingClient(ccxt.binance())
df = client.get_data_historical('BTC/USDT', '15m', 1000)
trader = qtr.ExampleStrategies('BTC/USDT', df=df, interval='15m')
trader.set_client(client)
trader.connect_graph(QuickTradeGraph(make_figure(height=731, width=1440, row_heights=[10, 5, 2])))
trader.strategy_2_sma(55, 21)
trader.backtest(deposit=1000, commission=0.075, bet=qtr.utils.np.inf)  # backtest on one pair
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

Use the strategy on real moneys. YES, IT'S FULLY AUTOMATED!

```python
import datetime
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.brokers import TradingClient
from quick_trade.plots import QuickTradeGraph, make_figure
import ccxt

ticker = 'MATIC/USDT'

start_time = datetime.datetime(2021,  # year
                               6,  # month
                               24,  # day

                               5,  # hour
                               16,  # minute
                               57)  # second (Leave a few seconds to download data from the exchange and strategy.)


class MyTrade(ExampleStrategies):
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
                 interval='1m',
                 df=client.get_data_historical(ticker, limit=10),
                 trading_on_client=True)
fig = make_figure()
graph = QuickTradeGraph(figure=fig)
trader.connect_graph(graph)
trader.set_client(client)

while True:
    if datetime.datetime.now() >= start_time:
        break

trader.realtime_trading(
    strategy=trader.strategy,
    ticker=ticker,
    coin_lotsize_division=True,
    limit=100,
    ignore_exceptions=False,
    wait_sl_tp_checking=5
)

```

![image](https://github.com/VladKochetov007/quick_trade/blob/master/img/realtime_example.png?raw=true)

# Help the project

BTC: ```33BEsCUi1C3NbnoNda32z9S5pXdaHBq6gA```

ETH ERC20: ```0x064fb4Dfe9c91913FE1847AAB04146b9757E4066```

USDT ERC20: ```0xc33AfF803aCdf08a1D0b8B4939E75598aB2518c1```

USDT TRC20: ```TDvegWDM7bvZthw8Z8Su6CsE8zasVGhWNa```

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
