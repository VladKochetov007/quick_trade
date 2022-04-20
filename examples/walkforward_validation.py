import copy
import pprint

from quick_trade.tuner.avoid_overfitting.walkforward import WalkForward, _static_data_historical_client
from custom_client import BinanceTradingClient
from quick_trade.plots import BasePlotlyGraph, make_figure
from quick_trade.tuner.avoid_overfitting.validation_analysis import slice_frame
from quick_trade.trading_sys import ExampleStrategies
from quick_trade.tuner import Arange, Linspace, QuickTradeTuner


ticker = 'BTC/USDT'
timeframe = '1h'
validation_split = 0
sort_by = 'profit/deviation ratio'

fig = BasePlotlyGraph(make_figure(rows=2))
client = BinanceTradingClient()
df = client.get_data_historical(ticker, interval=timeframe)


chunk_len = len(df)//15

splits = slice_frame(df, validation_split=validation_split)
df_train = splits['train']
df_val = splits['val']

train_client = _static_data_historical_client(BinanceTradingClient, df_train)
WF_train = WalkForward(client=train_client,
                       chunk_length=chunk_len)

val_client = _static_data_historical_client(BinanceTradingClient, df_val)
WF_val = WalkForward(client=val_client,
                     chunk_length=chunk_len)

tuner_configs = [
    {
        'strategy_bollinger_breakout':
            [
                 {'window_dev': 1,
                  'plot': False,
                  'window': Arange(5, 200, 10)}
            ]
    }
]
for cfg in tuner_configs:
    pprint.pprint(cfg)
    print('\n'*3)

results = []
trains = []

for i, config in enumerate(copy.deepcopy(tuner_configs)):
    WF_train.run_analysis(ticker=ticker,
                          timeframe=timeframe,
                          config=config,
                          tuner_instance=QuickTradeTuner,
                          trader_instance=ExampleStrategies,
                          sort_by=sort_by,
                          commission=0.12)
    results.append(WF_train.profit_deviation_ratio)
    trains.append(WF_train)

best_index = results.index(max(results))
best_config = tuner_configs[best_index]
print(trains[best_index].info())

if validation_split:
    WF_val.run_analysis(ticker=ticker,
                        timeframe=timeframe,
                        config=best_config,
                        tuner_instance=QuickTradeTuner,
                        trader_instance=ExampleStrategies,
                        sort_by=sort_by,
                        commission=0.12)
    print(WF_val.info())

fig.plot_line(line=trains[best_index].equity(),
             name='best train',
             color='red')
fig.plot_line(line=WF_val.equity(),
             name='best train validation',
             color='green',
             _row=2)
fig.log_y(1, 1)
fig.log_y(2, 1)
fig.show()
