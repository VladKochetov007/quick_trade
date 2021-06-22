# A guide to using quick_trade, from drawing up a strategy to selecting the best parameters and launching on the exchange:

## How can I create my own strategy?

- To create a strategy, you will need to generate values for:
    - `Trader.returns` - Directly the results of the strategy. 
    - `Trader._stop_losses` - Stop-loss list.
    - `Trader._take_profits` - Take-profit list.
    - `Trader._credit_leverages` - Not really leverages, but rather the values by which need to multiply the trade amount.
    - `Trader._open_lot_prices` - List of prices at which deals were opened.
  
?> Lengths of all these variables are equal to the length of the dataframe with the prices of the traded currency.

If your strategy does not provide for the use of anything from this list, quick_trade provides methods for setting 
default values (as if the trader and tester would not use them).

If your strategy does not generate stop loss or take profit, there is the [`Trader.set_open_stop_and_take`](https://vladkochetov007.github.io/quick_trade/#/docs/quick_trade/trading_sys?id=set_open_stop_and_take) method.
It automatically generates trade entry prices and, if necessary, stop-losses and take-profits.

If you need to generate stop-loss and take-profit:
```python
self.set_open_stop_and_take()
```
If your strategy PROVIDES a stop loss, then you need to set the value `set_stop` equal to `False`:
```python
self.set_open_stop_and_take(set_stop=False)  # The method does not change or set stop-loss.
```
Likewise with take profit and `set_take`.


If you want to set take-profit or stop-loss, you can specify the `take_profit` and `stop_loss` arguments in points.

?> tip: pips (aka point) = 1/10_000 of price

If you want to enter a trade not for the entire deposit, but for a part or more (leverage), you can use the 
`set_credit_leverages` method. It places the same `self._credit_leverages` for all candles.

```python
self.set_credit_leverages(credit_lev=0.25)  # 1/4 of deposit for trade
```
```python
self.set_credit_leverages(credit_lev=5)  # 5X credit leverage
```

## How can I test it?

## What if I combine the two strategies?

## More strategies?

## I made a strategy. How can I test it?

## My strategy is good! How can I start it?
