from app.tools.market import (
    get_stock_price,
    get_historical_prices,
)

print(get_stock_price.name)
print(get_historical_prices.name)

print(get_stock_price.args)
print(get_historical_prices.args)

result = get_stock_price.invoke({"ticker": "SAPR4"})
print(result)

result = get_historical_prices.invoke({"ticker": "SAPR4"})
print(result)