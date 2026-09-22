import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

BRAPI_BASE_URL = "https://brapi.dev/api/v2"


@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price and basic market information for a Brazilian stock."""
    
    url = f"{BRAPI_BASE_URL}/stocks/quote"

    headers = {}

    token = os.getenv("BRAPI_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        url,
        params={"symbols": ticker.upper()},
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        raise ValueError(f"No market data found for ticker: {ticker}")

    return data["results"][0]


@tool
def get_historical_prices(
    ticker: str,
    range_: str = "3mo",
    interval: str = "1d",
) -> dict:
    """Get historical stock price data for up to 3 months."""

    url = f"{BRAPI_BASE_URL}/stocks/historical"

    headers = {}

    token = os.getenv("BRAPI_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        url,
        params={
            "symbols": ticker.upper(),
            "range": range_,
            "interval": interval,
        },
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        raise ValueError(f"No historical data found for ticker: {ticker}")

    return data["results"][0]