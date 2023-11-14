from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import yfinance as yf
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates

def get_exchange_rate(from_currency, to_currency):
    """Method to get exchange rate between two currencies"""
    rate = CurrencyRates().get_rate(from_currency, to_currency)    
    return {"rate": rate}


def get_current_stock_price(ticker):
    """Method to get current stock price"""

    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    return {"price": recent.iloc[0]["Close"], "currency": ticker_data.fast_info["currency"]}


def get_stock_performance(ticker, days):
    """Method to get stock price change in percentage"""

    past_date = datetime.today() - timedelta(days=days)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(start=past_date)
    old_price = history.iloc[0]["Close"]
    current_price = history.iloc[-1]["Close"]
    return {"percent_change": ((current_price - old_price) / old_price) * 100}


class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""

    ticker: str = Field(description="Ticker symbol of the stock")


class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = """
        Useful when you want to get current stock price.
        You should enter the stock ticker symbol recognized by the yahoo finance
        """
    args_schema: Type[BaseModel] = CurrentStockPriceInput

    def _run(self, ticker: str):
        price_response = get_current_stock_price(ticker)
        return price_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_price does not support async")


class StockPercentChangeInput(BaseModel):
    """Inputs for get_stock_performance"""

    ticker: str = Field(description="Ticker symbol of the stock")
    days: int = Field(description="Timedelta days to get past date from current date")

class CurrencyExchangeRate(BaseModel):
    """Inputs for get_exchange_rate"""

    from_currency: str = Field(description="Symbol of the currency")
    to_currency: str = Field(description="Symbol of the currency")

class ExchangeRateTool(BaseTool):
    name = "get_exchange_rate"
    description = """
        Useful when you want to get exchange rate of two currencies.
        You should enter the currency symbol recognized by the forex-python. For both from and to currency inputs.
        output will be the rate of exchange between two currencies.
        """
    args_schema: Type[BaseModel] = CurrencyExchangeRate

    def _run(self, from_currency: str, to_currency: str):
        response = get_exchange_rate(from_currency, to_currency)
        return response

    def _arun(self, from_currency: str, to_currency: str):
        raise NotImplementedError("get_stock_performance does not support async")
    

class StockPerformanceTool(BaseTool):
    name = "get_stock_performance"
    description = """
        Useful when you want to check performance of the stock.
        You should enter the stock ticker symbol recognized by the yahoo finance.
        You should enter days as number of days from today from which performance needs to be check.
        output will be the change in the stock price represented as a percentage.
        """
    args_schema: Type[BaseModel] = StockPercentChangeInput

    def _run(self, ticker: str, days: int):
        response = get_stock_performance(ticker, days)
        return response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_performance does not support async")
    
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

tools = [CurrentStockPriceTool(), StockPerformanceTool(),ExchangeRateTool() ]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


print(agent.run("Give me recent stock prices of Google and Meta? If you dont know the ticker symbol, you can use the company name."))
print(agent.run(
    "What is the current price of Microsoft stock? How it has performed over past 3 months?"
))

print(agent.run(
    "In the past 3 months, which stock between Microsoft and Google has performed the best?"
))

print(agent.run("How much is 10000 USD in INR? What is the exchange rate between GBP and INR?"))
