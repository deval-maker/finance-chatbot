import os
from typing import Type

from forex_python.converter import CurrencyRates
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


def get_exchange_rate(from_currency, to_currency):
    """Method to get exchange rate between two currencies"""
    rate = CurrencyRates().get_rate(from_currency, to_currency)
    return {"rate": rate}


class CurrencyExchangeRate(BaseModel):
    """Inputs for get_exchange_rate"""

    from_currency: str = Field(description="Symbol of the currency")
    to_currency: str = Field(description="Symbol of the currency")


class ExchangeRateTool(BaseTool):
    name = "get_exchange_rate"
    description = """
        Useful when you want to get exchange rate of two currencies.
        You should enter the currency symbol recognized by the forex-python.
        For both from and to currency inputs.
        output will be the rate of exchange between two currencies.
        """
    args_schema: Type[BaseModel] = CurrencyExchangeRate

    def _run(self, from_currency: str, to_currency: str):
        response = get_exchange_rate(from_currency, to_currency)
        return response

    def _arun(self, from_currency: str, to_currency: str):
        raise NotImplementedError("get_stock_performance does not support async")


openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, openai_api_key=openai_api_key)
tools = [ExchangeRateTool()]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

print(agent.run("How much is 10000 USD in INR? What is the exchange rate between GBP and INR?"))

# Output:
# The current exchange rate between USD and INR is 1 USD = 83.27 INR.
# Therefore, 10000 USD is approximately equal to 832,652.30 INR.
# The current exchange rate between GBP and INR is 1 GBP = 101.94 INR.
