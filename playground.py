from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
import phi.api
import phi 
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
# openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="web_search_agent",
    tools=[DuckDuckGo],
    role = "search the web for information",
    model = Groq(id = "llama3-70b-8192"),
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
    )

financial_agent = Agent(
    name="finance AI agent",
    role = "search the web for financial information",
    model = Groq(id = "llama3-70b-8192"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                           company_news=True)],
    instructions=["use tables and graphs to display information"],
    show_tool_calls=True,
    markdown=True
    )

app= Playground(agents=[financial_agent,web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)