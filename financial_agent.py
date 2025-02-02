from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

web_search_agent = Agent(
    name="web_search_agent",
    tools=[DuckDuckGo],
    role = "search the web for information",
    model = Groq(id = "llama-3.3-70b-specdec"),
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
    )

financial_agent = Agent(
    name="finance AI agent",
    role = "search the web for financial information",
    model = Groq(id = "llama-3.3-70b-specdec"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                           company_news=True)],
    instructions=["use tables and graphs to display information"],
    show_tool_calls=True,
    markdown=True
    )

multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    instructions=["Always include sources","Use tables and graphs to display information"],
    show_tool_calls=True,
    markdown=True

)

multi_ai_agent.print_response("Summarize analyst recommendations for apple stock",stream=True)