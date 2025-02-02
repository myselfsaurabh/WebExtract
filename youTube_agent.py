from phi.agent import Agent
from phi.tools.youtube_tools import YouTubeTools

from phi.model.groq import Groq
import openai
import os
from dotenv import load_dotenv
# openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

agent = Agent(
    tools=[YouTubeTools()],
    show_tool_calls=True,
    model = Groq(id = "llama-3.3-70b-specdec"),
    description="You are a YouTube agent. Obtain the captions of a YouTube video and answer questions.",
)

agent.print_response("Summarize this video https://www.youtube.com/watch?v=Gbl3pp49xs4", markdown=True)