"""## ReACT Agent

Build a simple project: Search Assistant Agent
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from llama_index.llms.mistralai import MistralAI
def search(query:str) -> str:
  """
  Args:
      query: user prompt
  return:
  context (str): search results to the user query
  """
  # def search(query:str)
  req = DDGS()
  response = req.text(query,max_results=4)
  context = ""
  for result in response:
    context += result['body']
  return context

search_tool = FunctionTool.from_defaults(fn=search)
mistral_api = "aPS8Usmu3VjfVoJ0D7ehIdr0HmFYDOu7"

llm = MistralAI(model="mistral-large-latest",api_key=mistral_api)
agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True,allow_parallel_tool_calls=True)

response = agent.chat("tell me about Unleash AI. Its youtube handle is @unleashAI23. Also do analysis and say should people follow him?")

print(response)