# 🎯 What it does:
# • Automatically detects the current date
# • Fetches historical stock prices for the current month
# • Provides detailed statistical analysis and performance metrics
# . Provides How many tokens you have used
# • All with a single prompt!


from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from dotenv import load_dotenv
import os
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
import yfinance as yf
import pandas as pd
from datetime import datetime, date
import anthropic
# Load environment variables
load_dotenv('.env')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

client = anthropic.Anthropic()
code_spec = CodeInterpreterToolSpec()

tools = code_spec.to_tool_list()

#Specify the AGENT
tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer

llm_claude = Anthropic(model="claude-3-5-sonnet-20241022")

from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    tools,
    llm=llm_claude,
    verbose=True,
    allow_parallel_tool_calls=False,
    chat_history=[]
)



stock = 'TESLA'

prompt = f"""
Write a python code to :
- Detect which date is today
- Based on this date, fetch historical prices of {stock} from the beginning of the month until today.
- Analyze the last month prices
"""

resp = agent.chat(prompt)
agent.chat_history.append(resp)

# Get today's date
today = date.today()
print(f"Today's date is: {today}")

# Get the first day of the current month
start_date = today.replace(day=1)
print(f"Start date is: {start_date}")

# Fetch Tesla stock data
tesla = yf.download('TSLA', start=start_date, end=today)
print("\nTesla stock data shape:", tesla.shape)

# Basic analysis of the stock prices
print("\nPrice Summary Statistics:")
print(tesla['Close'].describe())

# Calculate daily returns
tesla['Daily_Return'] = tesla['Close'].pct_change()

# Print key metrics
print("\nKey Metrics:")
print(f"Starting Price: ${float(tesla['Close'].iloc[0]):.2f}")
print(f"Ending Price: ${float(tesla['Close'].iloc[-1]):.2f}")
print(f"Highest Price: ${float(tesla['High'].max()):.2f}")
print(f"Lowest Price: ${float(tesla['Low'].min()):.2f}")
print(f"Average Daily Return: {float(tesla['Daily_Return'].mean()*100):.2f}%")
print(f"Daily Return Volatility: {float(tesla['Daily_Return'].std()*100):.2f}%")

# Calculate total return for the period
total_return = float(((tesla['Close'].iloc[-1] - tesla['Close'].iloc[0]) / tesla['Close'].iloc[0]) * 100)
print(f"\nTotal Return for the period: {total_return:.2f}%")


analysis_text = f"""
stock Analysis for {stock}:
today's date: {today}
period: {start_date} to {today}
starting Price: ${float(tesla['Close'].iloc[0]):.2f}
ending Price: ${float(tesla['Close'].iloc[-1]):.2f}
total Return: {total_return:.2f}%
"""
response2 = client.messages.count_tokens(
   model="claude-3-5-sonnet-20241022",
   system="You are a financial analyst",
   messages=[{
       "role": "user",
       "content": analysis_text
   }],
)

print("\nToken count for analysis:", response2.json())
