# LLMProject

# imports
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import pandas as pd
import json
from agent import query_agent, create_agent

# install streamlit
pip install streamlit

# run the project
streamlit run interface.py
