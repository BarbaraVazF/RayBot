import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_CSV = os.path.join(BASE_DIR, "dados_carros.csv")

MODELO_LLM = "gpt-4o-mini"


def carregar_dataframe():
    try:
        df = pd.read_csv(ARQUIVO_CSV, encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        df = pd.read_csv(ARQUIVO_CSV, encoding="latin1")
    return df


def criar_agente(df):
    llm = ChatOpenAI(
        model=MODELO_LLM,
        temperature=0,
    )

    agente = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=50,
        agent_executor_kwargs={"handle_parsing_errors": True, "timeout": 40}
    )

    return agente