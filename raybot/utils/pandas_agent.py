import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODELO_LLM = "gpt-4o-mini"

def carregar_dataframe():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL não foi encontrada no .env")

    engine = create_engine(DATABASE_URL)

    query = "SELECT * FROM meta_insights_geral;"

    df = pd.read_sql(query, engine)

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