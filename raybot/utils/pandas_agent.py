import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODELO_LLM = "gpt-4o-mini"

def carregar_multiplas_tabelas(nome_tabelas: list[str]):
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL não foi encontrada no .env")

    engine = create_engine(DATABASE_URL)

    dfs = []
    for nome in nome_tabelas:
        try:
            df = pd.read_sql(f"SELECT * FROM {nome};", engine)
            df.__name__ = nome  
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao carregar tabela {nome}: {e}")

    return dfs  


def criar_agente_multiplas_tabelas(lista_dfs):
    llm = ChatOpenAI(
        model=MODELO_LLM,
        temperature=0,
    )

    agente = create_pandas_dataframe_agent(
        llm,
        lista_dfs, 
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=50,
        agent_executor_kwargs={"handle_parsing_errors": True, "timeout": 40},
    )

    return agente