import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

MODELO_LLM = "gpt-4o-mini"
PASTA_BASES = "base_leblon"

def carregar_multiplas_tabelas():
    if not os.path.exists(PASTA_BASES):
        raise FileNotFoundError(f"Pasta não encontrada: {PASTA_BASES}")

    dfs = []

    for arquivo in os.listdir(PASTA_BASES):
        caminho = os.path.join(PASTA_BASES, arquivo)
        nome_base = os.path.splitext(arquivo)[0]

        try:
            if arquivo.lower().endswith(".csv"):
                df = pd.read_csv(caminho)

            elif arquivo.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(caminho)

            else:
                continue

            df.__name__ = nome_base 
            dfs.append(df)

            print(f"✔ Base carregada: {arquivo} ({df.shape[0]} linhas)")

        except Exception as e:
            print(f"❌ Erro ao carregar {arquivo}: {e}")

    if not dfs:
        raise ValueError("Nenhuma base válida encontrada")

    return dfs



def criar_agente_multiplas_tabelas(lista_dfs):
    llm = ChatOpenAI(
        model=MODELO_LLM,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    agente = create_pandas_dataframe_agent(
        llm,
        lista_dfs,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=50,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "timeout": 40
        },
    )

    return agente

def main():
    dfs = carregar_multiplas_tabelas()
    agente = criar_agente_multiplas_tabelas(dfs)

    resposta = agente.invoke("sua pergunta")
    print(resposta["output"])


if __name__ == "__main__":
    main()