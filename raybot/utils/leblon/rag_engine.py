import glob
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .settings_leblon import PASTA_DOCUMENTACAO, PASTA_DB_RAG

def inicializar_rag():
    """Carrega ou cria o banco vetorial a partir dos XLSX."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(PASTA_DB_RAG) and os.listdir(PASTA_DB_RAG):
        print("📚 Carregando RAG existente do disco...")
        return Chroma(persist_directory=PASTA_DB_RAG, embedding_function=embeddings)

    print("⚙️ Criando novo índice RAG a partir de XLSX...")
    arquivos = glob.glob(os.path.join(PASTA_DOCUMENTACAO, "*.xlsx"))
    
    if not arquivos:
        print("⚠️ Nenhum XLSX encontrado para documentação.")
        return None

    documentos_texto = []
    for arq in arquivos:
        try:
            xls = pd.ExcelFile(arq)
            for aba in xls.sheet_names:
                df = pd.read_excel(arq, sheet_name=aba)
                df = df.astype(str) 
                texto_aba = f"### ARQUIVO: {os.path.basename(arq)}\n### ABA: {aba}\n\n"
                texto_aba += df.to_markdown(index=False)
                documentos_texto.append(texto_aba)
        except Exception as e:
            print(f"❌ Erro ao processar {arq}: {e}")

    if not documentos_texto:
        return None

    docs = [Document(page_content=t) for t in documentos_texto]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_divididos = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        docs_divididos,
        embedding=embeddings,
        persist_directory=PASTA_DB_RAG
    )
    return vectordb

def recuperar_contexto(vectordb, pergunta, k=5):
    if not vectordb:
        return ""
    try:
        resultados = vectordb.similarity_search(pergunta, k=k)
        if not resultados:
            return ""
        return "\n\n".join([r.page_content for r in resultados])
    except Exception as e:
        print(f"Erro no RAG: {e}")
        return ""