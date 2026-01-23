import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os
import glob
import sys
import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from func_timeout import func_timeout, FunctionTimedOut
from xlsx_csv import converter_todos_xlsx
import datetime

# --- IMPORTAÇÃO DAS TOOLS ---
import tools 

# Configuração de cores para logs
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except Exception:
    class _F:
        RESET = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
    Fore = _F()
    Style = _F()

load_dotenv()

# --- CONFIGURAÇÕES ---
PASTA_CSV = "base_leblon"
MODELO_LLM = "gpt-4o-mini" 

# ===============================
# Utilitários de log
# ===============================
def log_info(msg: str):
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {msg}")

def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")

def log_error(msg: str):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

# ====================================================
# RAG: carregar documentação a partir de arquivos XLSX
# ====================================================
def carregar_documentacao_xlsx(padrao_arquivos="documentacao_leblon/*.xlsx"):
    arquivos = glob.glob(padrao_arquivos)
    if not arquivos:
        log_warn("Nenhum XLSX encontrado em 'documentacao_leblon/'. RAG ficará desativado.")
        return None

    documentos_texto = []
    for arq in arquivos:
        # Ignora arquivos de manutenção manual se estiverem na pasta de doc
        if "indmantmanual" in os.path.basename(arq).lower():
            continue
            
        try:
            xls = pd.ExcelFile(arq)
            log_info(f"XLSX carregado: {arq}")
            for aba in xls.sheet_names:
                try:
                    df = pd.read_excel(arq, sheet_name=aba)
                    # Tratamento básico para transformar tabela em texto
                    df_lower = df.copy()
                    df_lower.columns = df_lower.columns.str.lower()
                    for col in df_lower.columns:
                        if df_lower[col].dtype == "object":
                            df_lower[col] = df_lower[col].astype(str).str.lower()
                    
                    texto_aba = f"### ARQUIVO: {arq}\n### ABA: {aba}\n\n" + df.to_markdown(index=False)
                    documentos_texto.append(texto_aba)
                except Exception as e:
                    continue
        except Exception as e:
            log_error(f"Erro ao carregar XLSX {arq}: {e}")

    if not documentos_texto:
        return None

    from langchain_core.documents import Document
    documentos = [Document(page_content=txt) for txt in documentos_texto]
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_divididos = splitter.split_documents(documentos)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma.from_documents(docs_divididos, embedding=embeddings, persist_directory="db_rag")
        log_info(f"📚 RAG carregado com {len(docs_divididos)} chunks.")
        return vectordb
    except Exception as e:
        log_error(f"Erro ao criar vectordb: {e}")
        return None

def recuperar_contexto_rag(vectordb, pergunta: str, k: int = 5) -> str:
    if vectordb is None: return "Nenhuma documentação carregada."
    try:
        resultados = vectordb.similarity_search(pergunta, k=k)
        if not resultados: return "Nenhuma informação relevante encontrada na documentação."
        return "\n\n".join([r.page_content for r in resultados])
    except Exception as e: return "Erro ao recuperar contexto documental."

# ====================================================
# Carga de Dados
# ====================================================
def carregar_dados(arquivos_csv):
    if not arquivos_csv:
        log_error("Nenhum arquivo CSV encontrado.")
        sys.exit(1)
    
    dfs_carregados = []
    for arquivo in arquivos_csv:
        try:
            df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8')
        except:
            try: df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin1')
            except: continue
        
        # Padronização de colunas
        df.columns = df.columns.str.lower()
        df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
        df["__origem"] = os.path.basename(arquivo)
        dfs_carregados.append(df)
        
    return dfs_carregados

# ====================================================
# MAIN
# ====================================================
def main():
    # 1. CARREGAR DADOS
    converter_todos_xlsx(PASTA_CSV)
    arquivos = glob.glob(os.path.join(PASTA_CSV, "*.csv"))
    lista_dfs = carregar_dados(arquivos)

    # --- PONTO CRUCIAL: Passar dados para o módulo tools ---
    tools.set_dados_globais(lista_dfs)
    # -------------------------------------------------------

    print("-" * 60)
    for i, df in enumerate(lista_dfs):
        log_info(f"📊 Tabela {i+1} ({df['__origem'].iloc[0]}): {len(df)} linhas")
    print("-" * 60)

    # 2. CARREGAR RAG
    vectordb = carregar_documentacao_xlsx("documentacao_leblon/*.xlsx")

    # 3. CONFIGURAR LLM
    llm = ChatOpenAI(model=MODELO_LLM, temperature=0)

    # 4. DEFINIR LISTA DE TOOLS (Vindo do módulo tools)
    lista_tools_kpi = [
        tools.calcular_icmq, tools.calcular_idf, tools.calcular_imp, 
        tools.calcular_oemcp, tools.calcular_oempp, tools.calcular_preventivas_liquidadas, 
        tools.calcular_km_falhas, tools.calcular_qetg, tools.calcular_qett, 
        tools.calcular_cdtdm, tools.calcular_caiefo, tools.calcular_qva, 
        tools.calcular_qvv, tools.calcular_tic, tools.calcular_to, tools.calcular_topp
    ]

    # 5. AGENTE UNIFICADO (A Chave do Sucesso do TESTE.PY)
    # O create_pandas_dataframe_agent é inteligente. Se você passar extra_tools, 
    # ele decide sozinho se usa Pandas ou se chama a Tool.
    agente = create_pandas_dataframe_agent(
        llm,
        lista_dfs,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=50,
        extra_tools=lista_tools_kpi, # Tools injetadas aqui
        agent_type="openai-tools",
        agent_executor_kwargs={"handle_parsing_errors": True, "timeout": 60}
    )

    # 6. LOOP DE INTERAÇÃO
    print("\n" + Fore.GREEN + "🤖 Sistema Iniciado (Modo Unificado). Olá! Como posso te ajudar?" + Style.RESET_ALL)
    
    while True:
        try:
            pergunta = (input("\nDigite sua pergunta (ou 'sair'): ")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if pergunta.lower() in ["sair", "exit", "tchau"]:
            break
                
        # Recupera contexto do RAG
        contexto_rag = recuperar_contexto_rag(vectordb, pergunta)

        # Monta info das colunas para o prompt (Igual ao TESTE.PY)
        info_colunas = []
        for i, df_ in enumerate(lista_dfs):
            nome_arq = df_['__origem'].iloc[0] if '__origem' in df_.columns else f"Tabela {i+1}"
            info_colunas.append(f"📍 TABELA {i+1} (Nome: {nome_arq}) - {len(df_)} linhas:\n   Colunas: {', '.join(df_.columns)}")
        texto_dados_disponiveis = "\n\n".join(info_colunas)
        
        hoje = datetime.datetime.now().strftime("%d/%m/%Y")

        # ==============================================================================
        # PROMPT UNIFICADO (Baseado no TESTE.PY)
        # ==============================================================================
        prompt_final = f"""
Você é um analista de dados sênior especializado em análise tabular e indicadores da Leblon.
Você tem acesso a {len(lista_dfs)} DataFrames carregados: df1, df2, etc.
Você TAMBÉM tem acesso a ferramentas (tools) específicas para cálculo de KPIs oficiais.

DATA DE HOJE: {hoje}

INSTRUÇÃO MESTRA:
1. Se a pergunta for sobre um INDICADOR ESPECÍFICO listado abaixo, USE A TOOL correspondente.
   Siglas: ICMQ, IDF, IMP, OEMCP, OEMPP, KmFalhas, QETG, QETT, CDTDM, CAIEFO, QVA, QVV, TIC, TO, TOPP, Preventivas Liquidadas.
   - Não tente calcular esses KPIs via pandas manualmente. Use a tool.
   - Se houver datas na pergunta (ex: "janeiro 2024"), converta para formato 'YYYY-MM-DD' e passe para a tool.

2. Para TODAS AS OUTRAS PERGUNTAS (Análise geral, contagens, somas, listagens, rankings):
   - Utilize Python/Pandas diretamente nos DataFrames.
   - Analise os nomes das colunas abaixo para entender onde estão os dados.
   - CTM = Custos/Peças
   - MANT001 = Ocorrências/Trocas/Quebras
   - MANT002 = Ordem de Serviço/Manutenção/Preventiva/Corretiva
   - MANT004 = Saídas
   - IND003 = Quilometragem (KmRodado)

OBSERVAÇÃO IMPORTANTE - TRATAMENTO DE IDENTIFICADORES:
    - O usuário pode perguntar números (ex: ônibus 32004) sem usar aspas. Para garantir que o filtro funcione, **SEMPRE converta a coluna alvo para string (.astype(str))** antes de comparar.

TABELAS DISPONÍVEIS (PANDAS):
{texto_dados_disponiveis}

CONTEXTO DOCUMENTAL (RAG):
{contexto_rag}

PERGUNTA DO USUÁRIO:
{pergunta}

ESTILO DA RESPOSTA:
Direta, objetiva, em Português (Brasil).
Se for dinheiro, use R$. Se for número, use formato brasileiro.
"""

        try:
            print(f"{Fore.CYAN}🤔 Processando...{Style.RESET_ALL}")
            resposta = func_timeout(60, agente.invoke, args=({"input": prompt_final},))
            texto = resposta.get("output", str(resposta))
            print("\n" + Fore.BLUE + "🤖 Resposta:" + Style.RESET_ALL)
            print(texto)
        except FunctionTimedOut:
            print(f"\n{Fore.RED}Tempo limite excedido.{Style.RESET_ALL}")
        except Exception as e:
            print(f"Erro: {e}")
            traceback.print_exc()

        print("-" * 60)

if __name__ == "__main__":
    main()