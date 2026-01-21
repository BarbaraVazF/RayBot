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

# Importa as ferramentas do outro arquivo
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

def safe_print_df_info(df: pd.DataFrame, label: str = ""):
    try:
        print(f"{Fore.MAGENTA}{label}{Style.RESET_ALL} Linhas: {len(df)} | Colunas: {', '.join(df.columns)}")
    except Exception as e:
        log_error(f"Erro ao imprimir info do DataFrame: {e}")

# ====================================================
# RAG: carregar documentação a partir de arquivos XLSX
# ====================================================
def carregar_documentacao_xlsx(padrao_arquivos="documentacao_leblon/*.xlsx"):
    """
    Carrega arquivos XLSX e converte todas as abas em blocos textuais,
    permitindo criar um banco vetorial para consulta semântica (RAG).
    """
    arquivos = glob.glob(padrao_arquivos)
    if not arquivos:
        log_warn("Nenhum XLSX encontrado em 'documentacao_leblon/'. RAG ficará desativado.")
        return None

    documentos_texto = []

    for arq in arquivos:
        try:
            xls = pd.ExcelFile(arq)
            log_info(f"XLSX carregado: {arq}")

            for aba in xls.sheet_names:
                try:
                    df = pd.read_excel(arq, sheet_name=aba)
                except Exception as e:
                    log_error(f"Erro ao ler aba '{aba}' no arquivo {arq}: {e}")
                    continue

                # Convertendo DataFrame em texto legível para o LLM
                df_lower = df.copy()
                df_lower.columns = df_lower.columns.str.lower()

                for col in df_lower.columns:
                    if df_lower[col].dtype == "object":
                        df_lower[col] = df_lower[col].astype(str).str.lower()

                texto_aba = f"### ARQUIVO: {arq}\n### ABA: {aba}\n\n"
                texto_aba += df.to_markdown(index=False)

                documentos_texto.append(texto_aba)

        except Exception as e:
            log_error(f"Erro ao carregar XLSX {arq}: {e}")
            traceback.print_exc()

    if not documentos_texto:
        log_warn("Nenhuma aba válida encontrada nos XLSX. RAG ficará desativado.")
        return None

    # Criar objetos Document para o splitter
    from langchain_core.documents import Document
    documentos = [Document(page_content=txt) for txt in documentos_texto]

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_divididos = splitter.split_documents(documentos)
    except Exception as e:
        log_error(f"Erro ao dividir textos dos XLSX: {e}")
        return None

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma.from_documents(
            docs_divididos,
            embedding=embeddings,
            persist_directory="db_rag"
        )
        log_info(f"📚 RAG carregado com {len(docs_divididos)} chunks dos arquivos XLSX.")
        return vectordb

    except Exception as e:
        log_error(f"Erro ao criar vectordb a partir dos XLSX: {e}")
        return None


def recuperar_contexto_rag(vectordb, pergunta: str, k: int = 5) -> str:
    if vectordb is None:
        return "Nenhuma documentação carregada."
    try:
        resultados = vectordb.similarity_search(pergunta, k=k)
        if not resultados:
            return "Nenhuma informação relevante encontrada na documentação."
        return "\n\n".join([r.page_content for r in resultados])
    except Exception as e:
        log_error(f"Erro ao recuperar contexto do RAG: {e}")
        return "Erro ao recuperar contexto documental."

# ====================================================
# Função de Carregamento Simples
# ====================================================
def carregar_dados(arquivos_csv):
    """
    Carrega arquivos CSV individualmente e retorna uma lista de DataFrames.
    """
    if not arquivos_csv:
        log_error("Nenhum arquivo CSV encontrado na pasta.")
        sys.exit(1)

    log_info(f"📁 CSVs detectados: {', '.join(arquivos_csv)}")

    dfs_carregados = []
    for arquivo in arquivos_csv:
        try:
            # Tenta detectar separador automaticamente
            df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin1')
            except Exception as e:
                log_error(f"Erro ao ler {arquivo}: {e}")
                continue

        df.columns = df.columns.str.lower()
        df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
        
        df["__origem"] = os.path.basename(arquivo)
        dfs_carregados.append(df)

    n = len(dfs_carregados)
    if n == 0:
        log_error("Nenhum DataFrame carregado com sucesso.")
        sys.exit(1)

    log_info("🔎 Tabelas carregadas individualmente.")
    return dfs_carregados

def realizar_join(
    lista_dfs: list,
    nome_tabela_esquerda: str,
    nome_tabela_direita: str,
    chave_esquerda: str,
    chave_direita: str,
    tipo_join: str = "left"
):
    # localizar tabelas pelo nome do arquivo original
    df_esq = next((df for df in lista_dfs if df["__origem"].iloc[0] == nome_tabela_esquerda), None)
    df_dir = next((df for df in lista_dfs if df["__origem"].iloc[0] == nome_tabela_direita), None)

    if df_esq is None:
        raise ValueError(f"A tabela '{nome_tabela_esquerda}' não foi encontrada.")

    if df_dir is None:
        raise ValueError(f"A tabela '{nome_tabela_direita}' não foi encontrada.")

    if chave_esquerda not in df_esq.columns:
        raise ValueError(f"A coluna '{chave_esquerda}' não existe em {nome_tabela_esquerda}.")

    if chave_direita not in df_dir.columns:
        raise ValueError(f"A coluna '{chave_direita}' não existe em {nome_tabela_direita}.")
    
    df_esq[chave_esquerda] = df_esq[chave_esquerda].astype(str).str.replace(r'\.0$', '', regex=True)
    df_dir[chave_direita] = df_dir[chave_direita].astype(str).str.replace(r'\.0$', '', regex=True)
    # ---------------------

    df_join = df_esq.merge(
        df_dir,
        left_on=chave_esquerda,
        right_on=chave_direita,
        how=tipo_join
    )

    return df_join


# ====================================================
# Função principal - analisar_dados_csv
# ====================================================
def main():
    # 1. CARREGAR DADOS
    converter_todos_xlsx(PASTA_CSV)
    arquivos = glob.glob(os.path.join(PASTA_CSV, "*.csv"))
    lista_dfs = carregar_dados(arquivos)


    # Expor dados às tools (IMPORTANTE: Usa a função setter do tools.py)
    tools.set_dados_globais(lista_dfs)

    print("-" * 60)
    for i, df in enumerate(lista_dfs):
        log_info(f"📊 Tabela {i+1} ({df['__origem'].iloc[0]}): {len(df)} linhas | Colunas: {', '.join(df.columns)}")
    print("-" * 60)

    # 2. CARREGAR RAG
    vectordb = carregar_documentacao_xlsx("documentacao_leblon/*.xlsx")

    # 3. CONFIGURAR LLM
    llm = ChatOpenAI(
        model=MODELO_LLM,
        temperature=0,
    )

    # 4. CRIAR AGENTE
    # Lista de tools (Vem do import tools)
    tools_kpi = [
        tools.calcular_icmq, 
        tools.calcular_idf, 
        tools.calcular_imp, 
        tools.calcular_oemcp, 
        tools.calcular_oempp, 
        tools.calcular_preventivas_liquidadas, 
        tools.calcular_km_falhas, 
        tools.calcular_qetg, 
        tools.calcular_qett, 
        tools.calcular_cdtdm, 
        tools.calcular_caiefo, 
        tools.calcular_qva, 
        tools.calcular_qvv, 
        tools.calcular_tic, 
        tools.calcular_to, 
        tools.calcular_topp
    ]

    agente = create_pandas_dataframe_agent(
        llm,
        lista_dfs,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=50,
        extra_tools=tools_kpi,
        
        agent_type="openai-tools",
        agent_executor_kwargs={"handle_parsing_errors": True, "timeout": 50}
    )

    historico = []

    # 5. LOOP DE PERGUNTAS
    print()
    print("Olá. Como posso te ajudar?")

    while True:
        try:
            pergunta = (input("\nDigite sua pergunta (ou 'sair'): ")).lower()
        except (KeyboardInterrupt, EOFError):
            break

        if pergunta == "sair":
            break

        historico.append(pergunta)

        # Prepara info das colunas para o prompt
        info_colunas = []
        for i, df_ in enumerate(lista_dfs):
            nome_arq = df_['__origem'].iloc[0] if '__origem' in df_.columns else f"Tabela {i+1}"
            info_colunas.append(f"📍 TABELA {i+1} (Nome: {nome_arq}) - {len(df_)} linhas:\n   Colunas: {', '.join(df_.columns)}")
        texto_dados_disponiveis = "\n\n".join(info_colunas)

        contexto_documentacao = recuperar_contexto_rag(vectordb, pergunta)

        hoje = datetime.datetime.now()
        data_atual_str = hoje.strftime("%d/%m/%Y")
        dia_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"][hoje.weekday()]

        # ===============================
        # PROMPT 
        # ===============================
        prompt = f"""
Você é um analista de dados sênior especializado em análise tabular.
Você tem acesso a {len(lista_dfs)} DataFrames carregados separadamente: df1, df2, etc.
Esses DataFrames não são unidos automaticamente.
VOCÊ SÓ DEVE USAR O DATAFRAME "INDMANTMANUAL" QUANDO A PERGUNTA INCLUIR EXATAMENTE O NOME DE UM DESSES INDICADORES: CAIEFO, CDTDM, QVA, QVV, TIC, TO, TOPP. 

**CONTEXTO TEMPORAL** HOJE É: {data_atual_str} ({dia_semana}).
Use esta data como referência absoluta para calcular termos relativos:
- "Ano passado": Calcule o ano de {data_atual_str} menos 1.
- "Mês passado": Calcule o mês anterior à data de {data_atual_str}.
- "Este ano" ou "Atualmente": Refere-se ao ano de {data_atual_str}.
- "Últimos 12 meses": De {data_atual_str} voltando 1 ano.

------------------------------------------------------------------------------------------------------------------------------

### 🚨 PROTOCOLO DE DECISÃO (LEIA COM ATENÇÃO) 🚨

Você deve classificar a pergunta do usuário em um dos dois caminhos abaixo e agir estritamente conforme a regra.

**CAMINHO 1: ANÁLISE GERAL (USO DE CÓDIGO PANDAS)**
GATILHO: A pergunta **NÃO** contém nenhuma das seguintes siglas: [ICMQ, IDF, IMP, OEMCP, OEMPP, KMFalhas, QETG, QETT, CDTDM, CAIEFO, QVA, QVV, TIC, TO, TOPP, Preventivas Liquidadas]
Exemplos: "Qual o custo total?", "Quantos ônibus temos?", "Qual a média de km?", "Mostre as 5 piores falhas".
**NÃO USE NENHUMA TOOL DEFINIDA.** É PROIBIDO chamar `calcular_...` neste caminho.
SIGA OS SEGUINTES PASSOS:
1. USE O RAG OBRIGATORIAMENTE: O RAG contém definições oficiais e estruturais, organizadas em três abas:
    - Aba 1 - Descrição das Planilhas: explica o propósito de cada tabela.
    - Aba 2 - Descrição das Colunas: detalha o significado e uso de cada coluna.
2. SIGA AS REGRAS DE INTERPRETAÇÃO ABAIXO:
    a. Você deve interpretar a pergunta pelo seu SIGNIFICADO e INTENÇÃO.
        - Considere singular e plural como equivalentes.
        - Considere variações verbais, abreviações e linguagem informal.
        - Reconheça sinônimos, termos equivalentes e variações semânticas.
        - Ignore erros leves de digitação ou variações comuns de escrita.
        - Ignore termos desnecessários na solicitação. Foque nas palavras chave.
    b. Caso a pergunta não seja compreendida de imediato, reformule-a internamente usando sinônimos e tente interpretá-la novamente.
    c. Caso a informação solicitada não exista, responda com as mensagens padronizadas.
3. Mapeie o segmento da pergunta (palavras-chave conceituais).
4. A partir do segmento identificado, encontre qual tabela mais tem similaridade com ele a partir da descrição e das suas colunas.
5. Selecione a(s) tabela(s), identifique o filtro pedido e encontre quais colunas serão utilizadas.
    - Quando envolver mais de uma tabela, garanta o relacionamento correto entre elas.
⚠️ Nunca use o RAG como fonte de dados numéricos.
⚠️ Nunca invente nomes de colunas. Use somente o que está explicitamente no RAG ou nos DataFrames.
⚠️ Nunca altere a fórmula de um indicador definida no RAG.
⚠️ Se o RAG estiver vazio ou irrelevante, ignore-o silenciosamente.

**CAMINHO 2: INDICADORES OFICIAIS (USO DE TOOLS)**
GATILHO: A pergunta contém **ESCRITA EXPLICITAMENTE** uma destas siglas: [ICMQ, IDF, IMP, OEMCP, OEMPP, KMFalhas, QETG, QETT, CDTDM, CAIEFO, QVA, QVV, TIC, TO, TOPP, Preventivas Liquidadas]
Se (e SOMENTE SE) a sigla estiver presente, chame a tool correspondente (ex: `calcular_icmq`).
SIGA OS SEGUINTES PASSOS:
1. Encontre a função tool com base no nome exato do indicador presente na pergunta:
    - Se a pergunta contém a string "ICMQ" -> use tool `calcular_icmq`
    - Se a pergunta contém a string "IDF" -> use tool `calcular_idf`
    - Se a pergunta contém a string "IMP" -> use tool `calcular_imp`
    - Se a pergunta contém a string "OEMCP" -> use tool `calcular_oemcp`
    - Se a pergunta contém a string "OEMPP" -> use tool `calcular_oempp`
    - Se a pergunta contém a string "Preventivas Liquidadas" -> use tool `calcular_preventivas_liquidadas`
    - Se a pergunta contém a string "KmFalhas" -> use tool `calcular_km_falhas`
    - Se a pergunta contém a string "QETG" -> use tool `calcular_qetg`
    - Se a pergunta contém a string "QETT" -> use tool `calcular_qett`
    - Se a pergunta contém a string "CDTDM" -> use tool `calcular_cdtdm`
    - Se a pergunta contém a string "CAIEFO" -> use tool `calcular_caiefo`
    - Se a pergunta contém a string "QVA" -> use tool `calcular_qva`
    - Se a pergunta contém a string "QVV" -> use tool `calcular_qvv`
    - Se a pergunta contém a string "TIC" -> use tool `calcular_tic`
    - Se a pergunta contém a string "TO" -> use tool `calcular_to`
    - Se a pergunta contém a string "TOPP" -> use tool `calcular_topp`
2. Verifique se o usuário menciona data: Se o usuário especificar um período temporal (ex: "mês passado", "em 2024"), converta para ISO **AAAA-MM-DD**:
    - `data_inicial`: Data de início.
    - `data_final`: Data de fim.
    - Exemplo: "Custo em Setembro de 2024" -> data_inicial="2024-09-01", data_final="2024-09-30".
    - Exemplo: "Até hoje" -> data_final="DATA_ATUAL".
    - Se não houver data especificada, não preencha esses campos nas tools.

------------------------------------------------------------------------------------------------------------------------------

### REGRAS DE PROCESSAMENTO E CÁLCULO
1. Use exclusivamente os dados existentes nos DataFrames carregados.
2. Não utilize conhecimento externo além do RAG.
3. Nunca invente colunas, valores, totais ou estatísticas.
4. Em caso de múltiplas tabelas, identifique aquela que contém a informação pela descrição do RAG.
5. Sempre realize cálculos reais quando possível.

### REGRAS DE RESPOSTAS
1. Responda sempre em português (Brasil).
2. Não explique métodos, cálculos internos ou passos. Apenas entregue o resultado final de forma objetiva e clara.
3. Sempre utilize o padrão brasileiro de formatação:
   - Valores monetários devem ser apresentados em reais (R$).
   - Datas devem seguir o formato DD/MM/AAAA.
   - Números devem seguir o padrão brasileiro:
     • ponto (.) para milhar
     • vírgula (,) para decimais

### MENSAGENS PADRONIZADAS (OBRIGATÓRIO)
Dados insuficientes:
“Não é possível responder com base nos dados, pois não há dados suficientes.”
Assunto fora do contexto:
“Este assunto está fora do contexto do dataset. Faça uma pergunta relacionada aos dados.”

TABELAS DISPONÍVEIS
{texto_dados_disponiveis}

CONTEXTO RAG
Use apenas para interpretação e mapeamento conceitual.
{contexto_documentacao}
 
PERGUNTA ATUAL 
{pergunta}

ESTILO DA RESPOSTA
Direta, objetiva, clara, amigável e sem explicar métodos nem cálculos
"""

        try:
            resposta = func_timeout(50, agente.invoke, args=({"input": prompt},))
            
            texto = resposta.get("output", None) or resposta.get("output_text", None) or str(resposta)
            print("\n" + Fore.BLUE + "🤖 Resposta:" + Style.RESET_ALL)
            print(texto)
            print("-" * 60)

        except FunctionTimedOut:
            # Mensagem em branco (padrão do sistema), sem códigos de cor
            print("\nO processamento excedeu o limite de tempo.")
            print("Não foi possível gerar uma resposta a tempo. Por favor, tente uma pergunta mais simples ou específica.")
            print("-" * 60)

        except Exception as e: 
            log_error(f"Erro ao processar: {e}")
            print("-" * 60)

if __name__ == "__main__":
    main()