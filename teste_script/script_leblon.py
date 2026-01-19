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
from langchain.tools import tool
from typing import Optional
import unicodedata
from pydantic import BaseModel, Field

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
# Tools
# ====================================================

# Variável global para armazenar os DFs e ser acessada pelas tools - evitar ter que passar dataframes complexos via argumento da tool do LLM
DADOS_GLOBAIS = []

def get_df_by_name(partial_name):
    """Retorna o primeiro DF cujo nome de origem contenha partial_name."""
    global DADOS_GLOBAIS
    for df in DADOS_GLOBAIS:
        if "__origem" in df.columns:
            nome_origem = str(df["__origem"].iloc[0]).lower()
            if partial_name.lower() in nome_origem:
                return df
    return None

def normalizar_texto(texto):
    """Remove acentos e coloca em minúsculas para comparação."""
    if not isinstance(texto, str):
        return str(texto)
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII').lower()

def aplicar_filtro_inteligente(df, termo_busca, valor_busca):
    """
    Procura TODAS as colunas que contenham 'termo_busca' (ex: 'empresa').
    Testa o filtro em cada uma. Se retornar linhas > 0, assume que aquela é a correta.
    Retorna: (DataFrame Filtrado, Nome da Coluna Usada) ou (None, None) se falhar.
    """
    termo = normalizar_texto(termo_busca)
    val = str(valor_busca).strip().lower()
    
    # 1. Identificar todas as colunas candidatas
    colunas_candidatas = []
    for col_original in df.columns:
        col_norm = normalizar_texto(col_original)
        # Verifica se o termo está na coluna (ex: 'empresa' em 'ctm[nome empresa]')
        if termo in col_norm:
            colunas_candidatas.append(col_original)
            
    if not colunas_candidatas:
        return None, None

    print(f"   🔎 Colunas candidatas para '{termo_busca}': {colunas_candidatas}")

    # 2. Tentar filtrar em cada candidata
    for col in colunas_candidatas:
        # Cria uma máscara segura convertendo tudo para string minúscula
        mask = df[col].astype(str).str.strip().str.lower() == val
        df_temp = df[mask]
        
        if len(df_temp) > 0:
            print(f"   ✅ Sucesso filtrando por: {col}")
            return df_temp, col
            
    # Se testou todas e nenhuma retornou dados (ex: procurou 'Leblon' na coluna de Código)
    return pd.DataFrame(), None # Retorna DF vazio mas não None, indicando que tentou

def encontrar_coluna_flexivel(df, termo_busca):
    """
    Encontra coluna ignorando case e acentuação.
    Ex: termo="onibus" encontra "ctm[ônibus]"
    """
    termo = normalizar_texto(termo_busca)
    
    # Mapa: {nome_normalizado: nome_real_no_dataframe}
    mapa_colunas = {normalizar_texto(c): c for c in df.columns}
    
    # 1. Tentativa exata (normalizada)
    if termo in mapa_colunas:
        return mapa_colunas[termo]
        
    # 2. Tentativa por contência (ex: 'onibus' dentro de 'ctm[onibus]')
    for col_norm, col_real in mapa_colunas.items():
        if termo in col_norm:
            return col_real
            
    return None

# --- Definição dos Schemas de Entrada ---
# Schema Pydantic
class InputCalculoKPI(BaseModel):
    filtro_coluna: Optional[str] = Field(default=None, description="Nome da coluna (ex: 'onibus', 'empresa')")
    filtro_valor: Optional[str] = Field(default=None, description="Valor do filtro (ex: 'b 1151', 'Leblon')")

@tool(args_schema=InputCalculoKPI)
def calcular_icmq(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o ICMQ (Custo / Km).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'ICMQ' OU 'ÍNDICE DE CUSTO'.
    ⚠️ PROIBIDO USAR para perguntas gerais como "Qual o custo total?", "Qual ônibus gastou mais?" ou "Quanto foi rodado?".
    Para essas perguntas gerais, use Pandas diretamente."""
    print(f"\n{Fore.CYAN}🛠️ TOOL ICMQ CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    
    try:
        df_ctm = get_df_by_name("CTM")
        df_ind = get_df_by_name("IND003")

        if df_ctm is None or df_ind is None: return "Erro: Tabelas CTM ou IND003 sumiram."

        df_ctm_filt = df_ctm.copy()
        df_ind_filt = df_ind.copy()
        msg_filtro = ""

        # --- Lógica de Filtro Inteligente ---
        if filtro_coluna and filtro_valor:
            # Filtra CTM
            res_ctm, col_usada_ctm = aplicar_filtro_inteligente(df_ctm_filt, filtro_coluna, filtro_valor)
            if res_ctm is not None and not res_ctm.empty:
                df_ctm_filt = res_ctm
            elif res_ctm is not None and res_ctm.empty:
                 # Achou a coluna mas o valor não bateu (pode ser que na CTM não tenha esse dado, mas na IND sim)
                 # Mantém vazio ou ignora? Vamos manter vazio para segurança
                 df_ctm_filt = res_ctm 
            
            # Filtra IND003
            res_ind, col_usada_ind = aplicar_filtro_inteligente(df_ind_filt, filtro_coluna, filtro_valor)
            if res_ind is not None and not res_ind.empty:
                df_ind_filt = res_ind
            elif res_ind is not None and res_ind.empty:
                df_ind_filt = res_ind

            # Verifica se o filtro "matou" os dados
            if len(df_ctm_filt) == 0 and len(df_ind_filt) == 0:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou resultados em nenhuma tabela."
            
            msg_filtro = f"(Filtro aplicado em: {col_usada_ctm or 'N/A'} e {col_usada_ind or 'N/A'})"

        # Colunas de Cálculo (Busca fixa para cálculo)
        _, col_custo = aplicar_filtro_inteligente(df_ctm_filt, "valorgasto", "") # Apenas busca nome
        # Hack: aplicar_filtro retorna None se valor for vazio, então usamos encontrar_coluna_flexivel antiga ou buscamos manual
        col_custo = next((c for c in df_ctm_filt.columns if "valorgasto" in normalizar_texto(c)), None)
        col_km = next((c for c in df_ind_filt.columns if "kmrodado" in normalizar_texto(c)), None)

        if not col_custo: return "Erro: Coluna 'ValorGasto' não encontrada."
        if not col_km: return "Erro: Coluna 'KmRodado' não encontrada."

        custo_total = pd.to_numeric(df_ctm_filt[col_custo], errors='coerce').fillna(0).sum()
        km_total = pd.to_numeric(df_ind_filt[col_km], errors='coerce').fillna(0).sum()

        print(f"   -> Custo: {custo_total:,.2f} | Km: {km_total:,.2f}")

        if km_total == 0:
            return f"ICMQ: Indefinido (Km=0). Custo: R$ {custo_total:,.2f} {msg_filtro}"

        icmq = custo_total / km_total
        return f"O ICMQ é R$ {icmq:,.4f}/Km. (Custo: R$ {custo_total:,.2f} / Km: {km_total:,.2f}) {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando ICMQ: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_idf(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o IDF (Índice de Falhas).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'IDF', 'ÍNDICE DE FALHAS' OU 'DESEMPENHO DA FROTA'.
    ⚠️ PROIBIDO USAR para perguntas como "Quais ônibus tiveram mais saídas?", "Quantas trocas ocorreram?" ou rankings.
    Para contagens e rankings, use Pandas diretamente."""
    print(f"\n{Fore.CYAN}🛠️ TOOL IDF CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    try:
        df_saidas = get_df_by_name("MANT004").copy()
        df_trocas = get_df_by_name("MANT001").copy()
        
        msg_filtro = ""

        if filtro_coluna and filtro_valor:
            # Filtro Inteligente MANT004
            res_saida, col_s = aplicar_filtro_inteligente(df_saidas, filtro_coluna, filtro_valor)
            if res_saida is not None: df_saidas = res_saida
            
            # Filtro Inteligente MANT001
            res_troca, col_t = aplicar_filtro_inteligente(df_trocas, filtro_coluna, filtro_valor)
            if res_troca is not None: df_trocas = res_troca

            if df_saidas.empty and df_trocas.empty:
                 return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' zerou os dados."
            
            msg_filtro = f"(Filtro: {col_s or 'N/A'} / {col_t or 'N/A'})"

        # Colunas de Cálculo
        col_prog = next((c for c in df_saidas.columns if "oidfcvprogramada" in normalizar_texto(c)), None)
        col_doc = next((c for c in df_trocas.columns if "oiddocumento" in normalizar_texto(c)), None)

        qtd_saidas = df_saidas[col_prog].nunique() if col_prog else 0
        qtd_trocas = df_trocas[col_doc].nunique() if col_doc else 0
        
        print(f"   -> Saídas: {qtd_saidas} | Trocas: {qtd_trocas}")

        if qtd_saidas == 0:
            return f"IDF: Indefinido (0 Saídas). Trocas: {qtd_trocas} {msg_filtro}"

        idf = (qtd_saidas - qtd_trocas) / qtd_saidas
        return f"O IDF é {idf:.2%} (Saídas: {qtd_saidas} - Trocas: {qtd_trocas}) {msg_filtro}"

    except Exception as e:
        return f"Erro IDF: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_imp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o IMP (Índice de Manutenção Preventiva).
    Fórmula: Preventivas / (Corretivas + Preventivas).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'IMP', 'PREVENTIVA' ou 'INDICE DE MANUTENÇÃO'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL IMP CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    
    try:
        # 1. Carregar Tabela MANT002
        df_mant = get_df_by_name("MANT002")
        if df_mant is None:
            return "Erro: Tabela MANT002 (Execução de Serviço) não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # 2. Aplicar Filtros Inteligentes (ex: Empresa, Ônibus)
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None:
                df_filt = res_filt
            
            if df_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' retornou 0 registros na MANT002."
            
            msg_filtro = f"(Filtro aplicado em: {col_usada or 'N/A'})"

        # 3. Identificar Colunas Necessárias (Flexível)
        # Busca colunas que contenham "tipomanutenção" e "oiddocumento" ignorando case/acentos
        col_tipo = next((c for c in df_filt.columns if "tipomanutenção" in normalizar_texto(c) or "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        if not col_tipo or not col_id:
            return f"Erro: Colunas 'TipoManutenção' ou 'OIDDocumento' não encontradas na MANT002."

        # 4. Calcular Contagens (Preventiva vs Corretiva)
        # Normaliza o conteúdo da coluna de tipo para comparação segura
        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)

        # Regra: Preventiva = 'preventiva' ou 'inspeção'
        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        # Regra: Corretiva = 'corretiva'
        mask_corr = series_tipo.str.contains('corretiva', case=False, regex=True)

        qtd_prev = df_filt[mask_prev][col_id].nunique()
        qtd_corr = df_filt[mask_corr][col_id].nunique()
        total = qtd_prev + qtd_corr

        print(f"   -> Preventivas: {qtd_prev} | Corretivas: {qtd_corr} | Total: {total}")

        # 5. Calcular Indicador
        if total == 0:
            return f"IMP: Indefinido (0 manutenções registradas). {msg_filtro}"

        imp = qtd_prev / total
        
        return (f"O IMP é {imp:.2%} "
                f"(Preventivas: {qtd_prev} / Corretivas: {qtd_corr} / Total: {total}) {msg_filtro}")

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando IMP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_oemcp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o OEMCP (Ordens Corretivas Pendentes).
    Contagem de OIDDocumento únicos.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMCP (CORRIGIDA - UNIQUE ID) CHAMADA{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # Filtros Inteligentes
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt
            msg_filtro = f"(Filtro: {col_usada or 'N/A'})"

        # Identificação de Colunas
        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        # Lógica Situação
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: # Fallback seguro
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas 'Tipo', 'Situação' ou 'OIDDocumento' não encontradas."

        # Regras
        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        # 1. Tipo = Corretiva
        mask_corr = series_tipo.str.contains('corretiva', case=False, regex=True)
        # 2. Status = Pendentes
        status_alvo = ["aguardando liberacao", "parado", "liberado", "em execucao"]
        mask_status = series_situacao.apply(lambda x: any(s in x for s in status_alvo))

        df_final = df_filt[mask_corr & mask_status]
        
        # CONTAGEM ÚNICA DE DOCUMENTOS
        qtd_docs = df_final[col_id].nunique()

        if qtd_docs == 0: return f"OEMCP: 0 ordens. {msg_filtro}"
        return f"O OEMCP é {qtd_docs} ordens (Corretivas em status pendente). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro OEMCP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_oempp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o OEMPP (Preventivas/Inspeções Pendentes).
    Contagem de OIDDocumento únicos.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMPP (CORRIGIDA - UNIQUE ID) CHAMADA{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt
            msg_filtro = f"(Filtro: {col_usada or 'N/A'})"

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: 
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas essenciais não encontradas na MANT002."

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        # 1. Tipo = Preventiva ou Inspeção
        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        # 2. Status = Pendentes
        status_alvo = ["aguardando liberacao", "parado", "liberado", "em execucao"]
        mask_status = series_situacao.apply(lambda x: any(s in x for s in status_alvo))

        df_final = df_filt[mask_prev & mask_status]
        
        # CONTAGEM ÚNICA DE DOCUMENTOS
        qtd_docs = df_final[col_id].nunique()

        if qtd_docs == 0: return f"OEMPP: 0 ordens. {msg_filtro}"
        return f"O OEMPP é {qtd_docs} ordens (Preventivas/Inspeções pendentes). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro OEMPP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_preventivas_liquidadas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula total de Preventivas e Inspeções com status LIQUIDADO.
    Contagem de OIDDocumento únicos.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL PREVENTIVAS LIQUIDADAS CHAMADA{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt
            msg_filtro = f"(Filtro: {col_usada or 'N/A'})"

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        # Busca coluna de Situação
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: 
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas essenciais não encontradas na MANT002."

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        # 1. Tipo = Preventiva ou Inspeção
        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        
        # 2. Status = Liquidado
        # Usamos contains para garantir que ache "liquidado" mesmo se tiver espaços ou sufixos
        mask_status = series_situacao.str.contains('liquidado', case=False, regex=True)

        df_final = df_filt[mask_prev & mask_status]
        
        # CONTAGEM ÚNICA DE DOCUMENTOS
        qtd_docs = df_final[col_id].nunique()

        if qtd_docs == 0: return f"Preventivas Liquidadas: 0. {msg_filtro}"
        return f"Total de Preventivas Liquidadas: {qtd_docs}. {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro Prev. Liquidadas: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_km_falhas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o KmFalhas (Km Rodado / Quantidade de Quebras).
    Fórmula: Soma(IND003[KmRodado]) / Contagem(MANT001 onde Tipo contem 'quebra').
    USE APENAS SE A PERGUNTA MENCIONAR 'KMFALHAS', 'KM POR QUEBRA' OU 'MKBF'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL KM_FALHAS CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    
    try:
        # 1. Carregar as tabelas necessárias
        df_km = get_df_by_name("IND003")
        df_ocorrencias = get_df_by_name("MANT001")

        if df_km is None or df_ocorrencias is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_ocorrencias_filt = df_ocorrencias.copy()
        msg_filtro = ""

        # 2. Aplicação de Filtros Inteligentes (ex: por Empresa ou Ônibus)
        if filtro_coluna and filtro_valor:
            # Filtra tabela de Km
            res_km, col_usada_km = aplicar_filtro_inteligente(df_km_filt, filtro_coluna, filtro_valor)
            if res_km is not None: 
                df_km_filt = res_km
            
            # Filtra tabela de Ocorrências
            res_oco, col_usada_oco = aplicar_filtro_inteligente(df_ocorrencias_filt, filtro_coluna, filtro_valor)
            if res_oco is not None: 
                df_ocorrencias_filt = res_oco
            
            # Se zerou ambos, retorna erro
            if df_km_filt.empty and df_ocorrencias_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados em IND003 ou MANT001."

            msg_filtro = f"(Filtro: {col_usada_km or 'N/A'} e {col_usada_oco or 'N/A'})"

        # 3. Identificação das Colunas Alvo
        # Busca coluna de Km na IND003
        col_km_nome = next((c for c in df_km_filt.columns if "kmrodado" in normalizar_texto(c)), None)
        
        # Busca coluna de Tipo na MANT001
        col_tipo_nome = next((c for c in df_ocorrencias_filt.columns if "tipo" in normalizar_texto(c)), None)

        if not col_km_nome: return "Erro: Coluna 'KmRodado' não encontrada na IND003."
        if not col_tipo_nome: return "Erro: Coluna 'Tipo' não encontrada na MANT001."

        # 4. Cálculo do Numerador (Total Km)
        total_km = pd.to_numeric(df_km_filt[col_km_nome], errors='coerce').fillna(0).sum()

        # 5. Cálculo do Denominador (Total Quebras)
        # Normaliza o texto da coluna Tipo para buscar a string 'quebra'
        series_tipo = df_ocorrencias_filt[col_tipo_nome].astype(str).apply(normalizar_texto)
        
        # Conta quantas linhas contêm a palavra 'quebra'
        qtd_quebras = series_tipo.str.contains('quebra', case=False, regex=False).sum()

        print(f"   -> Km Total: {total_km:,.2f} | Quebras identificadas: {qtd_quebras}")

        # 6. Resultado Final
        if qtd_quebras == 0:
            return f"KmFalhas: Indefinido (0 quebras registradas). Km Total: {total_km:,.2f}. {msg_filtro}"

        km_falhas = total_km / qtd_quebras
        
        return (f"O KmFalhas é {km_falhas:,.2f} Km/Quebra. "
                f"(Km Total: {total_km:,.0f} / Quebras: {qtd_quebras}) {msg_filtro}")

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando KmFalhas: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_qetg(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o QETG (Km Rodado / Trocas na Garagem).
    Fórmula: Soma(IND003[KmRodado]) / Contagem Distinta(MANT001[OIDDocumento] onde Tipo contém 'Garagem').
    USE APENAS SE A PERGUNTA MENCIONAR 'QETG' ou 'TROCAS NA GARAGEM'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL QETG CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    
    try:
        # 1. Carregar tabelas
        df_km = get_df_by_name("IND003")
        df_mant = get_df_by_name("MANT001")

        if df_km is None or df_mant is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_mant_filt = df_mant.copy()
        msg_filtro = ""

        # 2. Aplicar Filtros Inteligentes (Empresa/Ônibus)
        if filtro_coluna and filtro_valor:
            # Filtro na tabela de Km
            res_km, col_usada_km = aplicar_filtro_inteligente(df_km_filt, filtro_coluna, filtro_valor)
            if res_km is not None: df_km_filt = res_km
            
            # Filtro na tabela de Manutenção
            res_mant, col_usada_mant = aplicar_filtro_inteligente(df_mant_filt, filtro_coluna, filtro_valor)
            if res_mant is not None: df_mant_filt = res_mant
            
            if df_km_filt.empty and df_mant_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados."

            msg_filtro = f"(Filtro: {col_usada_km or 'N/A'} e {col_usada_mant or 'N/A'})"

        # 3. Identificar Colunas
        # Km na IND003
        col_km_nome = next((c for c in df_km_filt.columns if "kmrodado" in normalizar_texto(c)), None)
        
        # Tipo e ID na MANT001
        col_tipo_nome = next((c for c in df_mant_filt.columns if "tipo" in normalizar_texto(c)), None)
        col_id_nome = next((c for c in df_mant_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        if not col_km_nome: return "Erro: Coluna 'KmRodado' não encontrada na IND003."
        if not col_tipo_nome or not col_id_nome: return "Erro: Colunas 'Tipo' ou 'OIDDocumento' não encontradas na MANT001."

        # 4. Calcular Numerador (Km Total)
        total_km = pd.to_numeric(df_km_filt[col_km_nome], errors='coerce').fillna(0).sum()

        # 5. Calcular Denominador (Trocas Garagem - Contagem Distinta)
        # Normaliza coluna Tipo para busca segura
        series_tipo = df_mant_filt[col_tipo_nome].astype(str).apply(normalizar_texto)
        
        # Máscara: Onde aparece 'garagem'
        mask_garagem = series_tipo.str.contains('garagem', case=False, regex=False)
        
        # Contagem de IDs únicos filtrados
        qtd_trocas = df_mant_filt[mask_garagem][col_id_nome].nunique()

        print(f"   -> Km Total: {total_km:,.2f} | Trocas Garagem (Distintas): {qtd_trocas}")

        # 6. Retorno
        if qtd_trocas == 0:
            return f"QETG: Indefinido (0 trocas de garagem). Km Total: {total_km:,.2f}. {msg_filtro}"

        qetg = total_km / qtd_trocas
        
        return (f"O QETG é {qetg:,.2f} Km/Troca. "
                f"(Km Total: {total_km:,.0f} / Trocas Garagem: {qtd_trocas}) {msg_filtro}")

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando QETG: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_qett(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o QETT (Km Rodado / Trocas no Terminal).
    Fórmula: Soma(IND003[KmRodado]) / Contagem Distinta(MANT001[OIDDocumento] onde Tipo contém 'Terminal').
    USE APENAS SE A PERGUNTA MENCIONAR 'QETT' ou 'TROCAS NO TERMINAL'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL QETT CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")
    
    try:
        # 1. Carregar tabelas
        df_km = get_df_by_name("IND003")
        df_mant = get_df_by_name("MANT001")

        if df_km is None or df_mant is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_mant_filt = df_mant.copy()
        msg_filtro = ""

        # 2. Aplicar Filtros Inteligentes (Empresa/Ônibus)
        if filtro_coluna and filtro_valor:
            # Filtro na tabela de Km
            res_km, col_usada_km = aplicar_filtro_inteligente(df_km_filt, filtro_coluna, filtro_valor)
            if res_km is not None: df_km_filt = res_km
            
            # Filtro na tabela de Manutenção
            res_mant, col_usada_mant = aplicar_filtro_inteligente(df_mant_filt, filtro_coluna, filtro_valor)
            if res_mant is not None: df_mant_filt = res_mant
            
            if df_km_filt.empty and df_mant_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados."

            msg_filtro = f"(Filtro: {col_usada_km or 'N/A'} e {col_usada_mant or 'N/A'})"

        # 3. Identificar Colunas
        # Km na IND003
        col_km_nome = next((c for c in df_km_filt.columns if "kmrodado" in normalizar_texto(c)), None)
        
        # Tipo e ID na MANT001
        col_tipo_nome = next((c for c in df_mant_filt.columns if "tipo" in normalizar_texto(c)), None)
        col_id_nome = next((c for c in df_mant_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        if not col_km_nome: return "Erro: Coluna 'KmRodado' não encontrada na IND003."
        if not col_tipo_nome or not col_id_nome: return "Erro: Colunas 'Tipo' ou 'OIDDocumento' não encontradas na MANT001."

        # 4. Calcular Numerador (Km Total)
        total_km = pd.to_numeric(df_km_filt[col_km_nome], errors='coerce').fillna(0).sum()

        # 5. Calcular Denominador (Trocas Terminal - Contagem Distinta)
        # Normaliza coluna Tipo para busca segura
        series_tipo = df_mant_filt[col_tipo_nome].astype(str).apply(normalizar_texto)
        
        # Máscara: Onde aparece 'terminal'
        mask_terminal = series_tipo.str.contains('terminal', case=False, regex=False)
        
        # Contagem de IDs únicos filtrados
        qtd_trocas = df_mant_filt[mask_terminal][col_id_nome].nunique()

        print(f"   -> Km Total: {total_km:,.2f} | Trocas Terminal (Distintas): {qtd_trocas}")

        # 6. Retorno
        if qtd_trocas == 0:
            return f"QETT: Indefinido (0 trocas em terminal). Km Total: {total_km:,.2f}. {msg_filtro}"

        qett = total_km / qtd_trocas
        
        return (f"O QETT é {qett:,.2f} Km/Troca. "
                f"(Km Total: {total_km:,.0f} / Trocas Terminal: {qtd_trocas}) {msg_filtro}")

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando QETT: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_cdtdm(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador CDTDM.
    Lógica: Soma da coluna 'Valor' da tabela INDMANTMANUAL 
    onde a coluna 'Simbolo' é igual a 'CDTDML'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL CDTDM CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}'")

    try:
        # 1. Carregar tabela
        df_manual = get_df_by_name("INDMANTMANUAL")
        
        if df_manual is None:
            return "Erro: Tabela INDMANTMANUAL não encontrada."

        df_filt = df_manual.copy()
        msg_filtro = ""

        # 2. Aplicar Filtros Inteligentes (Empresa/Filial/etc)
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
                msg_filtro = f"(Filtro: {col_usada or 'N/A'})"
            
            if df_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados na INDMANTMANUAL."

        # 3. Identificar Colunas (Valor e Simbolo)
        col_valor = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_simbolo = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)

        if not col_valor: return "Erro: Coluna 'Valor' não encontrada na INDMANTMANUAL."
        if not col_simbolo: return "Erro: Coluna 'Simbolo' não encontrada na INDMANTMANUAL."

        # 4. Aplicar a Lógica do Indicador (Simbolo == CDTDML)
        # Normalização para garantir que ache independente de maiúsculas/minúsculas ou espaços
        series_simbolo = df_filt[col_simbolo].astype(str).str.strip().str.upper()
        
        mask_cdtdm = series_simbolo == "CDTDML"
        
        df_final = df_filt[mask_cdtdm]
        
        # 5. Somar Valor
        total_valor = pd.to_numeric(df_final[col_valor], errors='coerce').fillna(0).sum()
        qtd_registros = len(df_final)

        print(f"   -> Total CDTDM: R$ {total_valor:,.2f} | Registros: {qtd_registros}")

        if qtd_registros == 0:
            return f"CDTDM: R$ 0,00 (Nenhum registro com Simbolo 'CDTDML' encontrado). {msg_filtro}"

        return f"O valor do indicador CDTDM é R$ {total_valor:,.2f} ({qtd_registros} registros encontrados). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando CDTDM: {str(e)}"

# ==============================================================================
# FUNÇÃO AUXILIAR (Lógica Unificada de Prefixo)
# ==============================================================================
def _calcular_indicador_prefixo(nome_indicador: str, string_busca: str, qtd_letras: int, filtro_coluna: Optional[str], filtro_valor: Optional[str]) -> str:
    """
    Função genérica para calcular soma de valores baseada no início da descrição (prefixo).
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL {nome_indicador} (PREFIXO '{string_busca}') CHAMADA:{Style.RESET_ALL} Filtro='{filtro_coluna}={filtro_valor}'")

    try:
        df_manual = get_df_by_name("INDMANTMANUAL")
        if df_manual is None: 
            return "Erro: Tabela INDMANTMANUAL não encontrada."

        df_filt = df_manual.copy()
        msg_filtro = ""

        # 1. Aplicação de Filtros Inteligentes (Empresa, Filial, etc)
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
                msg_filtro = f"(Filtro: {col_usada or 'N/A'})"
            
            # Se o filtro zerou os dados, retorna erro específico
            if df_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados na INDMANTMANUAL."

        # 2. Identificação das Colunas (Valor e Descricao)
        col_valor = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_desc = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)

        if not col_valor: return "Erro: Coluna 'Valor' não encontrada na INDMANTMANUAL."
        if not col_desc: return "Erro: Coluna 'Descricao' não encontrada na INDMANTMANUAL."

        # 3. Lógica de Prefixo (Starts With)
        # Normaliza: converte para string, remove espaços nas pontas e põe em maiúsculo
        series_desc = df_filt[col_desc].astype(str).str.strip().str.upper()
        
        # Verifica se os primeiros N caracteres correspondem à string de busca
        mask_prefixo = series_desc.str.slice(0, qtd_letras) == string_busca.upper()
        
        df_final = df_filt[mask_prefixo]
        
        # 4. Soma e Contagem
        total_valor = pd.to_numeric(df_final[col_valor], errors='coerce').fillna(0).sum()
        qtd_registros = len(df_final)

        print(f"   -> Total {nome_indicador}: R$ {total_valor:,.2f} | Registros: {qtd_registros}")

        if qtd_registros == 0:
            return f"{nome_indicador}: R$ 0,00 (Nenhum registro iniciando com '{string_busca}' encontrado). {msg_filtro}"

        return f"O valor do indicador {nome_indicador} é R$ {total_valor:,.2f} ({qtd_registros} registros encontrados). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando {nome_indicador}: {str(e)}"

# ==============================================================================
# TOOLS KPI (Definições Individuais)
# ==============================================================================

@tool(args_schema=InputCalculoKPI)
def calcular_caiefo(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador CAIEFO.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 6 primeiras letras de Descricao são 'CAIEFO'.
    """
    return _calcular_indicador_prefixo("CAIEFO", "CAIEFO", 6, filtro_coluna, filtro_valor)

@tool(args_schema=InputCalculoKPI)
def calcular_qva(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador QVA.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'QVA'.
    """
    return _calcular_indicador_prefixo("QVA", "QVA", 3, filtro_coluna, filtro_valor)

@tool(args_schema=InputCalculoKPI)
def calcular_qvv(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador QVV.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'QVV'.
    """
    return _calcular_indicador_prefixo("QVV", "QVV", 3, filtro_coluna, filtro_valor)

@tool(args_schema=InputCalculoKPI)
def calcular_tic(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador TIC.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'TIC'.
    """
    return _calcular_indicador_prefixo("TIC", "TIC", 3, filtro_coluna, filtro_valor)

@tool(args_schema=InputCalculoKPI)
def calcular_to(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador TO.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 2 primeiras letras de Descricao são 'TO'.
    """
    return _calcular_indicador_prefixo("TO", "TO", 2, filtro_coluna, filtro_valor)

@tool(args_schema=InputCalculoKPI)
def calcular_topp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """Calcula o indicador TOPP.
    Lógica: Soma INDMANTMANUAL[Valor] onde as 4 primeiras letras de Descricao são 'TOPP'.
    """
    return _calcular_indicador_prefixo("TOPP", "TOPP", 4, filtro_coluna, filtro_valor)

# ====================================================
# Função principal - analisar_dados_csv
# ====================================================
def main():
    # 1. CARREGAR DADOS
    converter_todos_xlsx(PASTA_CSV)
    arquivos = glob.glob(os.path.join(PASTA_CSV, "*.csv"))
    lista_dfs = carregar_dados(arquivos)


    # Expor dados às tools
    global DADOS_GLOBAIS
    DADOS_GLOBAIS = lista_dfs

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
    # Lista de tools
    tools_kpi = [calcular_icmq, calcular_idf, calcular_imp, calcular_oemcp, calcular_oempp, calcular_preventivas_liquidadas, calcular_km_falhas, calcular_qetg, calcular_qett, 
    calcular_cdtdm, calcular_caiefo, calcular_qva, calcular_qvv, calcular_tic, calcular_to, calcular_topp]

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

        # ===============================
        # PROMPT 
        # ===============================
        prompt = f"""

Você é um analista de dados sênior especializado em análise tabular.
Você tem acesso a {len(lista_dfs)} DataFrames carregados separadamente: df1, df2, etc.
Esses DataFrames não são unidos automaticamente.

USO DO RAG (OBRIGATÓRIO E SIMPLIFICADO)
O RAG contém definições oficiais e estruturais, organizadas em três abas:
    - Aba 1 - Descrição das Planilhas: explica o propósito de cada tabela.
    - Aba 2 - Descrição das Colunas: detalha o significado e uso de cada coluna.
PROCESSO OBRIGATÓRIO DE INTERPRETAÇÃO:
1. Mapear o segmento da pergunta (palavras-chave conceituais)
2. Identificar se a pergunta envolve um indicador. Apenas se ela tiver o nome de um desses indicadores:
    - Se a pergunta for sobre **ICMQ** (Custo por Km) -> tool `calcular_icmq`.
    - Se a pergunta for sobre **IDF** (Índice de Falhas) -> tool `calcular_idf`.
    - Se a pergunta for sobre **IMP** (Índice de Manutenção Preventiva) -> tool `calcular_imp`.
    - Se a pergunta for sobre **OEMCP** (Ordens Corretivas Pendentes) -> tool `calcular_oemcp`.
    - Se a pergunta for sobre **OEMPP** (Ordens Preventivas Pendentes) -> tool `calcular_oempp`.
    - Se a pergunta for sobre **Preventivas Liquidadas** (Executadas/Finalizadas) -> tool `calcular_preventivas_liquidadas`.
    - Se a pergunta for sobre **KmFalhas** (Índice de KM Rodado por Falha) -> `tool calcular_km_falhas`
    - Se a pergunta for sobre **QETG** (Índice de KM Rodado por Falha na Garagem) -> `calcular_qetg`
    - Se a pergunta for sobre **QETT** (Índice de KM Rodado por Falha no Terminal) -> `calcular_qett`
    - Se a pergunta for sobre **CDTDM** (Índice de Valor por CDTDM) -> `calcular_cdtdm`
    - Se a pergunta for sobre **CAIEFO** (Índice de Valor por CAIEFO) -> `calcular_caiefo`
    - Se a pergunta for sobre **QVA** (Índice de Valor por QVA) -> `calcular_qva`
    - Se a pergunta for sobre **QVV** (Índice de Valor por QVV) -> `calcular_qvv`
    - Se a pergunta for sobre **TIC** (Índice de Valor por TIC) -> `calcular_tic`
    - Se a pergunta for sobre **TO** (Índice de Valor por TO) -> `calcular_to`
    - Se a pergunta for sobre **TOPP** (Índice de Valor por TOPP) -> `calcular_topp`
    Observação: Você só tem permissão para usar as tools se o usuário mencionar **EXPLICITAMENTE**:
    - Palavras-chave: "ICMQ", "IDF", "IMP", "OEMCP", "OEMPP", "Preventivas Liquidadas", "Preventivas Finalizadas"; "KMFalhas"; "QETG"; "QETT"; "CDTDM"; "CAIEFO"; "QVA"; "QVV"; "TIC"; "TO"; "TOPP".
3. Caso não seja um indicador, a partir do segmento identificado, encontrar qual tabela mais tem similaridade com ele a partir da descrição e das suas colunas
4. Selecionar a(s) tabela(s), identificar o filtro pedido na solicitação e encontrar quais colunas serão utilizadas - considerar tudo no cálculo e na resposta
    - Quando envolver mais de uma tabela, garantir o relacionamento correto entre elas (ex.: identificadores comuns)
⚠️ Nunca use o RAG como fonte de dados numéricos.
⚠️ Nunca invente nomes de colunas. Use somente o que está explicitamente no RAG ou nos DataFrames.
⚠️ Nunca altere a fórmula de um indicador definida no RAG.
⚠️ Se o RAG estiver vazio, irrelevante ou não ajudar no termo consultado, ignore-o silenciosamente.

REGRAS INTERPRETAÇÃO
1. Você deve interpretar a pergunta pelo seu SIGNIFICADO e INTENÇÃO, e não pela forma exata das palavras. Sempre normalize a pergunta antes de responder: 
   - Considere singular e plural como equivalentes.
   - Considere variações verbais, abreviações e linguagem informal.
   - Reconheça sinônimos, termos equivalentes e variações semânticas.
   - Ignore erros leves de digitação ou variações comuns de escrita.
   - Ignore termos desnecessários na solicitação. Foque nas palavras chave.
2. Caso a pergunta não seja compreendida de imediato, reformule-a internamente usando sinônimos, termos equivalentes e linguagem mais neutra, e tente interpretá-la novamente antes de pedir esclarecimentos.
3. Caso a informação solicitada não exista, responda com as mensagens padronizadas.

REGRAS DE PROCESSAMENTO E CÁLCULO
1. Use exclusivamente os dados existentes nos DataFrames carregados.
2. Não utilize conhecimento externo além do RAG.
3. Nunca invente colunas, valores, totais ou estatísticas.
4. Em caso de múltiplas tabelas, identifique aquela que contém a informação pela descrição do RAG.
5. Sempre realize cálculos reais quando possível.

REGRAS DE RESPOSTAS
1. Responda sempre em português (Brasil).
2. Não explique métodos, cálculos internos ou passos. Apenas entregue o resultado final de forma objetiva e clara.
3. Sempre utilize o padrão brasileiro de formatação:
   - Valores monetários devem ser apresentados em reais (R$), com:
    - Datas devem seguir o formato DD/MM/AAAA.
   - Números devem seguir o padrão brasileiro:
     • ponto (.) para milhar
     • vírgula (,) para decimais

MENSAGENS PADRONIZADAS (OBRIGATÓRIO)
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