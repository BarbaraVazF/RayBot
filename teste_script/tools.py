import pandas as pd
from langchain.tools import tool
from typing import Optional
import unicodedata
from pydantic import BaseModel, Field
import traceback
import os

# Configuração de cores para logs (necessário para os prints dentro das tools)
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

# ====================================================
# Variáveis Globais e Utilitários para Tools
# ====================================================

# Variável global para armazenar os DFs dentro do escopo das tools
DADOS_GLOBAIS = []

def set_dados_globais(lista_dfs):
    """Função para receber os dados do main.py e disponibilizar para as tools"""
    global DADOS_GLOBAIS
    DADOS_GLOBAIS = lista_dfs

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

# ====================================================
# Schema e Definição das Tools
# ====================================================

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