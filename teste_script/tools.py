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

MAPA_DATAS = {
    "INDMANTMANUAL": "DtMovimento",
    "CTM": "DtGasto",
    "MANT001": "DtOcorrencia",
    "MANT002": "DtManutencao",
    "MANT004": "DtSaida",
    "IND003": "DtOperacao"
}

def aplicar_filtro_periodo(df, nome_tabela_referencia, data_ini, data_fim):
    """
    Filtra o DataFrame baseado na coluna de data específica.
    Versão HÍBRIDA: Aceita mistura de formatos (ISO e BR) no mesmo arquivo.
    """
    if not data_ini and not data_fim:
        return df, ""

    col_data_nome = MAPA_DATAS.get(nome_tabela_referencia)
    
    if not col_data_nome:
        col_data_nome = next((c for c in df.columns if "data" in normalizar_texto(c) or "dt" in normalizar_texto(c)), None)
    else:
        col_data_nome = encontrar_coluna_flexivel(df, col_data_nome)

    if not col_data_nome:
        print(f"{Fore.YELLOW}[WARN] Coluna de data não encontrada para {nome_tabela_referencia}.{Style.RESET_ALL}")
        return df, " (⚠️ Data ñ encontrada)"

    try:
        df_temp = df.copy()
        
        # 1. Limpeza básica
        series_raw = df_temp[col_data_nome].astype(str).str.strip()
        
        # 2. DEBUG: Mostra amostra para conferência
        print(f"{Fore.MAGENTA}🔎 FORMATO DATA [{nome_tabela_referencia}]: {series_raw.head(3).tolist()}{Style.RESET_ALL}")

        # 3. TENTATIVA INTELIGENTE (Format='mixed')
        # O 'mixed' permite que uma linha seja DD/MM/AAAA e a outra AAAA-MM-DD
        try:
            df_temp[col_data_nome] = pd.to_datetime(series_raw, dayfirst=True, format='mixed', errors='coerce')
        except:
            # Fallback para Pandas antigo que não suporta 'mixed'
            df_temp[col_data_nome] = pd.to_datetime(series_raw, dayfirst=True, errors='coerce')
            
            # Se falhou muito, tenta recuperar formato ISO explícito nas linhas com erro
            mask_erro = df_temp[col_data_nome].isna()
            if mask_erro.sum() > 0:
                print(f"   ℹ️ Tentando recuperação ISO para {mask_erro.sum()} linhas falhas...")
                recuperado = pd.to_datetime(series_raw[mask_erro], format='%Y-%m-%d', errors='coerce')
                df_temp.loc[mask_erro, col_data_nome] = recuperado

        # 4. Verifica erros restantes
        qtd_erros = df_temp[col_data_nome].isna().sum()
        if qtd_erros > 0:
            pct = (qtd_erros / len(df)) * 100
            if pct > 5: # Só avisa se perder mais de 5%
                print(f"{Fore.RED}[CRÍTICO] Ainda falhou ler {qtd_erros} ({pct:.1f}%) linhas em '{nome_tabela_referencia}'.{Style.RESET_ALL}")
            df_temp = df_temp.dropna(subset=[col_data_nome])

        mask = pd.Series(True, index=df_temp.index)
        txt_periodo = ""

        if data_ini:
            dt_i = pd.to_datetime(data_ini)
            mask &= (df_temp[col_data_nome] >= dt_i)
            txt_periodo += f" >= {data_ini}"
        
        if data_fim:
            # Garante final do dia (23:59:59)
            dt_f = pd.to_datetime(data_fim) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= (df_temp[col_data_nome] <= dt_f)
            txt_periodo += f" <= {data_fim}"

        indices_validos = df_temp[mask].index
        df_filtrado = df.loc[indices_validos]

        print(f"   📅 Filtro Data ({col_data_nome}): {len(df)} -> {len(df_filtrado)} registros.")
        
        if len(df_filtrado) == 0:
            return df_filtrado, f" (0 registros em {txt_periodo})"
            
        return df_filtrado, f" (Ref. Data: {txt_periodo})"

    except Exception as e:
        print(f"{Fore.RED}[ERRO] Crash filtro data: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return df, " (Erro Data)"

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
    filtro_coluna: Optional[str] = Field(default=None, description="Nome da coluna para filtro categórico (ex: 'onibus', 'empresa')")
    filtro_valor: Optional[str] = Field(default=None, description="Valor do filtro categórico (ex: 'b 1151', 'Leblon')")
    data_inicial: Optional[str] = Field(default=None, description="Data inicial no formato AAAA-MM-DD (ex: '2025-01-01')")
    data_final: Optional[str] = Field(default=None, description="Data final no formato AAAA-MM-DD (ex: '2025-01-31')")

@tool(args_schema=InputCalculoKPI)
def calcular_icmq(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o ICMQ (Custo / Km).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'ICMQ'."""
    
    # Log atualizado para mostrar as datas
    print(f"\n{Fore.CYAN}🛠️ TOOL ICMQ CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        df_ctm = get_df_by_name("CTM")
        df_ind = get_df_by_name("IND003")

        if df_ctm is None or df_ind is None: return "Erro: Tabelas CTM ou IND003 sumiram."

        # =================================================================
        # 1. FILTRO DE DATA (Aplicado nas duas tabelas envolvidas)
        # =================================================================
        # A função aplicar_filtro_periodo sabe que CTM usa "DtGasto" e IND003 usa "DtOperacao"
        # graças ao MAPA_DATAS definido globalmente.
        df_ctm_filt, msg_dt_ctm = aplicar_filtro_periodo(df_ctm, "CTM", data_inicial, data_final)
        df_ind_filt, msg_dt_ind = aplicar_filtro_periodo(df_ind, "IND003", data_inicial, data_final)

        msg_filtro = f"{msg_dt_ctm} {msg_dt_ind}"

        # Se o filtro de data removeu tudo, retornamos aviso logo aqui
        if df_ctm_filt.empty and df_ind_filt.empty:
            return f"ICMQ: Sem dados encontrados para o período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Lógica de Filtro Inteligente Original)
        # =================================================================
        if filtro_coluna and filtro_valor:
            # Filtra CTM (usando o DF já filtrado por data)
            res_ctm, col_usada_ctm = aplicar_filtro_inteligente(df_ctm_filt, filtro_coluna, filtro_valor)
            if res_ctm is not None:
                df_ctm_filt = res_ctm
            
            # Filtra IND003 (usando o DF já filtrado por data)
            res_ind, col_usada_ind = aplicar_filtro_inteligente(df_ind_filt, filtro_coluna, filtro_valor)
            if res_ind is not None:
                df_ind_filt = res_ind

            # Verifica se o filtro categórico "matou" os dados restantes
            if df_ctm_filt.empty and df_ind_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou resultados (dentro do período selecionado)."
            
            msg_filtro += f" (Filtro Coluna: {col_usada_ctm or 'N/A'} e {col_usada_ind or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO (Mantido igual)
        # =================================================================
        # Colunas de Cálculo (Busca fixa para cálculo)
        _, col_custo = aplicar_filtro_inteligente(df_ctm_filt, "valorgasto", "") 
        # Fallback manual caso o intelligent search retorne None por tabela vazia
        if not col_custo: col_custo = next((c for c in df_ctm_filt.columns if "valorgasto" in normalizar_texto(c)), None)
        
        col_km = next((c for c in df_ind_filt.columns if "kmrodado" in normalizar_texto(c)), None)

        if not col_custo: return "Erro: Coluna 'ValorGasto' não encontrada."
        if not col_km: return "Erro: Coluna 'KmRodado' não encontrada."

        custo_total = pd.to_numeric(df_ctm_filt[col_custo], errors='coerce').fillna(0).sum()
        km_total = pd.to_numeric(df_ind_filt[col_km], errors='coerce').fillna(0).sum()

        print(f"   -> Custo: {custo_total:,.2f} | Km: {km_total:,.2f}")

        if km_total == 0:
            return f"ICMQ: Indefinido (Km=0). Custo: R$ {custo_total:,.2f} {msg_filtro}"

        icmq = custo_total / km_total
        return f"O ICMQ é R$ {icmq:,.4f}/Km. (Custo: R$ {custo_total:,.2f} / Km: {km_total:,.2f}) {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando ICMQ: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_idf(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o IDF (Índice de Falhas).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'IDF'."""
    
    print(f"\n{Fore.CYAN}🛠️ TOOL IDF CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        # Carrega os DataFrames originais
        df_saidas = get_df_by_name("MANT004")
        df_trocas = get_df_by_name("MANT001")

        if df_saidas is None or df_trocas is None:
            return "Erro: Tabelas MANT004 ou MANT001 não encontradas."

        # Copia para não alterar o global
        df_saidas_filt = df_saidas.copy()
        df_trocas_filt = df_trocas.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA (Aplicado nas duas tabelas)
        # =================================================================
        # MAPA_DATAS já sabe que MANT004 usa "DtSaida" e MANT001 usa "DtOcorrencia"
        df_saidas_filt, msg_dt1 = aplicar_filtro_periodo(df_saidas_filt, "MANT004", data_inicial, data_final)
        df_trocas_filt, msg_dt2 = aplicar_filtro_periodo(df_trocas_filt, "MANT001", data_inicial, data_final)
        
        msg_filtro += f"{msg_dt1} {msg_dt2}"

        # Se não houver saídas no período, não dá para calcular o índice
        if df_saidas_filt.empty:
             return f"IDF: Sem dados de Saídas (MANT004) no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Empresa, Ônibus, etc)
        # =================================================================
        if filtro_coluna and filtro_valor:
            # Filtro Inteligente MANT004
            res_saida, col_s = aplicar_filtro_inteligente(df_saidas_filt, filtro_coluna, filtro_valor)
            if res_saida is not None: 
                df_saidas_filt = res_saida
            
            # Filtro Inteligente MANT001
            res_troca, col_t = aplicar_filtro_inteligente(df_trocas_filt, filtro_coluna, filtro_valor)
            if res_troca is not None: 
                df_trocas_filt = res_troca

            if df_saidas_filt.empty:
                 return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' zerou as Saídas (MANT004)."
            
            msg_filtro += f" (Filtro Coluna: {col_s or 'N/A'} / {col_t or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        # Identificação de Colunas
        col_prog = next((c for c in df_saidas_filt.columns if "oidfcvprogramada" in normalizar_texto(c)), None)
        col_doc = next((c for c in df_trocas_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        # Contagem
        qtd_saidas = df_saidas_filt[col_prog].nunique() if col_prog else 0
        qtd_trocas = df_trocas_filt[col_doc].nunique() if col_doc else 0
        
        print(f"   -> Saídas: {qtd_saidas} | Trocas: {qtd_trocas}")

        if qtd_saidas == 0:
            return f"IDF: Indefinido (0 Saídas). Trocas: {qtd_trocas} {msg_filtro}"

        idf = (qtd_saidas - qtd_trocas) / qtd_saidas
        return f"O IDF é {idf:.2%} (Saídas: {qtd_saidas} - Trocas: {qtd_trocas}) {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando IDF: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_imp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o IMP (Índice de Manutenção Preventiva).
    Fórmula: Preventivas / (Corretivas + Preventivas).
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'IMP'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL IMP CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        # 1. Carregar Tabela MANT002
        df_mant = get_df_by_name("MANT002")
        if df_mant is None:
            return "Erro: Tabela MANT002 (Execução de Serviço) não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        # Filtra MANT002 (Geralmente usa DtManutencao)
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"IMP: Sem dados no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (ex: Empresa, Ônibus)
        # =================================================================
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None:
                df_filt = res_filt
            
            if df_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' retornou 0 registros (no período)."
            
            msg_filtro += f" (Filtro aplicado em: {col_usada or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        # Identificar Colunas Necessárias (Flexível)
        col_tipo = next((c for c in df_filt.columns if "tipomanutenção" in normalizar_texto(c) or "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        if not col_tipo or not col_id:
            return f"Erro: Colunas 'TipoManutenção' ou 'OIDDocumento' não encontradas na MANT002."

        # Calcular Contagens (Preventiva vs Corretiva)
        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)

        # Regra: Preventiva = 'preventiva' ou 'inspeção'
        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        # Regra: Corretiva = 'corretiva'
        mask_corr = series_tipo.str.contains('corretiva', case=False, regex=True)

        qtd_prev = df_filt[mask_prev][col_id].nunique()
        qtd_corr = df_filt[mask_corr][col_id].nunique()
        total = qtd_prev + qtd_corr

        print(f"   -> Preventivas: {qtd_prev} | Corretivas: {qtd_corr} | Total: {total}")

        # Calcular Indicador
        if total == 0:
            return f"IMP: Indefinido (0 manutenções registradas). {msg_filtro}"

        imp = qtd_prev / total
        
        return (f"O IMP é {imp:.2%} "
                f"(Preventivas: {qtd_prev} / Corretivas: {qtd_corr} / Total: {total}) {msg_filtro}")

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando IMP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_oemcp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o OEMCP (Ordens Corretivas Pendentes). Contagem de OIDDocumento únicos.
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'OEMCP'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMCP (CORRIGIDA - UNIQUE ID) CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        # Filtra MANT002 (Geralmente usa DtManutencao conforme MAPA_DATAS)
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"OEMCP: Sem dados no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Empresa, Ônibus, etc)
        # =================================================================
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
            
            # Se o filtro categórico zerou tudo
            if df_filt.empty:
                 return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' zerou os dados (dentro do período)."
            
            msg_filtro += f" (Filtro Coluna: {col_usada or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO (Lógica Original Mantida)
        # =================================================================
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
def calcular_oempp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o OEMPP (Preventivas/Inspeções Pendentes). Contagem de OIDDocumento únicos.
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'OEMPP'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMPP (CORRIGIDA - UNIQUE ID) CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        # Filtra MANT002 (Usa DtManutencao conforme MAPA_DATAS)
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"OEMPP: Sem dados no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Empresa, Ônibus, etc)
        # =================================================================
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
            
            if df_filt.empty:
                 return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' zerou os dados (dentro do período)."
            
            msg_filtro += f" (Filtro Coluna: {col_usada or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
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
def calcular_preventivas_liquidadas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula total de Preventivas e Inspeções com status LIQUIDADO. Contagem de OIDDocumento únicos.
    USE APENAS SE A PERGUNTA MENCIONAR EXPLICITAMENTE 'Preventivas Liquidadas'.
    NÃO É RELACIONADA A DINHEIRO, NÃO USE R$
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL PREVENTIVAS LIQUIDADAS CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"Quantidade de Preventivas Liquidadas: 0 (Sem dados no período). {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO
        # =================================================================
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
            
            if df_filt.empty:
                 return f"Quantidade de Preventivas Liquidadas: 0 (Filtro '{filtro_coluna}={filtro_valor}' zerou os dados). {msg_filtro}"
            
            msg_filtro += f" (Filtro Coluna: {col_usada or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
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
        
        # 2. Status = Liquidado
        mask_status = series_situacao.str.contains('liquidado', case=False, regex=True)

        df_final = df_filt[mask_prev & mask_status]
        
        # CONTAGEM ÚNICA DE DOCUMENTOS
        qtd_docs = df_final[col_id].nunique()

        if qtd_docs == 0: return f"Quantidade de Preventivas Liquidadas: 0. {msg_filtro}"
        
        # AQUI ESTÁ A CORREÇÃO PRINCIPAL: "Quantidade ... ordens"
        return f"Quantidade de Preventivas Liquidadas: {qtd_docs} ordens. {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro Prev. Liquidadas: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_km_falhas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o KmFalhas (Km Rodado / Quantidade de Quebras). 
    Fórmula: Soma(IND003[KmRodado]) / Contagem(MANT001 onde Tipo contem 'quebra').
    USE APENAS SE A PERGUNTA MENCIONAR 'KMFALHAS'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL KM_FALHAS CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        # 1. Carregar as tabelas necessárias
        df_km = get_df_by_name("IND003")
        df_ocorrencias = get_df_by_name("MANT001")

        if df_km is None or df_ocorrencias is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_ocorrencias_filt = df_ocorrencias.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA (Aplicado em IND003 e MANT001)
        # =================================================================
        # O MAPA_DATAS sabe que IND003 usa DtOperacao e MANT001 usa DtOcorrencia
        df_km_filt, msg_dt_km = aplicar_filtro_periodo(df_km_filt, "IND003", data_inicial, data_final)
        df_ocorrencias_filt, msg_dt_oco = aplicar_filtro_periodo(df_ocorrencias_filt, "MANT001", data_inicial, data_final)
        
        msg_filtro += f"{msg_dt_km} {msg_dt_oco}"

        # Se zerou a tabela de Km, nem adianta continuar (não existe MKBF sem Km)
        if df_km_filt.empty:
             return f"KmFalhas: Sem dados de quilometragem (IND003) no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (ex: por Empresa ou Ônibus)
        # =================================================================
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
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados (no período selecionado)."

            msg_filtro += f" (Filtro Coluna: {col_usada_km or 'N/A'} e {col_usada_oco or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        # Identificação das Colunas Alvo
        col_km_nome = next((c for c in df_km_filt.columns if "kmrodado" in normalizar_texto(c)), None)
        col_tipo_nome = next((c for c in df_ocorrencias_filt.columns if "tipo" in normalizar_texto(c)), None)

        if not col_km_nome: return "Erro: Coluna 'KmRodado' não encontrada na IND003."
        if not col_tipo_nome: return "Erro: Coluna 'Tipo' não encontrada na MANT001."

        # 4. Cálculo do Numerador (Total Km)
        total_km = pd.to_numeric(df_km_filt[col_km_nome], errors='coerce').fillna(0).sum()

        # 5. Cálculo do Denominador (Total Quebras)
        series_tipo = df_ocorrencias_filt[col_tipo_nome].astype(str).apply(normalizar_texto)
        qtd_quebras = series_tipo.str.contains('quebra', case=False, regex=False).sum()

        print(f"   -> Km Total: {total_km:,.2f} | Quebras identificadas: {qtd_quebras}")

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
def calcular_qetg(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o QETG (Km Rodado / Trocas na Garagem). 
    Fórmula: Soma(IND003[KmRodado]) / Contagem Distinta(MANT001[OIDDocumento] onde Tipo contém 'Garagem').
    USE APENAS SE A PERGUNTA MENCIONAR 'QETG'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL QETG CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        # 1. Carregar tabelas
        df_km = get_df_by_name("IND003")
        df_mant = get_df_by_name("MANT001")

        if df_km is None or df_mant is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_mant_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        # Filtra IND003 e MANT001 pelas datas
        df_km_filt, msg_dt_km = aplicar_filtro_periodo(df_km_filt, "IND003", data_inicial, data_final)
        df_mant_filt, msg_dt_mant = aplicar_filtro_periodo(df_mant_filt, "MANT001", data_inicial, data_final)

        msg_filtro += f"{msg_dt_km} {msg_dt_mant}"

        if df_km_filt.empty:
             return f"QETG: Sem dados de quilometragem (IND003) no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Empresa/Ônibus)
        # =================================================================
        if filtro_coluna and filtro_valor:
            # Filtro na tabela de Km
            res_km, col_usada_km = aplicar_filtro_inteligente(df_km_filt, filtro_coluna, filtro_valor)
            if res_km is not None: df_km_filt = res_km
            
            # Filtro na tabela de Manutenção
            res_mant, col_usada_mant = aplicar_filtro_inteligente(df_mant_filt, filtro_coluna, filtro_valor)
            if res_mant is not None: df_mant_filt = res_mant
            
            if df_km_filt.empty and df_mant_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados (no período)."

            msg_filtro += f"(Filtro: {col_usada_km or 'N/A'} e {col_usada_mant or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        # Identificar Colunas
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

        print(f"   -> Km Total: {total_km:,.2f} | Trocas Garagem (Distintas): {qtd_trocas}")

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
def calcular_qett(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o QETT (Km Rodado / Trocas no Terminal). 
    Fórmula: Soma(IND003[KmRodado]) / Contagem Distinta(MANT001[OIDDocumento] onde Tipo contém 'Terminal').
    USE APENAS SE A PERGUNTA MENCIONAR 'QETT'.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL QETT CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")
    
    try:
        # 1. Carregar tabelas
        df_km = get_df_by_name("IND003")
        df_mant = get_df_by_name("MANT001")

        if df_km is None or df_mant is None:
            return "Erro: Tabelas IND003 (Km) ou MANT001 (Ocorrências) não encontradas."

        df_km_filt = df_km.copy()
        df_mant_filt = df_mant.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        # Filtra IND003 e MANT001 pelas datas
        df_km_filt, msg_dt_km = aplicar_filtro_periodo(df_km_filt, "IND003", data_inicial, data_final)
        df_mant_filt, msg_dt_mant = aplicar_filtro_periodo(df_mant_filt, "MANT001", data_inicial, data_final)

        msg_filtro += f"{msg_dt_km} {msg_dt_mant}"

        if df_km_filt.empty:
             return f"QETT: Sem dados de quilometragem (IND003) no período solicitado. {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO (Empresa/Ônibus)
        # =================================================================
        if filtro_coluna and filtro_valor:
            # Filtro na tabela de Km
            res_km, col_usada_km = aplicar_filtro_inteligente(df_km_filt, filtro_coluna, filtro_valor)
            if res_km is not None: df_km_filt = res_km
            
            # Filtro na tabela de Manutenção
            res_mant, col_usada_mant = aplicar_filtro_inteligente(df_mant_filt, filtro_coluna, filtro_valor)
            if res_mant is not None: df_mant_filt = res_mant
            
            if df_km_filt.empty and df_mant_filt.empty:
                return f"Erro: O filtro '{filtro_coluna}={filtro_valor}' não retornou dados (no período)."

            msg_filtro += f"(Filtro: {col_usada_km or 'N/A'} e {col_usada_mant or 'N/A'})"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        # Identificar Colunas
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

        print(f"   -> Km Total: {total_km:,.2f} | Trocas Terminal (Distintas): {qtd_trocas}")

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
def calcular_cdtdm(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador CDTDM. 
    Lógica: Soma da coluna 'Valor' da tabela INDMANTMANUAL .
    USE APENAS SE A PERGUNTA MENCIONAR 'CDTDM'."""
    print(f"\n{Fore.CYAN}🛠️ TOOL CDTDM CHAMADA:{Style.RESET_ALL} Coluna='{filtro_coluna}', Valor='{filtro_valor}', Data='{data_inicial} a {data_final}'")

    try:
        # 1. Carregar tabela
        df_manual = get_df_by_name("INDMANTMANUAL")
        
        if df_manual is None:
            return "Erro: Tabela INDMANTMANUAL não encontrada."

        df_filt = df_manual.copy()
        msg_filtro = ""

        # =================================================================
        # 1. FILTRO DE DATA
        # =================================================================
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "INDMANTMANUAL", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"CDTDM: 0.00 (Sem dados no período). {msg_filtro}"

        # =================================================================
        # 2. FILTRO CATEGÓRICO
        # =================================================================
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
                msg_filtro += f" (Filtro Coluna: {col_usada or 'N/A'})"
            
            if df_filt.empty:
                return f"CDTDM: 0.00 (Filtro zerou os dados). {msg_filtro}"

        # =================================================================
        # 3. CÁLCULO
        # =================================================================
        col_valor = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_simbolo = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)

        if not col_valor or not col_simbolo: return "Erro: Colunas não encontradas."

        series_simbolo = df_filt[col_simbolo].astype(str).str.strip().str.upper()
        mask_cdtdm = series_simbolo == "CDTDML"
        df_final = df_filt[mask_cdtdm]
        
        total_valor = pd.to_numeric(df_final[col_valor], errors='coerce').fillna(0).sum()
        qtd_registros = len(df_final)

        print(f"   -> Total CDTDM: {total_valor:,.2f} | Registros: {qtd_registros}")

        if qtd_registros == 0:
            return f"CDTDM: 0.00 (Nenhum registro encontrado). {msg_filtro}"

        # MUDANÇA DRÁSTICA AQUI:
        # Usamos a palavra "PONTUAÇÃO" e instruímos explicitamente a não usar moeda.
        return f"A Pontuação Total do CDTDM é {total_valor:,.2f} pontos. (NOTA PARA O AGENTE: Este valor é um índice, NÃO adicione R$ ou formatação monetária). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando CDTDM: {str(e)}"

# ==============================================================================
# FUNÇÃO AUXILIAR (Lógica Unificada de Prefixo com DATA)
# ==============================================================================
# ==============================================================================
# FUNÇÃO AUXILIAR (Lógica Unificada de Prefixo com DATA)
# ==============================================================================
def _calcular_indicador_prefixo(nome_indicador: str, string_busca: str, qtd_letras: int, 
                                filtro_coluna: Optional[str], filtro_valor: Optional[str],
                                data_inicial: Optional[str], data_final: Optional[str]) -> str:
    """
    Função genérica para calcular soma de valores baseada no início da descrição (prefixo).
    Agora com suporte a filtro de DATA.
    """
    print(f"\n{Fore.CYAN}🛠️ TOOL {nome_indicador} (PREFIXO '{string_busca}') CHAMADA:{Style.RESET_ALL} Filtro='{filtro_coluna}={filtro_valor}', Data='{data_inicial} a {data_final}'")

    try:
        df_manual = get_df_by_name("INDMANTMANUAL")
        if df_manual is None: 
            return "Erro: Tabela INDMANTMANUAL não encontrada."

        df_filt = df_manual.copy()
        msg_filtro = ""

        # 1. FILTRO DE DATA
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "INDMANTMANUAL", data_inicial, data_final)
        msg_filtro += msg_data

        if df_filt.empty:
            return f"{nome_indicador}: 0.00 (Sem dados no período). {msg_filtro}"

        # 2. FILTRO CATEGÓRICO
        if filtro_coluna and filtro_valor:
            res_filt, col_usada = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: 
                df_filt = res_filt
                msg_filtro += f" (Filtro Coluna: {col_usada or 'N/A'})"
            
            if df_filt.empty:
                return f"{nome_indicador}: 0.00 (Filtro zerou os dados). {msg_filtro}"

        # 3. CÁLCULO
        col_valor = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_desc = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)

        if not col_valor or not col_desc: return "Erro: Colunas Valor/Descricao não encontradas."

        series_desc = df_filt[col_desc].astype(str).str.strip().str.upper()
        mask_prefixo = series_desc.str.slice(0, qtd_letras) == string_busca.upper()
        df_final = df_filt[mask_prefixo]
        
        total_valor = pd.to_numeric(df_final[col_valor], errors='coerce').fillna(0).sum()
        qtd_registros = len(df_final)

        print(f"   -> Total {nome_indicador}: {total_valor:,.2f} | Registros: {qtd_registros}")

        if qtd_registros == 0:
            return f"{nome_indicador}: 0.00 (Nenhum registro encontrado). {msg_filtro}"

        # RETORNO FORÇADO: Substitui "O valor é..." por "Índice acumulado..."
        return f"Índice acumulado {nome_indicador}: {total_valor:,.2f} (unidades/pontos). {msg_filtro}"

    except Exception as e:
        traceback.print_exc()
        return f"Erro processando {nome_indicador}: {str(e)}"

# ==============================================================================
# TOOLS WRAPPERS ATUALIZADOS
# ==============================================================================

@tool(args_schema=InputCalculoKPI)
def calcular_caiefo(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador CAIEFO. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 6 primeiras letras de Descricao são 'CAIEFO'.
    USE APENAS SE A PERGUNTA MENCIONAR 'CAIEFO'."""
    return _calcular_indicador_prefixo("CAIEFO", "CAIEFO", 6, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_qva(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador QVA. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'QVA'.
    USE APENAS SE A PERGUNTA MENCIONAR 'QVA'."""
    return _calcular_indicador_prefixo("QVA", "QVA", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_qvv(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador QVV. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'QVV'.
    USE APENAS SE A PERGUNTA MENCIONAR 'QVV'."""
    return _calcular_indicador_prefixo("QVV", "QVV", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_tic(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador TIC. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 3 primeiras letras de Descricao são 'TIC'.
    USE APENAS SE A PERGUNTA MENCIONAR 'TIC'."""
    return _calcular_indicador_prefixo("TIC", "TIC", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_to(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador TO. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 2 primeiras letras de Descricao são 'TO'.
    USE APENAS SE A PERGUNTA MENCIONAR 'TO'."""
    return _calcular_indicador_prefixo("TO", "TO", 2, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_topp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o indicador TOPP. 
    Lógica: Soma INDMANTMANUAL[Valor] onde as 4 primeiras letras de Descricao são 'TOPP'.
    USE APENAS SE A PERGUNTA MENCIONAR 'TOPP'."""
    return _calcular_indicador_prefixo("TOPP", "TOPP", 4, filtro_coluna, filtro_valor, data_inicial, data_final)