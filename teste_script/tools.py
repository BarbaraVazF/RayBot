import datetime
import pandas as pd
from langchain.tools import tool
from typing import Optional
import unicodedata
from pydantic import BaseModel, Field
import traceback
import re 

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

# ====================================================
# Variáveis Globais e Utilitários para Tools
# ====================================================

DADOS_GLOBAIS = []

def set_dados_globais(lista_dfs):
    global DADOS_GLOBAIS
    DADOS_GLOBAIS = lista_dfs

def get_df_by_name(partial_name):
    global DADOS_GLOBAIS
    for df in DADOS_GLOBAIS:
        if "__origem" in df.columns:
            nome_origem = str(df["__origem"].iloc[0]).lower()
            if partial_name.lower() in nome_origem:
                return df
    return None

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return str(texto)
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII').lower()

def aplicar_filtro_inteligente(df, termo_busca, valor_busca):
    termo = normalizar_texto(termo_busca)
    val = str(valor_busca).strip().lower()
    
    colunas_candidatas = []
    for col_original in df.columns:
        col_norm = normalizar_texto(col_original)
        if termo in col_norm:
            colunas_candidatas.append(col_original)
            
    if not colunas_candidatas:
        return None, None

    print(f"   🔎 Colunas candidatas para '{termo_busca}': {colunas_candidatas}")

    for col in colunas_candidatas:
        mask = df[col].astype(str).str.strip().str.lower() == val
        df_temp = df[mask]
        
        if len(df_temp) > 0:
            print(f"   ✅ Sucesso filtrando por: {col}")
            return df_temp, col
            
    return pd.DataFrame(), None

MAPA_DATAS = {
    "INDMANTMANUAL": "DtMovimento",
    "CTM": "DtGasto",
    "MANT001": "DtOcorrencia",
    "MANT002": "DtManutencao",
    "MANT004": "DataSaida",
    "IND003": "DtOperacao"
}

def aplicar_filtro_periodo(df, nome_tabela_referencia, data_ini, data_fim):
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
        series_raw = df_temp[col_data_nome].astype(str).str.strip()
        
        try:
            df_temp[col_data_nome] = pd.to_datetime(series_raw, dayfirst=True, format='mixed', errors='coerce')
        except:
            df_temp[col_data_nome] = pd.to_datetime(series_raw, dayfirst=True, errors='coerce')
            mask_erro = df_temp[col_data_nome].isna()
            if mask_erro.sum() > 0:
                recuperado = pd.to_datetime(series_raw[mask_erro], format='%Y-%m-%d', errors='coerce')
                df_temp.loc[mask_erro, col_data_nome] = recuperado

        df_temp = df_temp.dropna(subset=[col_data_nome])
        mask = pd.Series(True, index=df_temp.index)
        txt_periodo = ""

        if data_ini:
            dt_i = pd.to_datetime(data_ini)
            mask &= (df_temp[col_data_nome] >= dt_i)
            txt_periodo += f" >= {data_ini}"
        
        if data_fim:
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
        return df, " (Erro Data)"

def encontrar_coluna_flexivel(df, termo_busca):
    termo = normalizar_texto(termo_busca)
    mapa_colunas = {normalizar_texto(c): c for c in df.columns}
    if termo in mapa_colunas: return mapa_colunas[termo]
    for col_norm, col_real in mapa_colunas.items():
        if termo in col_norm: return col_real
    return None

# ====================================================
# Schema Padrão
# ====================================================

class InputCalculoKPI(BaseModel):
    filtro_coluna: Optional[str] = Field(default=None, description="Nome da coluna para filtro categórico (ex: 'onibus', 'empresa')")
    filtro_valor: Optional[str] = Field(default=None, description="Valor do filtro categórico (ex: 'b 1151', 'Leblon')")
    data_inicial: Optional[str] = Field(default=None, description="Data inicial no formato AAAA-MM-DD")
    data_final: Optional[str] = Field(default=None, description="Data final no formato AAAA-MM-DD")

# ====================================================
# TOOLS EXISTENTES (Mantidas originais, apenas importadas)
# ====================================================
# Nota: Estou mantendo suas funções originais exatamente como estavam, 
# para garantir que o cálculo base não mude.

@tool(args_schema=InputCalculoKPI)
def calcular_icmq(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o ICMQ (Custo / Km).
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL ICMQ CHAMADA:{Style.RESET_ALL} {data_inicial} a {data_final}")
    try:
        df_ctm = get_df_by_name("CTM")
        df_ind = get_df_by_name("IND003")
        if df_ctm is None or df_ind is None: return "Erro: Tabelas sumiram."

        df_ctm_filt, _ = aplicar_filtro_periodo(df_ctm, "CTM", data_inicial, data_final)
        df_ind_filt, _ = aplicar_filtro_periodo(df_ind, "IND003", data_inicial, data_final)

        if df_ctm_filt.empty and df_ind_filt.empty: return "ICMQ: Sem dados."

        if filtro_coluna and filtro_valor:
            r1, _ = aplicar_filtro_inteligente(df_ctm_filt, filtro_coluna, filtro_valor)
            if r1 is not None: df_ctm_filt = r1
            r2, _ = aplicar_filtro_inteligente(df_ind_filt, filtro_coluna, filtro_valor)
            if r2 is not None: df_ind_filt = r2

        _, col_custo = aplicar_filtro_inteligente(df_ctm_filt, "valorgasto", "")
        if not col_custo: col_custo = next((c for c in df_ctm_filt.columns if "valorgasto" in normalizar_texto(c)), None)
        col_km = next((c for c in df_ind_filt.columns if "kmrodado" in normalizar_texto(c)), None)

        if not col_custo or not col_km: return "Erro: Colunas não encontradas."

        custo_total = pd.to_numeric(df_ctm_filt[col_custo], errors='coerce').fillna(0).sum()
        km_total = pd.to_numeric(df_ind_filt[col_km], errors='coerce').fillna(0).sum()

        if km_total == 0: return f"ICMQ: Indefinido (Km=0). Custo: R$ {custo_total:,.2f}"
        icmq = custo_total / km_total
        return f"O ICMQ é R$ {icmq:,.4f}/Km (Lembre-se: Quanto MENOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_idf(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o IDF (Índice de Falhas).
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL IDF CHAMADA:{Style.RESET_ALL}")
    try:
        df_saidas = get_df_by_name("MANT004")
        df_trocas = get_df_by_name("MANT001")
        if not df_saidas is not None: return "Erro dados."
        
        df_s, _ = aplicar_filtro_periodo(df_saidas.copy(), "MANT004", data_inicial, data_final)
        df_t, _ = aplicar_filtro_periodo(df_trocas.copy(), "MANT001", data_inicial, data_final)
        
        if filtro_coluna and filtro_valor:
            r1, _ = aplicar_filtro_inteligente(df_s, filtro_coluna, filtro_valor)
            if r1 is not None: df_s = r1
            r2, _ = aplicar_filtro_inteligente(df_t, filtro_coluna, filtro_valor)
            if r2 is not None: df_t = r2

        col_prog = next((c for c in df_s.columns if "oidfcvprogramada" in normalizar_texto(c)), None)
        col_doc = next((c for c in df_t.columns if "oiddocumento" in normalizar_texto(c)), None)

        qtd_saidas = df_s[col_prog].nunique() if col_prog else 0
        qtd_trocas = df_t[col_doc].nunique() if col_doc else 0

        if qtd_saidas == 0: return "IDF: Indefinido (0 Saídas)."
        idf = (qtd_saidas - qtd_trocas) / qtd_saidas
        return f"O IDF é {idf:.2%} (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_imp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o IMP.
    Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL IMP CHAMADA:{Style.RESET_ALL}")
    try:
        df = get_df_by_name("MANT002")
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "MANT002", data_inicial, data_final)
        if filtro_coluna and filtro_valor:
            r, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if r is not None: df_filt = r

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False)
        mask_corr = series_tipo.str.contains('corretiva', case=False)
        
        qtd_prev = df_filt[mask_prev][col_id].nunique()
        qtd_corr = df_filt[mask_corr][col_id].nunique()
        total = qtd_prev + qtd_corr
        
        if total == 0: return "IMP: Indefinido."
        imp = qtd_prev / total
        return f"O IMP é {imp:.2%} (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_oemcp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o OEMCP (Ordens Corretivas Pendentes).
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMCP CHAMADA:{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)

        if df_filt.empty:
            return f"OEMCP: Sem dados no período solicitado. {msg_data}"

        if filtro_coluna and filtro_valor:
            res_filt, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        # Tenta 'situacaodocumento', depois 'status', depois 'situacao' (ignorando datas)
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: 
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas essenciais (Tipo/Situação/OID) não encontradas."

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        mask_corr = series_tipo.str.contains('corretiva', case=False, regex=True)
        status_alvo = ["aguardando liberacao", "parado", "liberado", "em execucao"]
        mask_status = series_situacao.apply(lambda x: any(s in x for s in status_alvo))

        df_final = df_filt[mask_corr & mask_status]
        qtd_docs = df_final[col_id].nunique()

        return f"O OEMCP é {qtd_docs} ordens (Lembre-se: Quanto MENOR, MELHOR.)."

    except Exception as e:
        traceback.print_exc()
        return f"Erro OEMCP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_oempp(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula o OEMPP (Preventivas Pendentes).
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL OEMPP CHAMADA:{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)

        if df_filt.empty:
            return f"OEMPP: Sem dados no período solicitado. {msg_data}"

        if filtro_coluna and filtro_valor:
            res_filt, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: 
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas essenciais não encontradas."

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        status_alvo = ["aguardando liberacao", "parado", "liberado", "em execucao"]
        mask_status = series_situacao.apply(lambda x: any(s in x for s in status_alvo))

        df_final = df_filt[mask_prev & mask_status]
        qtd_docs = df_final[col_id].nunique()

        return f"O OEMPP é {qtd_docs} ordens (Lembre-se: Quanto MENOR, MELHOR.)."

    except Exception as e:
        traceback.print_exc()
        return f"Erro OEMPP: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_preventivas_liquidadas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula Preventivas Liquidadas.
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL PREV. LIQUIDADAS CHAMADA:{Style.RESET_ALL}")

    try:
        df_mant = get_df_by_name("MANT002")
        if df_mant is None: return "Erro: Tabela MANT002 não encontrada."

        df_filt = df_mant.copy()
        df_filt, msg_data = aplicar_filtro_periodo(df_filt, "MANT002", data_inicial, data_final)

        if df_filt.empty:
            return f"Quantidade de Preventivas Liquidadas: 0 (Sem dados). {msg_data}"

        if filtro_coluna and filtro_valor:
            res_filt, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if res_filt is not None: df_filt = res_filt

        col_tipo = next((c for c in df_filt.columns if "tipomanutencao" in normalizar_texto(c)), None)
        col_id = next((c for c in df_filt.columns if "oiddocumento" in normalizar_texto(c)), None)
        
        col_situacao = next((c for c in df_filt.columns if "situacaodocumento" in normalizar_texto(c)), None)
        if not col_situacao: col_situacao = next((c for c in df_filt.columns if "status" in normalizar_texto(c)), None)
        if not col_situacao: 
             col_situacao = next((c for c in df_filt.columns if "situacao" in normalizar_texto(c) and not any(x in normalizar_texto(c) for x in ['dt', 'hr', 'data'])), None)

        if not col_tipo or not col_situacao or not col_id:
            return "Erro: Colunas essenciais não encontradas."

        series_tipo = df_filt[col_tipo].astype(str).apply(normalizar_texto)
        series_situacao = df_filt[col_situacao].astype(str).apply(normalizar_texto)

        mask_prev = series_tipo.str.contains('preventiva|inspecao', case=False, regex=True)
        mask_status = series_situacao.str.contains('liquidado', case=False, regex=True)

        df_final = df_filt[mask_prev & mask_status]
        qtd_docs = df_final[col_id].nunique()

        return f"Quantidade de Preventivas Liquidadas: {qtd_docs} ordens (Lembre-se: Quanto MAIOR, MELHOR.)."

    except Exception as e:
        traceback.print_exc()
        return f"Erro Prev. Liquidadas: {str(e)}"

@tool(args_schema=InputCalculoKPI)
def calcular_km_falhas(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula KmFalhas.
    Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL KMFALHAS CHAMADA:{Style.RESET_ALL}")
    try:
        df_km = get_df_by_name("IND003")
        df_oco = get_df_by_name("MANT001")
        
        df_k, _ = aplicar_filtro_periodo(df_km.copy(), "IND003", data_inicial, data_final)
        df_o, _ = aplicar_filtro_periodo(df_oco.copy(), "MANT001", data_inicial, data_final)

        if filtro_coluna and filtro_valor:
            r1, _ = aplicar_filtro_inteligente(df_k, filtro_coluna, filtro_valor)
            if r1 is not None: df_k = r1
            r2, _ = aplicar_filtro_inteligente(df_o, filtro_coluna, filtro_valor)
            if r2 is not None: df_o = r2

        col_km = next((c for c in df_k.columns if "kmrodado" in normalizar_texto(c)), None)
        col_tipo = next((c for c in df_o.columns if any(x in normalizar_texto(c) for x in ["detalhesservico", "tipo"])), None)

        total_km = pd.to_numeric(df_k[col_km], errors='coerce').fillna(0).sum()
        qtd_quebras = df_o[col_tipo].astype(str).apply(normalizar_texto).str.contains('quebra').sum()

        if qtd_quebras == 0: return f"KmFalhas: Indefinido (0 quebras). Km: {total_km}"
        res = total_km / qtd_quebras
        return f"O KmFalhas é {res:,.2f} Km/Quebra (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_qetg(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula QETG.
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL QETG CHAMADA:{Style.RESET_ALL}")
    try:
        df_km = get_df_by_name("IND003")
        df_man = get_df_by_name("MANT001")
        
        df_k, _ = aplicar_filtro_periodo(df_km.copy(), "IND003", data_inicial, data_final)
        df_m, _ = aplicar_filtro_periodo(df_man.copy(), "MANT001", data_inicial, data_final)

        if filtro_coluna and filtro_valor:
            r1, _ = aplicar_filtro_inteligente(df_k, filtro_coluna, filtro_valor)
            if r1 is not None: df_k = r1
            r2, _ = aplicar_filtro_inteligente(df_m, filtro_coluna, filtro_valor)
            if r2 is not None: df_m = r2
            
        col_km = next((c for c in df_k.columns if "kmrodado" in normalizar_texto(c)), None)
        col_tipo = next((c for c in df_m.columns if any(x in normalizar_texto(c) for x in ["detalhesservico", "tipo"])), None)
        col_id = next((c for c in df_m.columns if "oiddocumento" in normalizar_texto(c)), None)

        total_km = pd.to_numeric(df_k[col_km], errors='coerce').fillna(0).sum()
        mask = df_m[col_tipo].astype(str).apply(normalizar_texto).str.contains('garagem')
        qtd = df_m[mask][col_id].nunique()

        if qtd == 0: return f"QETG: Indefinido. Km: {total_km}"
        res = total_km / qtd
        return f"O QETG é {res:,.2f} Km/Troca (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_qett(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula QETT.
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    print(f"\n{Fore.CYAN}🛠️ TOOL QETT CHAMADA:{Style.RESET_ALL}")
    try:
        df_km = get_df_by_name("IND003")
        df_man = get_df_by_name("MANT001")
        
        df_k, _ = aplicar_filtro_periodo(df_km.copy(), "IND003", data_inicial, data_final)
        df_m, _ = aplicar_filtro_periodo(df_man.copy(), "MANT001", data_inicial, data_final)

        if filtro_coluna and filtro_valor:
            r1, _ = aplicar_filtro_inteligente(df_k, filtro_coluna, filtro_valor)
            if r1 is not None: df_k = r1
            r2, _ = aplicar_filtro_inteligente(df_m, filtro_coluna, filtro_valor)
            if r2 is not None: df_m = r2
            
        col_km = next((c for c in df_k.columns if "kmrodado" in normalizar_texto(c)), None)
        col_tipo = next((c for c in df_m.columns if any(x in normalizar_texto(c) for x in ["detalhesservico", "tipo"])), None)
        col_id = next((c for c in df_m.columns if "oiddocumento" in normalizar_texto(c)), None)

        total_km = pd.to_numeric(df_k[col_km], errors='coerce').fillna(0).sum()
        mask = df_m[col_tipo].astype(str).apply(normalizar_texto).str.contains('terminal')
        qtd = df_m[mask][col_id].nunique()

        if qtd == 0: return f"QETT: Indefinido. Km: {total_km}"
        res = total_km / qtd
        return f"O QETT é {res:,.2f} Km/Troca (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

def _calcular_indicador_prefixo(nome, prefixo, chars, f_col, f_val, d_ini, d_fim):
    """Função interna auxiliar para índices manuais."""
    try:
        df = get_df_by_name("INDMANTMANUAL")
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "INDMANTMANUAL", d_ini, d_fim)
        if f_col and f_val:
            r, _ = aplicar_filtro_inteligente(df_filt, f_col, f_val)
            if r is not None: df_filt = r
        
        col_valor = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_desc = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)
        
        if not col_valor or not col_desc: return "Erro colunas."
        
        mask = df_filt[col_desc].astype(str).str.strip().str.upper().str.slice(0, chars) == prefixo.upper()
        total = pd.to_numeric(df_filt[mask][col_valor], errors='coerce').fillna(0).sum()

        if nome == "TO":
            return f"Índice acumulado {nome}: {total:,.2f} pontos (Lembre-se: Quanto MENOR, MELHOR.)."

        if nome == "TOPP":
            return f"Índice acumulado {nome}: {total:,.2f} pontos (Lembre-se: Quanto MENOR, MELHOR.)."
        
        return f"Índice acumulado {nome}: {total:,.2f} pontos."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_cdtdm(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """Calcula CDTDM (MANTMANUAL 'CDTDML').
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    try:
        df = get_df_by_name("INDMANTMANUAL")
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "INDMANTMANUAL", data_inicial, data_final)
        if filtro_coluna and filtro_valor:
            r, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if r is not None: df_filt = r
        col_v = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_s = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)
        mask = df_filt[col_s].astype(str).str.upper().str.strip() == "CDTDML"
        total = pd.to_numeric(df_filt[mask][col_v], errors='coerce').fillna(0).sum()
        return f"A Pontuação Total do CDTDM é {total:,.2f} pontos (Lembre-se: Quanto MENOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

# Tools wrappers para prefixos
# ====================================================
# CORREÇÃO: ADICIONE AS DOCSTRINGS ABAIXO
# ====================================================

# Tools wrappers para prefixos
@tool(args_schema=InputCalculoKPI)
def calcular_caiefo(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador CAIEFO (Vistorias de Limpeza/Manutenção)."""
    return _calcular_indicador_prefixo("CAIEFO", "CAIEFO", 6, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_qva(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador QVA (Quantidade de Veículos Aprovados)."""
    return _calcular_indicador_prefixo("QVA", "QVA", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_qvv(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador QVV (Quantidade de Veículos Vistoriados)."""
    return _calcular_indicador_prefixo("QVV", "QVV", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_tic(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador TIC (Total de Itens Conformes/Corretos).
    NÃO APLIQUE FILTROS QUE NÃO SÃO SOLICITADOS NA PERGUNTA"""
    return _calcular_indicador_prefixo("TIC", "TIC", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_to(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador TO (Total de Ocorrências/Observações).
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    return _calcular_indicador_prefixo("TO", "TO", 2, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_topp(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador TOPP (Total de Ocorrências Ponderadas/Prioritárias).
    IMPORTANTE: Quanto MENOR o valor, MELHOR o resultado. Quanto MAIOR o valor, PIOR o resultado"""
    return _calcular_indicador_prefixo("TOPP", "TOPP", 4, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_tia(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula o indicador TIA (Total de Itens Avaliados)."""
    return _calcular_indicador_prefixo("TIA", "TIA", 3, filtro_coluna, filtro_valor, data_inicial, data_final)

@tool(args_schema=InputCalculoKPI)
def calcular_iavlit(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula IAVLIT (QVA/QVV).
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    try:
        df = get_df_by_name("INDMANTMANUAL")
        # 1. Filtros
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "INDMANTMANUAL", data_inicial, data_final)
        if filtro_coluna and filtro_valor:
            r, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if r is not None: df_filt = r
            
        # 2. Identifica Colunas
        col_v = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_s = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)
        col_d = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)
        
        if not col_v: return "Erro: Coluna Valor não encontrada."

        # 3. Função Auxiliar de Busca (Olha Símbolo OU Descrição)
        def calcular_soma(sigla, chars):
            mask = pd.Series(False, index=df_filt.index)
            # Tenta pelo Símbolo (exato)
            if col_s:
                mask |= (df_filt[col_s].astype(str).str.strip().str.upper() == sigla)
            # Tenta pela Descrição (prefixo)
            if col_d:
                mask |= (df_filt[col_d].astype(str).str.strip().str.upper().str.slice(0, chars) == sigla)
            return pd.to_numeric(df_filt[mask][col_v], errors='coerce').fillna(0).sum()

        # 4. Cálculos
        val_qva = calcular_soma("QVA", 3)
        val_qvv = calcular_soma("QVV", 3)
        
        print(f"   DEBUG IAVLIT -> QVA: {val_qva} | QVV: {val_qvv}")

        if val_qva == 0 and val_qvv == 0: return "O IAVLIT é 1.00 (QVA e QVV zerados)."
        if val_qvv == 0: return f"IAVLIT: Indefinido (QVA: {val_qva})."
        
        res = val_qva / val_qvv
        return f"O IAVLIT é {res:,.4f} (QVA: {val_qva:,.0f} / QVV: {val_qvv:,.0f}) (Lembre-se: Quanto MAIOR, MELHOR.)."

    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_pcv(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula PCV (TIC / 66% TIA).
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    try:
        df = get_df_by_name("INDMANTMANUAL")
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "INDMANTMANUAL", data_inicial, data_final)
        if filtro_coluna and filtro_valor:
            r, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if r is not None: df_filt = r
            
        col_v = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_s = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)
        col_d = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)

        def calcular_soma(sigla, chars):
            mask = pd.Series(False, index=df_filt.index)
            if col_s: mask |= (df_filt[col_s].astype(str).str.strip().str.upper() == sigla)
            if col_d: mask |= (df_filt[col_d].astype(str).str.strip().str.upper().str.slice(0, chars) == sigla)
            return pd.to_numeric(df_filt[mask][col_v], errors='coerce').fillna(0).sum()
        
        val_tic = calcular_soma("TIC", 3)
        val_tia = calcular_soma("TIA", 3)
        
        target = val_tia * 0.66
        if target == 0: return "PCV: 100.00% (Base TIA zero)."
        res = min(val_tic / target, 1.0)
        return f"O PCV é {res:.2%} (TIC: {val_tic} / Meta: {target:.1f}) (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_ioalo(filtro_coluna: Optional[str]=None, filtro_valor: Optional[str]=None, data_inicial: Optional[str]=None, data_final: Optional[str]=None) -> str:
    """Calcula IOALO (CAIEMF / CAIEFO).
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado. Quanto MENOR o valor, PIOR o resultado"""
    try:
        df = get_df_by_name("INDMANTMANUAL")
        df_filt, _ = aplicar_filtro_periodo(df.copy(), "INDMANTMANUAL", data_inicial, data_final)
        if filtro_coluna and filtro_valor:
            r, _ = aplicar_filtro_inteligente(df_filt, filtro_coluna, filtro_valor)
            if r is not None: df_filt = r
            
        col_v = next((c for c in df_filt.columns if "valor" in normalizar_texto(c)), None)
        col_s = next((c for c in df_filt.columns if "simbolo" in normalizar_texto(c)), None)
        col_d = next((c for c in df_filt.columns if "descricao" in normalizar_texto(c)), None)
        
        def calcular_soma(sigla, chars):
            mask = pd.Series(False, index=df_filt.index)
            if col_s: mask |= (df_filt[col_s].astype(str).str.strip().str.upper() == sigla)
            if col_d: mask |= (df_filt[col_d].astype(str).str.strip().str.upper().str.slice(0, chars) == sigla)
            return pd.to_numeric(df_filt[mask][col_v], errors='coerce').fillna(0).sum()
        
        val_aprov = calcular_soma("CAIEMF", 6)
        val_vist = calcular_soma("CAIEFO", 6)
        
        if val_vist == 0: return "IOALO: Indefinido."
        res = val_aprov / val_vist
        return f"O IOALO é {res:.2%} ({val_aprov} / {val_vist}) (Lembre-se: Quanto MAIOR, MELHOR.)."
    except Exception as e: return f"Erro: {e}"

@tool(args_schema=InputCalculoKPI)
def calcular_indoa(filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None, data_inicial: Optional[str] = None, data_final: Optional[str] = None) -> str:
    """
    Calcula o INDOA: Média simples de 6 indicadores (OEMCP, OEMPP, CDTDM, QETT, QETG, IAVLIT).
    Atribui 100 pontos se o indicador atingir a meta ou 0 se falhar.
    IMPORTANTE: Quanto MAIOR o valor, MELHOR o resultado.
    """
    print(f"\n{Fore.MAGENTA}🛠️ TOOL INDOA CHAMADA{Style.RESET_ALL}")
    
    # Lista de indicadores e lógica (True se 'Quanto Menor Melhor', False se 'Quanto Maior Melhor')
    indicadores_indoa = {
        "OEMCP": True, 
        "OEMPP": True, 
        "CDTDM": True, 
        "QETT": False, 
        "QETG": False, 
        "IAVLIT": False
    }
    
    # Mapeamento de funções
    funcs = {
        "OEMCP": calcular_oemcp.func,
        "OEMPP": calcular_oempp.func,
        "CDTDM": calcular_cdtdm.func,
        "QETT": calcular_qett.func,
        "QETG": calcular_qetg.func,
        "IAVLIT": calcular_iavlit.func
    }

    # Determinar a empresa para buscar a meta (padrão 'Leblon' se não informado)
    empresa_meta = "Leblon"
    if filtro_coluna and "empresa" in filtro_coluna.lower():
        empresa_meta = filtro_valor
    
    # Data de referência para meta (usa data_inicial ou hoje)
    dt_ref = data_inicial if data_inicial else datetime.datetime.now().strftime("%Y-%m-%d")

    pontos_totais = 0
    detalhes = []

    for kpi, menor_melhor in indicadores_indoa.items():
        try:
            # 1. Calcula Valor Atual
            res_txt = funcs[kpi](filtro_coluna=filtro_coluna, filtro_valor=filtro_valor, data_inicial=data_inicial, data_final=data_final)
            valor = extrair_valor_numerico(res_txt)
            
            # 2. Busca Meta
            meta_txt = consultar_meta_indicador.func(indicador=kpi, empresa=empresa_meta, data_referencia=dt_ref)
            meta = extrair_valor_numerico(meta_txt)
            
            if valor is not None and meta is not None:
                # 3. Lógica de Pontuação
                atingiu = False
                if menor_melhor:
                    atingiu = (valor <= meta)
                else:
                    atingiu = (valor >= meta)
                
                ponto = 100 if atingiu else 0
                pontos_totais += ponto
                status = "✅" if atingiu else "❌"
                detalhes.append(f"{kpi}: {valor:,.2f} (Meta: {meta:,.2f}) {status}")
            else:
                detalhes.append(f"{kpi}: Dados ou Meta ausentes ⚠️")
                
        except Exception as e:
            detalhes.append(f"{kpi}: Erro no cálculo")

    resultado_final = pontos_totais / 6
    msg_detalhes = "\n   ".join(detalhes)
    
    return (f"O INDOA é {resultado_final:,.2f} pontos.\n"
            f"Composição:\n   {msg_detalhes}\n"
            f"(Cálculo: Soma de pontos / 6. Máximo 100. Quanto MAIOR, MELHOR.)")

# ====================================================
#  NOVA LÓGICA DE COMPARAÇÃO / EVOLUÇÃO
# ====================================================

# Mapa de Funções e Regras de Negócio
CONFIG_KPI = {
    "ICMQ": {"func": calcular_icmq, "melhor": "MIN"},
    "IDF":  {"func": calcular_idf,  "melhor": "MAX"},
    "IMP":  {"func": calcular_imp,  "melhor": "MAX"},
    "OEMCP": {"func": calcular_oemcp, "melhor": "MIN"},
    "OEMPP": {"func": calcular_oempp, "melhor": "MIN"},
    "KMFALHAS": {"func": calcular_km_falhas, "melhor": "MAX"},
    "QETG": {"func": calcular_qetg, "melhor": "MAX"},
    "QETT": {"func": calcular_qett, "melhor": "MAX"},
    "CDTDM": {"func": calcular_cdtdm, "melhor": "MIN"},
    "TO": {"func": calcular_to, "melhor": "MIN"},
    "TOPP": {"func": calcular_topp, "melhor": "MIN"},
    "PREVENTIVAS LIQUIDADAS": {"func": calcular_preventivas_liquidadas, "melhor": "MAX"},
    "IAVLIT": {"func": calcular_iavlit, "melhor": "MAX"},
    "PCV": {"func": calcular_pcv, "melhor": "MAX"},
    "IOALO": {"func": calcular_ioalo, "melhor": "MAX"},
    "INDOA": {"func": calcular_indoa, "melhor": "MAX"},
    # Padrão MIN (Penalidades) para os demais se não especificado
    "CAIEFO": {"func": calcular_caiefo, "melhor": "MIN"},
    "QVA": {"func": calcular_qva, "melhor": "MIN"},
    "QVV": {"func": calcular_qvv, "melhor": "MIN"},
    "TIC": {"func": calcular_tic, "melhor": "MIN"},
    "TIA": {"func": calcular_tia, "melhor": "MIN"},
}

def extrair_valor_numerico(texto: str) -> Optional[float]:
    """Remove R$, %, texto e retorna o float. Ex: 'O ICMQ é R$ 5,50' -> 5.50"""
    # 1. Tenta achar padrão monetário ou decimal brasileiro
    # Procura dígitos, opcionalmente pontos de milhar, virgula decimal, dígitos
    try:
        # Regex captura: (números com ponto/virgula) logo após "é " ou "R$ " ou ":"
        # Simplificação: Pega o primeiro número float válido na string
        padrao = r"([\d\.]+,\d+|[\d,]+\.\d+|\d+)"
        matches = re.findall(padrao, texto)
        
        if not matches: return None
        
        # Pega o primeiro match que parece ser o valor principal
        valor_str = matches[0]
        
        # Limpa pontuação para float python (Troca , por . se for decimal BR)
        # Se tiver ',' e '.' (ex: 1.000,50), remove ponto e troca virgula
        if ',' in valor_str and '.' in valor_str:
            valor_str = valor_str.replace('.', '').replace(',', '.')
        elif ',' in valor_str:
            valor_str = valor_str.replace(',', '.')
            
        return float(valor_str)
    except:
        return None

class InputAnaliseEvolucao(BaseModel):
    indicador: str = Field(..., description="Nome exato do indicador (ex: 'ICMQ', 'IDF', 'KmFalhas')")
    filtro_coluna: Optional[str] = Field(default=None, description="Coluna de filtro (ex: 'onibus')")
    filtro_valor: Optional[str] = Field(default=None, description="Valor do filtro (ex: '1234')")
    data_atual_ini: str = Field(..., description="Data Inicio Periodo Atual (AAAA-MM-DD)")
    data_atual_fim: str = Field(..., description="Data Fim Periodo Atual (AAAA-MM-DD)")
    data_anterior_ini: str = Field(..., description="Data Inicio Periodo Anterior (AAAA-MM-DD)")
    data_anterior_fim: str = Field(..., description="Data Fim Periodo Anterior (AAAA-MM-DD)")

@tool(args_schema=InputAnaliseEvolucao)
def analisar_evolucao_kpi(indicador: str, data_atual_ini: str, data_atual_fim: str, data_anterior_ini: str, data_anterior_fim: str, filtro_coluna: Optional[str] = None, filtro_valor: Optional[str] = None) -> str:
    """
    Compara o valor de um indicador entre dois períodos e diz se MELHOROU ou PIOROU.
    Use esta tool sempre que a pergunta for sobre 'evolução', 'comparação', 'melhoria' ou 'tendência'.
    """
    nome_kpi = indicador.upper().strip()
    config = CONFIG_KPI.get(nome_kpi)
    
    # Tenta achar por aproximação se não achar exato
    if not config:
        for k, v in CONFIG_KPI.items():
            if k in nome_kpi or nome_kpi in k:
                config = v
                nome_kpi = k
                break
    
    if not config:
        return f"Erro: Indicador '{indicador}' não configurado para análise de evolução."
    
    tool_objeto = config["func"]
    
    # Isso evita o erro 'StructuredTool object is not callable'
    funcao_python_real = tool_objeto.func 
    
    direcao_melhor = config["melhor"] # MAX ou MIN
    
    print(f"\n{Fore.MAGENTA}📈 ANALISANDO EVOLUÇÃO [{nome_kpi}]{Style.RESET_ALL}")
    print(f"   Periodo 1 (Anterior): {data_anterior_ini} a {data_anterior_fim}")
    print(f"   Periodo 2 (Atual):    {data_atual_ini} a {data_atual_fim}")

    try:
        # 1. Calcula Período Anterior (Usando .func)
        res_ant_txt = funcao_python_real(filtro_coluna=filtro_coluna, filtro_valor=filtro_valor, data_inicial=data_anterior_ini, data_final=data_anterior_fim)
        val_ant = extrair_valor_numerico(res_ant_txt)
        
        # 2. Calcula Período Atual (Usando .func)
        res_atual_txt = funcao_python_real(filtro_coluna=filtro_coluna, filtro_valor=filtro_valor, data_inicial=data_atual_ini, data_final=data_atual_fim)
        val_atual = extrair_valor_numerico(res_atual_txt)
    except Exception as e:
        return f"Erro interno ao executar cálculo comparativo: {str(e)}"

    # Verifica erros de extração
    if val_ant is None or val_atual is None:
        return (f"Não foi possível comparar numericamente.\n"
                f"Anterior: {res_ant_txt}\nAtual: {res_atual_txt}")

    # 3. Calcula Delta
    delta = val_atual - val_ant
    if val_ant != 0:
        pct = (delta / val_ant) * 100
    else:
        pct = 0.0 # Evita div por zero

    # 4. Determina Veredito (Melhorou/Piorou)
    veredito = "ESTÁVEL"
    
    if abs(pct) < 0.01: # Variação desprezível
        veredito = "ESTÁVEL"
    else:
        if direcao_melhor == "MAX": # Quanto maior, melhor (ex: IDF)
            if delta > 0: veredito = "MELHOROU (Subiu ✅)"
            else: veredito = "PIOROU (Caiu ❌)"
        else: # Quanto menor, melhor (ex: Custo ICMQ)
            if delta < 0: veredito = "MELHOROU (Caiu ✅)"
            else: veredito = "PIOROU (Subiu ❌)"

    # Formatação bonita
    s_val_ant = f"{val_ant:,.2f}"
    s_val_atl = f"{val_atual:,.2f}"
    
    return (f"📊 Análise de Evolução - {nome_kpi}:\n"
            f"• Período Anterior: {s_val_ant}\n"
            f"• Período Atual:    {s_val_atl}\n"
            f"• Variação: {delta:+,.2f} ({delta/val_ant if val_ant else 0:+.1%})\n"
            f"• Resultado: O indicador {veredito}.")

class InputMeta(BaseModel):
    indicador: str = Field(..., description="Sigla do indicador (ex: 'ICMQ', 'IDF')")
    empresa: str = Field(..., description="Nome da empresa (ex: 'Leblon', 'Nobel', 'São Bento')")
    data_referencia: str = Field(..., description="Data para busca da meta no formato AAAA-MM-DD (usar sempre dia 01 do mês)")

@tool(args_schema=InputMeta)
def consultar_meta_indicador(indicador: str, empresa: str, data_referencia: str) -> str:
    """
    Consulta a meta oficial de um indicador para uma empresa e data específica.
    """
    try:
        df_metas = get_df_by_name("METAS_INDICADORES")
        if df_metas is None: return "Tabela de metas não carregada."

        # Normalização para busca
        df_m = df_metas.copy()
        df_m['data_dt'] = pd.to_datetime(df_m['data'], errors='coerce')
        dt_busca = pd.to_datetime(data_referencia)

        # Filtro por Empresa e Data (Mês/Ano)
        mask = (df_m['empresa'].str.lower() == empresa.lower()) & \
               (df_m['data_dt'].dt.month == dt_busca.month) & \
               (df_m['data_dt'].dt.year == dt_busca.year)
        
        resultado = df_m[mask]

        if resultado.empty:
            return f"Meta não encontrada para {empresa} em {data_referencia}."

        col_indicador = encontrar_coluna_flexivel(df_m, indicador.upper())
        if not col_indicador:
            return f"Indicador {indicador} não encontrado na tabela de metas."

        valor_meta = resultado[col_indicador].iloc[0]
        return f"A meta de {indicador} para {empresa} em {dt_busca.strftime('%m/%Y')} é {valor_meta}."
    except Exception as e:
        return f"Erro ao consultar meta: {e}"