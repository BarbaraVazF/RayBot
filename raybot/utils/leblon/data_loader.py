import os
import glob
import pandas as pd

def carregar_dataframes(caminho_base):
    padrao = os.path.join(caminho_base, "*.csv")
    arquivos = glob.glob(padrao)
    
    dfs = []
    
    if not arquivos:
        print(f"⚠️ Nenhum CSV encontrado em: {caminho_base}")
        return []

    for arquivo in arquivos:
        try:
            try:
                df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8')
            except Exception:
                df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin1')

            df.columns = df.columns.str.lower().str.strip()
            nome_arquivo = os.path.basename(arquivo)
            df.attrs['name'] = nome_arquivo 
            
            dfs.append(df)
            print(f"✅ Carregado: {nome_arquivo} ({len(df)} linhas)")
            
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo}: {e}")

    return dfs