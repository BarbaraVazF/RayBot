import pandas as pd
import os
import glob

def xlsx_to_csv(path_xlsx):
    """
    Converte um único XLSX em CSV no mesmo diretório e apaga o XLSX original.
    """
    path_csv = os.path.splitext(path_xlsx)[0] + ".csv"

    try:
        df = pd.read_excel(path_xlsx)
        df.to_csv(path_csv, index=False, encoding="utf-8")
        print(f"Convertido: {path_xlsx} → {path_csv}")

        os.remove(path_xlsx)
        print(f"Arquivo removido: {path_xlsx}")

    except Exception as e:
        print(f"Erro ao converter {path_xlsx}: {e}")


def converter_todos_xlsx(diretorio="base_caligares/"):
    """
    Converte todos os XLSX da pasta base_caligares em CSV e remove os originais.
    """
    arquivos = glob.glob(os.path.join(diretorio, "*.xlsx"))

    if not arquivos:
        print("Nenhum XLSX de entrada encontrado.")
        return

    print(f"{len(arquivos)} XLSX encontrados para conversão (entrada).")

    for arq in arquivos:
        xlsx_to_csv(arq)