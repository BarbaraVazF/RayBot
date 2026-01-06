import os
import threading
from django.conf import settings 
from utils.pandas_agent import criar_agente_multiplas_tabelas 
from .rag_engine import inicializar_rag, recuperar_contexto 
from .data_loader import carregar_dataframes 

class LeblonService:
    _instance = None
    _lock = threading.Lock() 

    def __init__(self):
        if not hasattr(self, 'initialized'):
             self._initialize()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("🚀 Inicializando Serviço Leblon...")
        
        self.path_dados = os.path.join(settings.BASE_DIR, 'utils', 'base_leblon')
        
        self.lista_dfs = carregar_dataframes(self.path_dados)
        
        if self.lista_dfs:
            self.agente = criar_agente_multiplas_tabelas(self.lista_dfs)
        else:
            self.agente = None
            print("❌ ALERTA: Nenhum dado carregado. O agente não funcionará.")

        self.vectordb = inicializar_rag() 
        self.initialized = True

    def responder(self, pergunta, historico):
        if not self.agente:
            return "Erro: As bases de dados não foram carregadas corretamente."

        contexto = recuperar_contexto(self.vectordb, pergunta)
        
        colunas_texto = "\n".join(
            f"📍 TABELA {i+1} ({df.attrs.get('name', 'Dados')}) → {', '.join(df.columns)}"
            for i, df in enumerate(self.lista_dfs)
        )
        
        historico_texto = "\n".join(f"- {h}" for h in historico[-5:]) 

        prompt = f"""
        Você é um analista de dados.
        CONTEXTO DA DOCUMENTAÇÃO (RAG):
        {contexto}

        HISTÓRICO:
        {historico_texto}

        TABELAS DISPONÍVEIS:
        {colunas_texto}

        PERGUNTA:
        {pergunta}

        Responda em português, usando apenas os dados fornecidos.
        """

        try:
            resposta = self.agente.invoke({"input": prompt})
            return resposta.get("output", "Não foi possível gerar resposta.")
        except Exception as e:
            return f"Erro no processamento: {str(e)}"