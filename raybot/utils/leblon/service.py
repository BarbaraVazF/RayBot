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
            print("❌ ALERTA: Nenhum dado carregado.")

        self.vectordb = inicializar_rag() 
        self.initialized = True

    def responder(self, pergunta, historico):
        """
        Gera o prompt com as novas regras e invoca o agente.
        """
        if not self.agente:
            return "Erro: As bases de dados não foram carregadas corretamente."

        lista_dfs = self.lista_dfs 

        contexto_documentacao = recuperar_contexto(self.vectordb, pergunta)
        
        info_colunas = []
        for i, df in enumerate(lista_dfs):
            nome_arq = df.attrs.get('name', f"Tabela {i+1}")
            info_colunas.append(f"📍 TABELA {i+1} (Nome: {nome_arq}) - {len(df)} linhas:\n   Colunas: {', '.join(df.columns)}")
        
        texto_dados_disponiveis = "\n\n".join(info_colunas)

        prompt = f"""
        Você é um analista de dados sênior especializado em análise tabular.
        Você tem acesso a {len(lista_dfs)} DataFrames carregados separadamente: df1, df2, etc.
        Esses DataFrames não são unidos automaticamente.

        USO DO RAG (OBRIGATÓRIO E SIMPLIFICADO)
        O RAG contém definições oficiais e estruturais, organizadas em três abas:
            - Aba 1 - Descrição das Planilhas: explica o propósito de cada tabela.
            - Aba 2 - Descrição das Colunas: detalha o significado e uso de cada coluna.
            - Aba 3 - Cálculo dos Indicadores: define indicadores oficiais, suas fórmulas e quais colunas/tabelas devem ser utilizadas no cálculo.
        PROCESSO OBRIGATÓRIO DE INTERPRETAÇÃO:
        1. Mapear o segmento da pergunta (palavras-chave conceituais)
        2. Identificar se a pergunta envolve um indicador
            - Caso envolva, localizar o indicador na aba “Cálculo dos Indicadores”
            - Identificar as tabelas e colunas envolvidas no cálculo conforme definido no RAG
            - O cálculo deve respeitar exatamente a fórmula e a lógica descritas
        3. Caso não seja um indicador, a partir do segmento identificado, encontrar qual tabela mais tem similaridade com ele a partir da descrição e das suas colunas
        4. Selecionar a(s) tabela(s) e encontrar quais colunas serão utilizadas
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
            resposta = self.agente.invoke({"input": prompt})
            return resposta.get("output", "Não foi possível gerar resposta.")
        except Exception as e:
            return f"Erro no processamento: {str(e)}"