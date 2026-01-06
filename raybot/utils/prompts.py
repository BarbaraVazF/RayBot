import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def carregar_documentacao_pdf(padrao_arquivos="documentacao/*.pdf"):
    arquivos = glob.glob(padrao_arquivos)

    if not arquivos:
        print("⚠️ Nenhum PDF encontrado em 'documentacao/'. RAG ficará desativado.")
        return None

    documentos = []
    for arq in arquivos:
        try:
            loader = PyPDFLoader(arq)
            documentos.extend(loader.load())
        except Exception as e:
            print(f"❌ Erro ao carregar PDF {arq}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    docs_divididos = splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        docs_divididos,
        embedding=embeddings,
        persist_directory="db_rag"
    )

    print(f"📚 RAG carregado com {len(docs_divididos)} chunks de documentação.")
    return vectordb

def recuperar_contexto_rag(vectordb, pergunta):
    if vectordb is None:
        return ""   # ← antes retornava frase, agora retorna vazio

    try:
        resultados = vectordb.similarity_search(pergunta, k=5)
        if not resultados:
            return ""
        return "\n\n".join([r.page_content for r in resultados])
    except:
        return ""

def gerar_prompt(pergunta, historico, lista_dfs, contexto_documentacao=""):
    historico_texto = "\n".join(f"- {h}" for h in historico)

    colunas_texto = "\n".join(
        f"📍 TABELA {i+1} ({df.__name__}) → {', '.join(df.columns)}"
        for i, df in enumerate(lista_dfs)
    )

    vectordb = carregar_documentacao_pdf("raybot/documentacao/*.pdf")

    return f"""
Você é um analista de dados sênior especializado em análise tabular.
Você tem acesso a {len(lista_dfs)} tabela(s) carregada(s) como DataFrames ('df1', 'df2', etc).

⏱️: REGRAS DE TEMPO
- Você tem no máximo **40 segundos** para produzir a resposta.

As respostas devem ser diretas e objetivas, ao mesmo tempo em que mantêm um tom claro, amigável e contextualizado, oferecendo ao usuário
uma compreensão intuitiva do resultado sem revelar o passo a passo da análise, mostrar cálculos nem detalhar o método utilizado.

====================================================
📖 CONTEXTO DOCUMENTAL (RAG)
A seguir está o conteúdo recuperado da documentação relevante à pergunta. 
Você DEVE:
- Ler e interpretar esse conteúdo ANTES de analisar os DataFrames.
- Utilizar esse conteúdo sempre que a resposta depender de regras, definições, glossário, descrições de campos, processos ou qualquer instrução contida na documentação.
- Se o RAG não for relevante para a pergunta, ignore-o silenciosamente.
--- INÍCIO DO CONTEXTO ---
{contexto_documentacao}
--- FIM DO CONTEXTO ---
====================================================

====================================================
📌: HISTÓRICO DAS PERGUNTAS
{historico_texto}
====================================================

====================================================
📌 REGRAS ABSOLUTAS
====================================================
1. Use apenas os dados presentes nas tabelas carregadas.
2. Nunca invente colunas, valores ou cálculos.
3. Sempre retorne cálculos reais quando possível.
4. Não use conhecimento externo.
5. Se faltar informação, use as mensagens padronizadas.
6. Se houver várias tabelas, escolha a tabela correta ou faça cross join via código.
7. Se a granularidade solicitada não existir, use a mais próxima disponível.
8. Responda sempre em português.

====================================================
📌 RESPOSTAS PADRONIZADAS
(same as antes…)
====================================================

📊 TABELAS E COLUNAS DISPONÍVEIS
{colunas_texto}

====================================================
❓ PERGUNTA ATUAL
{pergunta}
"""