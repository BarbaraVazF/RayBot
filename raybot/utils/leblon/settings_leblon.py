import os
from django.conf import settings

BASE_DIR = settings.BASE_DIR

PASTA_BASE_DADOS = os.path.join(BASE_DIR, "utils", "base_leblon")
PASTA_DOCUMENTACAO = os.path.join(BASE_DIR, "utils", "documentacao_leblon")
PASTA_DB_RAG = os.path.join(BASE_DIR, "db_rag")

MODELO_LLM = "gpt-4o-mini"