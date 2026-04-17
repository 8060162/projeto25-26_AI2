# analysis/__init__.py
# Pacote de scripts de análise e diagnóstico dos embeddings ChromaDB.
#
# Scripts disponíveis:
#   visualize.py  — visualização interactiva no Spotlight (Renumics)
#   evaluate.py   — métricas de coesão intra-cluster e separação inter-cluster
#
# Dependência partilhada:
#   _chromadb_loader.py — acesso único ao ChromaDB (não importar directamente)
#
# Uso:
#   python -m src.analysis.visualize
#   python -m src.analysis.evaluate