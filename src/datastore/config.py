import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH         = os.path.join(BASE_DIR, "data", "chromaDB")
JSON_FOLDER     = os.path.join(BASE_DIR, "data", "processed")
COLLECTION_NAME = "artigos_legislativos"

# Chunking
CHUNK_SIZE  = 550
OVERLAP     = 80