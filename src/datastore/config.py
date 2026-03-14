import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chromaDB")
COLLECTION_NAME = "artigos_legislativos"
N_RESULTS = 3