import json
import os

from indexer.structure_pipeline import StructurePipeline


pdf_path = "data/raw/Despacho P.PORTO-P- 043-2025_Regulamento de Propinas.pdf"

pipeline = StructurePipeline()

result = pipeline.run(pdf_path)

os.makedirs("output", exist_ok=True)

with open("data/processed/estrutura.json", "w", encoding="utf-8") as f:

    json.dump(result, f, indent=4, ensure_ascii=False)

print("Parsing concluído")