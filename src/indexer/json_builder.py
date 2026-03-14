class RegulationJSONBuilder:

    def build(self, doc_id, metadata, estrutura):

        return {
            doc_id: {
                "doc_id": doc_id,
                "metadata": metadata,
                "estrutura": estrutura
            }
        }