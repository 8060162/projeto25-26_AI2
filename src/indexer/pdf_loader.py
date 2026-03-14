from langchain_community.document_loaders import UnstructuredPDFLoader

class PDFLoader:
    def load(self, file_path: str):
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="hi_res",
            languages=["por"]
        )
        
        docs = loader.load()
        pages_dict = {}
        
        for doc in docs:
            # CORREÇÃO: Garantir que extraímos o número da página com segurança
            # O Unstructured às vezes coloca o page_number dentro de um campo 'metadata'
            p_num = doc.metadata.get("page_number")
            
            # Se o p_num vier como None ou não existir, usamos 1 como padrão
            if p_num is None:
                p_num = 1
                
            if p_num not in pages_dict:
                pages_dict[p_num] = []
            
            # Guardamos apenas o essencial para o parser
            pages_dict[p_num].append({
                "text": str(doc.page_content).strip(),
                "category": doc.metadata.get("category", "Uncategorized")
            })
            
        return pages_dict