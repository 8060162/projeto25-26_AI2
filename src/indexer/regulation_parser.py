import re
from utils.regex_patterns import CAP_PATTERN, ART_PATTERN

class RegulationParser:
    def parse(self, pages):
        """
        Transforma a lista de páginas numa estrutura hierárquica.
        """
        document_tree = {
            "metadata": {"total_pages": len(pages)},
            "preambulo": "",
            "capitulos": {}
        }
        
        current_cap = "PREAMBULO"
        current_art = None
        
        # Estado inicial para capturar o preâmbulo antes do primeiro Capítulo/Artigo
        in_preambulo = True

        for page in pages:
            page_num = page["page"]
            # IMPORTANTE: Não removemos os \n aqui para manter a estrutura
            lines = page["text"].split('\n')
            
            for line in lines:
                clean_line = line.strip()
                if not clean_line: continue

                # 1. Detetar Novo Capítulo
                cap_match = CAP_PATTERN.search(clean_line)
                if cap_match:
                    in_preambulo = False
                    current_cap = clean_line
                    document_tree["capitulos"][current_cap] = {
                        "titulo_capitulo": clean_line,
                        "artigos": {}
                    }
                    current_art = None
                    continue

                # 2. Detetar Novo Artigo
                art_match = ART_PATTERN.search(clean_line)
                if art_match:
                    in_preambulo = False
                    art_id = f"ART_{art_match.group(1)}"
                    current_art = art_id
                    
                    # Inicializa o artigo na estrutura do capítulo atual
                    if current_cap not in document_tree["capitulos"]:
                        document_tree["capitulos"][current_cap] = {"artigos": {}}
                    
                    document_tree["capitulos"][current_cap]["artigos"][current_art] = {
                        "header": clean_line,
                        "conteudo": "",
                        "page": page_num
                    }
                    continue

                # 3. Distribuição do Texto
                if in_preambulo:
                    document_tree["preambulo"] += clean_line + " "
                elif current_art:
                    # Adiciona texto ao artigo atual
                    document_tree["capitulos"][current_cap]["artigos"][current_art]["conteudo"] += clean_line + " "
                elif current_cap != "PREAMBULO":
                    # Texto que pertence ao capítulo mas ainda não a um artigo (ex: títulos de capítulo)
                    document_tree["capitulos"][current_cap]["titulo_capitulo"] += " " + clean_line

        return document_tree