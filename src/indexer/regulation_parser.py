import re
from utils.regex_patterns import ART_PATTERN, CAP_PATTERN, PAGE_NOISE
from .metadata_extractor import MetadataExtractor

class RegulationParser:
    def parse(self, pages_dict: dict, filename: str) -> dict:
        # 1. Estrutura base do JSON
        document_tree = {
            "document_info": {},
            "preambulo": "",
            "estrutura": {}
        }
        
        # Estados para saber onde inserir o texto
        current_cap_id = "CAP_0"
        current_art_id = None
        preambulo_lines = []

        # Inicializamos um capítulo padrão caso o PDF comece logo com artigos
        document_tree["estrutura"][current_cap_id] = {
            "titulo": "Disposições Iniciais",
            "artigos": {}
        }

        # Transformamos o dicionário de páginas numa lista sequencial de elementos
        elements = []
        for p in sorted(pages_dict.keys()):
            for el in pages_dict[p]:
                text = el["text"].strip()
                # Filtro de ruído: ignora índices (......) e lixo de rodapé
                if not text or "...." in text or PAGE_NOISE.search(text):
                    continue
                elements.append({"text": text, "page": p})

        for i, el in enumerate(elements):
            text = el["text"]
            
            # --- DETECÇÃO DE CAPÍTULO ---
            cap_match = CAP_PATTERN.match(text)
            if cap_match:
                cap_label = cap_match.group(1).upper()
                current_cap_id = f"CAP_{cap_label}"
                
                document_tree["estrutura"][current_cap_id] = {
                    "titulo": text, # Ex: "CAPÍTULO I DISPOSIÇÕES GERAIS"
                    "artigos": {}
                }
                current_art_id = None
                continue

            # --- DETECÇÃO DE ARTIGO ---
            art_match = ART_PATTERN.match(text)
            if art_match:
                art_num = art_match.group(1)
                current_art_id = f"ART_{art_num}"
                
                # Tenta extrair o título (mesma linha ou espreita a próxima)
                titulo_art = text[art_match.end():].strip(" .º°o-")
                if not titulo_art and (i + 1) < len(elements):
                    proxima = elements[i+1]["text"]
                    if len(proxima) < 100 and not ART_PATTERN.match(proxima):
                        titulo_art = proxima

                document_tree["estrutura"][current_cap_id]["artigos"][current_art_id] = {
                    "titulo": titulo_art if titulo_art else "Sem Título",
                    "conteudo": "",
                    "pagina": el["page"]
                }
                continue

            # --- PREENCHIMENTO DE CONTEÚDO ---
            if current_art_id:
                art_ref = document_tree["estrutura"][current_cap_id]["artigos"][current_art_id]
                # Evita repetir o título no início do conteúdo
                if text != art_ref["titulo"]:
                    espaco = " " if art_ref["conteudo"] else ""
                    art_ref["conteudo"] += espaco + text
            else:
                # Se não temos artigo ativo, o texto pertence ao preâmbulo
                preambulo_lines.append(text)

        # 2. Finalização: Limpeza e Metadados
        document_tree["preambulo"] = " ".join(preambulo_lines[:10]) # Primeiras linhas
        document_tree["document_info"] = MetadataExtractor.extract(document_tree["preambulo"], filename)

        # Se o CAP_0 não foi usado, removemos para limpar o JSON
        if not document_tree["estrutura"]["CAP_0"]["artigos"]:
            del document_tree["estrutura"]["CAP_0"]

        return document_tree