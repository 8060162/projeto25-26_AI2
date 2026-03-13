import re
from utils.regex_patterns import CAP_PATTERN, ART_PATTERN
from .metadata_extractor import MetadataExtractor

# Artigos reais: "Artigo Nº" isolado no início da linha
ART_HEADING_PATTERN = re.compile(
    r"^Artigo\s+(\d+)[.ºo]?\s*$",
    re.IGNORECASE
)

# Capítulo normalizado: extrai o número romano para usar como chave (ex: "CAP_I")
CAP_NUM_PATTERN = re.compile(
    r"CAP[IÍ]TULO\s+([IVXLCDM]+)",
    re.IGNORECASE
)

# Cabeçalhos de página e rodapés
PAGE_HEADER_PATTERNS = [
    r"^REGULAMENTO\s+P\.PORTO/P-\s*\d+/\d+$",
    r"^PROPINAS DO INSTITUTO POLIT[EÉ]CNICO DO PORTO$",
    r"^\d+\s*\|\s*\d+$",
    r"^Página \d+",
    r"^\d+$",
]

# Notas de rodapé: "1 Acessível em ..."
FOOTNOTE_PATTERN = re.compile(r"^\d+\s+[A-Za-záàâãéèêíóôõúçÁÀÂÃÉÈÊÍÓÔÕÚÇ]")

# URLs soltos (resíduo de notas de rodapé que passam o filtro anterior)
URL_PATTERN = re.compile(r"^https?://\S+$")


class RegulationParser:
    def parse(self, pages, filename):
        document_tree = {
            "document_info": {},
            "preambulo": "",
            "estrutura": {}
        }

        current_cap_key = None   # ex: "CAP_I"
        current_art_key = None   # ex: "ART_1"
        pending_art_titulo = []  # acumula linhas do título multi-linha
        in_preambulo = True
        in_index = False
        preambulo_fechado = False  # True assim que entra num capítulo real

        for page in pages:
            page_num = page["page"]
            lines = page["text"].split('\n')

            for line in lines:
                clean_line = line.strip()

                if not clean_line:
                    continue

                # 1. FILTRO DE RUÍDO
                if self._is_noise(clean_line):
                    continue

                # 2. GESTÃO DO ÍNDICE
                # Activa ao encontrar "ÍNDICE" sozinho; desactiva no 1º capítulo real
                if "ÍNDICE" in clean_line.upper() and len(clean_line) < 20:
                    in_index = True
                    continue

                if in_index:
                    if CAP_PATTERN.search(clean_line) and ".........." not in clean_line:
                        in_index = False
                        # não damos continue — esta linha é o 1º capítulo real
                    else:
                        continue

                # 3. ACTIVAÇÃO DO PREÂMBULO
                # Aguarda "Considerando" para começar a acumular conteúdo útil
                if in_preambulo and not preambulo_fechado:
                    if "CONSIDERANDO" not in clean_line.upper() and not CAP_PATTERN.search(clean_line):
                        continue

                # 4. DETECÇÃO DE CAPÍTULO
                if CAP_PATTERN.search(clean_line) and ".........." not in clean_line:
                    in_preambulo = False
                    preambulo_fechado = True
                    current_art_key = None
                    pending_art_titulo = []

                    cap_num_match = CAP_NUM_PATTERN.search(clean_line)
                    current_cap_key = f"CAP_{cap_num_match.group(1)}" if cap_num_match else clean_line

                    if current_cap_key not in document_tree["estrutura"]:
                        document_tree["estrutura"][current_cap_key] = {
                            "titulo": "",   # preenchido pela linha de intro (ex: "DISPOSIÇÕES GERAIS")
                            "artigos": {}
                        }
                    continue

                # 5. DETECÇÃO DE ARTIGO
                art_match = ART_HEADING_PATTERN.match(clean_line)
                if art_match:
                    # Fecha o título pendente do artigo anterior antes de abrir o novo
                    self._flush_pending_titulo(
                        document_tree, current_cap_key, current_art_key, pending_art_titulo
                    )
                    pending_art_titulo = []

                    in_preambulo = False
                    preambulo_fechado = True
                    current_art_key = f"ART_{art_match.group(1)}"

                    if current_cap_key is None:
                        current_cap_key = "CAP_GERAL"
                        document_tree["estrutura"][current_cap_key] = {"titulo": "", "artigos": {}}

                    document_tree["estrutura"][current_cap_key]["artigos"][current_art_key] = {
                        "titulo": "",
                        "conteudo": "",
                        "pagina": page_num
                    }
                    continue

                # 6. TÍTULO DO ARTIGO (multi-linha)
                # Acumula linhas em maiúsculas imediatamente após o header do artigo,
                # enquanto o conteúdo ainda está vazio — lida com títulos que quebram linha.
                if (current_art_key
                        and current_cap_key in document_tree["estrutura"]
                        and current_art_key in document_tree["estrutura"][current_cap_key]["artigos"]):
                    art_obj = document_tree["estrutura"][current_cap_key]["artigos"][current_art_key]
                    if not art_obj["conteudo"] and self._looks_like_article_title(clean_line):
                        pending_art_titulo.append(clean_line)
                        continue
                    elif pending_art_titulo:
                        # Primeira linha que não é título: fecha o título acumulado
                        self._flush_pending_titulo(
                            document_tree, current_cap_key, current_art_key, pending_art_titulo
                        )
                        pending_art_titulo = []

                # 7. ACUMULAÇÃO DE CONTEÚDO
                if in_preambulo:
                    document_tree["preambulo"] += clean_line + " "
                elif current_art_key and current_cap_key in document_tree["estrutura"]:
                    document_tree["estrutura"][current_cap_key]["artigos"][current_art_key]["conteudo"] += clean_line + " "
                elif current_cap_key and current_cap_key in document_tree["estrutura"]:
                    # Linha imediatamente após o capítulo e antes do 1º artigo = titulo do capítulo
                    cap_obj = document_tree["estrutura"][current_cap_key]
                    if not cap_obj["titulo"] and self._looks_like_article_title(clean_line):
                        cap_obj["titulo"] = clean_line
                    # intro extra (raro): ignorado intencionalmente para não poluir o chunk

        # Fecha título pendente do último artigo
        self._flush_pending_titulo(
            document_tree, current_cap_key, current_art_key, pending_art_titulo
        )

        # Limpeza final de espaços
        document_tree["preambulo"] = document_tree["preambulo"].strip()
        for cap in document_tree["estrutura"].values():
            for art in cap["artigos"].values():
                art["conteudo"] = art["conteudo"].strip()

        # Remove capítulos sem título (artefactos residuais)
        document_tree["estrutura"] = {
            k: v for k, v in document_tree["estrutura"].items() if v["titulo"]
        }

        document_tree["document_info"] = MetadataExtractor.extract(
            document_tree["preambulo"], filename
        )
        return document_tree

    @staticmethod
    def _flush_pending_titulo(tree, cap_key, art_key, pending: list):
        """Junta as linhas de título acumuladas e guarda no artigo."""
        if not pending or not cap_key or not art_key:
            return
        cap = tree["estrutura"].get(cap_key)
        if not cap:
            return
        art = cap["artigos"].get(art_key)
        if art:
            art["titulo"] = " ".join(pending).strip()

    def _is_noise(self, line: str) -> bool:
        for pattern in PAGE_HEADER_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        if FOOTNOTE_PATTERN.match(line) and len(line) < 120:
            return True
        if URL_PATTERN.match(line):
            return True
        return False

    @staticmethod
    def _looks_like_article_title(line: str) -> bool:
        """
        Heurística: título de artigo/capítulo é maioritariamente em maiúsculas,
        curto, sem ponto final, não começa com dígito.
        """
        if len(line) > 80:
            return False
        if line[0].isdigit():
            return False
        if line.endswith('.'):
            return False
        upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
        return upper_ratio > 0.6