from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def construir_texto(filename: str, cap_titulo: str, art_id: str, art_titulo: str, conteudo: str) -> str:
    """Formata o texto final que será indexado no ChromaDB."""
    return (
        f"FICHEIRO: {filename}\n"
        f"CAPÍTULO: {cap_titulo}\n"
        f"ARTIGO: {art_id} - {art_titulo}\n"
        f"CONTEÚDO: {conteudo}"
    )


def dividir_em_chunks(texto: str, conteudo: str) -> list[str]:
    """Devolve lista de chunks. Se o conteúdo couber num único chunk, devolve lista com um elemento."""
    if len(conteudo) > CHUNK_SIZE:
        return text_splitter.split_text(texto)
    return [texto]