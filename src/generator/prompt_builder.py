"""
prompt_builder.py
-----------------
Responsabilidade única: construir os prompts para o pipeline RAG.
O conteúdo editorial do prompt sistema é carregado de um ficheiro externo,
separando lógica de construção de conteúdo configurável.
"""

from pathlib import Path
from retriever.models import ArtigoContexto

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _carregar_template_sistema() -> str:
    """Carrega o template do prompt sistema a partir do ficheiro externo."""
    caminho = _PROMPTS_DIR / "sistema.txt"
    try:
        return caminho.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Template do prompt sistema não encontrado em: {caminho}"
        )


def formatar_contexto(artigos: list[ArtigoContexto]) -> str:
    """Formata os artigos para o prompt, preservando proveniência e hierarquia."""
    blocos = []
    for art in artigos:
        bloco = (
            f"### FONTE: {art.source} | {art.capitulo_titulo}\n"
            f"### {art.artigo_id} - {art.artigo_titulo}\n"
            f"{art.conteudo}"
        )
        blocos.append(bloco)
    return "\n\n---\n\n".join(blocos)


def construir_prompt(pergunta: str, artigos: list[ArtigoContexto]) -> tuple[str, str]:
    """Constrói o par (prompt_sistema, prompt_utilizador) para o modelo.

    Args:
        pergunta: Questão do utilizador.
        artigos:  Artigos recuperados pelo retriever.

    Returns:
        Tuplo (prompt_sistema, prompt_utilizador).
    """
    contexto = formatar_contexto(artigos)
    prompt_sistema = _carregar_template_sistema()

    prompt_utilizador = (
        f"Aqui estão os regulamentos que deves consultar:\n{contexto}\n\n"
        f"---\n"
        f"Pergunta do Aluno: {pergunta}\n\n"
        f"Responde de forma útil e amigável:"
    )

    return prompt_sistema, prompt_utilizador