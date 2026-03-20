"""
prompt_builder.py
-----------------
Responsabilidade única: construir os prompts para o pipeline RAG.

O template do prompt sistema é carregado uma única vez no momento de
importação do módulo (lazy init com cache), eliminando I/O repetido
a cada pergunta e falhando de forma explícita na inicialização caso
o ficheiro esteja ausente — não em runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from retriever.models import ArtigoContexto

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=1)
def _carregar_template_sistema() -> str:
    """
    Carrega o template do prompt sistema a partir do ficheiro externo.

    Decorado com @lru_cache(maxsize=1): o ficheiro é lido uma única vez
    e o resultado fica em memória para todas as chamadas subsequentes.

    Raises:
        FileNotFoundError: se o ficheiro sistema.txt não existir.
                           O erro ocorre na primeira chamada (tipicamente
                           no arranque do pipeline), não durante uma
                           conversa activa.
    """
    caminho = _PROMPTS_DIR / "sistema.txt"
    try:
        return caminho.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Template do prompt sistema não encontrado em: {caminho}. "
            f"Verifica se o ficheiro 'prompts/sistema.txt' existe."
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
    """
    Constrói o par (prompt_sistema, prompt_utilizador) para o modelo.

    Args:
        pergunta: Questão do utilizador.
        artigos:  Artigos recuperados pelo retriever.

    Returns:
        Tuplo (prompt_sistema, prompt_utilizador).

    Raises:
        FileNotFoundError: propagada de _carregar_template_sistema()
                           se o ficheiro de template estiver ausente.
    """
    contexto       = formatar_contexto(artigos)
    prompt_sistema = _carregar_template_sistema()

    prompt_utilizador = (
        f"Aqui estão os regulamentos que deves consultar:\n{contexto}\n\n"
        f"---\n"
        f"Pergunta do Aluno: {pergunta}\n\n"
        f"Responde de forma útil e amigável:"
    )

    return prompt_sistema, prompt_utilizador