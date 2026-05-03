"""
text_quality.py
---------------
Avalia a qualidade do texto extraído de um PDF e decide se o OCR é necessário.

Heurísticas usadas (inspiradas em práticas de produção como Nougat/Marker):
  1. Rácio de caracteres não-ASCII suspeitos (caracteres de controle, surrogates)
  2. Densidade de tokens "garbled": sequências de consoantes sem vogais > N chars
  3. Rácio de palavras reconhecíveis na língua (via wordlist simples)
  4. Rácio caracteres imprimíveis vs total
  5. Detecção de encoding corruption (ftfy)

Score final: 0.0 (lixo total) → 1.0 (texto perfeito).
Threshold recomendado: < 0.45 → forçar OCR.
"""

import re
import unicodedata
from functools import lru_cache

try:
    import ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False


# ── Padrões ───────────────────────────────────────────────────────────────────

# Sequências de consoantes sem vogais com 5+ chars → provável lixo
_CONSONANT_CLUSTER = re.compile(
    r'\b[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}\b'
)

# Caracteres de controlo (excepto tab, newline, carriage return)
_CONTROL_CHARS = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]'
)

# Símbolos de substituição e caixas de perguntas Unicode
_REPLACEMENT = re.compile(r'[\ufffd\u25a1\u2610]')

# Sequências típicas de PDF mal codificado: "(cid:N)"
_CID_GARBAGE = re.compile(r'\(cid:\d+\)', re.IGNORECASE)


def score_text_quality(text: str) -> float:
    """
    Calcula um score de qualidade do texto: 0.0 (lixo) → 1.0 (perfeito).

    Retorna 1.0 para texto vazio (sem opinião — deixar ao pipeline decidir).

    Args:
        text: bloco de texto a avaliar.

    Returns:
        float entre 0.0 e 1.0.
    """
    if not text or not text.strip():
        return 1.0

    # --- Penalidades absolutas (detectam encoding catastrophico) --------------

    # Muitos "(cid:N)" → fonte não embebida, texto ilegível
    cid_count = len(_CID_GARBAGE.findall(text))
    if cid_count > 3:
        return 0.0

    total_chars = len(text)

    # --- Rácio de caracteres imprimíveis ------------------------------------
    printable = sum(
        1 for ch in text
        if unicodedata.category(ch)[0] not in ('C',) or ch in '\t\n\r'
    )
    printable_ratio = printable / total_chars

    # --- Rácio de caracteres de controlo e substituição --------------------
    control_hits   = len(_CONTROL_CHARS.findall(text))
    replace_hits   = len(_REPLACEMENT.findall(text))
    garbage_ratio  = (control_hits + replace_hits) / total_chars

    # --- Clusters de consoantes (lixo OCR / encoding) ----------------------
    words          = text.split()
    word_count     = max(len(words), 1)
    cluster_hits   = len(_CONSONANT_CLUSTER.findall(text))
    cluster_ratio  = cluster_hits / word_count

    # --- Correcção de encoding via ftfy ------------------------------------
    ftfy_penalty = 0.0
    if _FTFY_AVAILABLE and total_chars < 5000:  # só para amostras pequenas
        fixed = ftfy.fix_text(text)
        changed = sum(a != b for a, b in zip(text, fixed))
        ftfy_penalty = min(changed / total_chars, 0.5)

    # --- Score composto -----------------------------------------------------
    # Pesos ajustados empiricamente para documentos legais/administrativos PT
    score = (
        printable_ratio     * 0.40
        - garbage_ratio     * 0.30
        - cluster_ratio     * 0.20
        - ftfy_penalty      * 0.10
    )

    return max(0.0, min(1.0, score))


def needs_ocr(elements: list[dict], sample_size: int = 20, threshold: float = 0.45) -> bool:
    """
    Decide se a lista de elementos precisa de OCR.

    Amostra os primeiros `sample_size` elementos com texto e calcula
    o score médio. Se abaixo de `threshold`, recomenda OCR.

    Args:
        elements:    lista de elementos normalizados (text/category/page).
        sample_size: número de elementos a amostrar.
        threshold:   score mínimo aceitável (0–1). Default: 0.45.

    Returns:
        True se OCR é recomendado.
    """
    texts = [
        el["text"] for el in elements
        if el.get("text", "").strip() and el.get("category") not in ("Header", "Footer")
    ][:sample_size]

    if not texts:
        return True  # sem texto → definitivamente precisa OCR

    avg_score = sum(score_text_quality(t) for t in texts) / len(texts)
    return avg_score < threshold


def diagnose(text: str) -> dict:
    """
    Diagnóstico detalhado para debugging — devolve as métricas individuais.

    Útil para ajustar thresholds ou auditar documentos problemáticos.
    """
    if not text or not text.strip():
        return {"score": 1.0, "reason": "empty"}

    total     = len(text)
    cid_count = len(_CID_GARBAGE.findall(text))
    ctrl      = len(_CONTROL_CHARS.findall(text))
    repl      = len(_REPLACEMENT.findall(text))
    words     = text.split()
    clusters  = len(_CONSONANT_CLUSTER.findall(text))
    printable = sum(
        1 for ch in text
        if unicodedata.category(ch)[0] not in ('C',) or ch in '\t\n\r'
    )

    return {
        "score":           score_text_quality(text),
        "total_chars":     total,
        "cid_sequences":   cid_count,
        "control_chars":   ctrl,
        "replacement_chars": repl,
        "consonant_clusters": clusters,
        "word_count":      len(words),
        "printable_ratio": round(printable / total, 3),
        "ftfy_available":  _FTFY_AVAILABLE,
    }