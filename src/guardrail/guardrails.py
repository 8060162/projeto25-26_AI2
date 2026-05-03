import logging
import re
from typing import List, Tuple
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

class Guardrails:
    """
    Módulo Independente de Segurança (Guardrails).
    Responsabilidade: Validar integridade de entradas e saídas.
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        # Modelo carregado apenas uma vez
        log.info(f"A inicializar Guardrails com o modelo: {model_name}")
        self._reranker = CrossEncoder(model_name)
        self._pii_nif = re.compile(r'\b\d{9}\b')

    def validate_input(self, query: str) -> Tuple[bool, str]:
        """
        Analisa a pergunta do utilizador.
        Returns: (is_safe, processed_query)
        """
        if len(query.strip()) < 3:
            return False, "Pergunta demasiado curta."
        
        # Sanitização de PII (NIF)
        clean_query = self._pii_nif.sub("[NIF_REDACTED]", query)
        
        # Filtro de segurança básico (Jailbreak/Injection)
        if any(term in query.lower() for term in ["ignore", "instruções do sistema"]):
            return False, "Tentativa de manipulação de prompt detetada."

        return True, clean_query

    def validate_groundedness(self, answer: str, context: str, threshold: float = 0.5) -> bool:
        """
        Verifica se a resposta gerada tem suporte no contexto (Anti-Alucinação).
        Usa o Cross-Encoder para calcular a fidelidade semântica.
        """
        if not answer or not context:
            return False
            
        score = self._reranker.predict([context, answer])
        log.info(f"Guardrail Score (Groundedness): {score:.4f}")
        
        return score >= threshold

    def validate_relevance(self, query: str, context: str, threshold: float = 0.3) -> bool:
        """
        Valida se o contexto recuperado é realmente útil para a pergunta.
        Evita que o LLM tente responder com base em lixo.
        """
        score = self._reranker.predict([query, context])
        return score >= threshold