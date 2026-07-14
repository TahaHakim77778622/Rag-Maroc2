"""Interface LLM : implémentation factice + point d’extension pour une API réelle."""

from __future__ import annotations

import abc


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def complete(self, prompt: str) -> str:
        """Génère la réponse à partir du prompt complet (system + contexte + question)."""


class MockLLMClient(LLMClient):
    """
    Ne appelle aucun modèle : utile pour valider le pipeline sans clé API.
    Remplace par OpenAI / Mistral / etc. plus tard (même signature).
    """

    def complete(self, prompt: str) -> str:
        n = len(prompt)
        return (
            "[LLM mock] Aucun modèle n’est branché. Le prompt a été construit correctement "
            f"({n} caractères). Branchez un vrai LLM dans une sous-classe de LLMClient.\n\n"
            "Pour tester le retrieval seul, utilisez : python scripts/search_faiss.py \"...\""
        )
