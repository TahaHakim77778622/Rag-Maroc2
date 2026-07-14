"""
Client LLM via API compatible OpenAI (SDK officiel).

Fonctionne avec :
  - OpenAI (https://api.openai.com/v1)
  - Groq, Mistral, Together, etc. : définir OPENAI_BASE_URL + clé fournie par le fournisseur.

Ne jamais committer de clé : utiliser un fichier .env (voir env.example à la racine).
"""

from __future__ import annotations

import logging
from typing import Any

from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIChatClient(LLMClient):
    """Appelle chat.completions avec le prompt RAG complet comme message utilisateur."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from openai import OpenAI  # noqa: PLC0415

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = extra_kwargs or {}
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self._client = OpenAI(**kwargs)
        logger.info("LLM : modèle=%s base_url=%s", model, base_url or "(défaut OpenAI)")

    def complete(self, prompt: str) -> str:
        from openai import APIError, APITimeoutError, RateLimitError  # noqa: PLC0415

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **self._extra,
            )
        except RateLimitError as e:
            raise RuntimeError(
                "Quota ou limite de débit API dépassée. Réessayez plus tard."
            ) from e
        except APITimeoutError as e:
            raise RuntimeError("Délai d’attente API LLM dépassé.") from e
        except APIError as e:
            raise RuntimeError(f"Erreur API LLM : {e.message or e}") from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Appel LLM impossible : {e}") from e

        choice = resp.choices[0].message
        text = (choice.content or "").strip()
        if not text:
            raise RuntimeError("Le modèle a renvoyé une réponse vide.")
        return text
