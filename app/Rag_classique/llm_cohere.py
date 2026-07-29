"""
Client LLM via l’API Cohere (Chat v2).

Clé : COHERE_API_KEY (dashboard https://dashboard.cohere.com )
Modèles chat à jour (ex.) : command-r-08-2024, command-r-plus-08-2024, command-a-03-2025 — voir https://docs.cohere.com/docs/models

Ne jamais committer la clé : utiliser .env (voir env.example).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.Rag_classique.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _extract_assistant_text(message: Any) -> str:
    """Lit le texte dans message.content (liste d’items text / thinking)."""
    parts: list[str] = []
    content = getattr(message, "content", None) or []
    for item in content:
        t = getattr(item, "type", None)
        if t == "text":
            tx = getattr(item, "text", None)
            if tx:
                parts.append(tx)
        elif t == "thinking":
            th = getattr(item, "thinking", None)
            if th:
                parts.append(th)
    return "\n".join(parts).strip()


def _is_retryable_cohere_error(exc: Exception) -> bool:
    """Timeouts réseau / surcharge ponctuelle : on retente plutôt que d’abandonner."""
    tname = type(exc).__name__
    if tname in (
        "ReadTimeout",
        "ReadTimeoutError",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "ProtocolError",
    ):
        return True
    s = str(exc).lower()
    return any(
        x in s
        for x in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
        )
    )


class CohereChatClient(LLMClient):
    """Envoie le prompt RAG complet comme message utilisateur (API Chat v2)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 240.0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        from cohere import ClientV2  # noqa: PLC0415

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self._client = ClientV2(**kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max(1, max_retries)
        logger.info("LLM Cohere : modèle=%s timeout=%s retries=%s", model, timeout, self._max_retries)

    def complete(self, prompt: str) -> str:
        from cohere.types.chat_message_v2 import UserChatMessageV2  # noqa: PLC0415

        messages = [UserChatMessageV2(content=prompt)]

        last_exc: Exception | None = None
        resp = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                break
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt + 1 < self._max_retries and _is_retryable_cohere_error(e):
                    wait = min(2.0 * (2**attempt), 30.0)
                    logger.warning(
                        "Cohere indisponible (tentative %s/%s) : %s — nouvel essai dans %.1fs",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Erreur API Cohere : {e}") from e
        if resp is None:
            assert last_exc is not None
            raise RuntimeError(f"Erreur API Cohere : {last_exc}") from last_exc

        text = _extract_assistant_text(resp.message)
        if not text:
            raise RuntimeError("Cohere a renvoyé une réponse vide.")
        return text
