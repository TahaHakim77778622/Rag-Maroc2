"""
Sélection du client LLM selon les variables d’environnement (et optionnellement .env).

Fournisseurs :
  - Cohere : COHERE_API_KEY (+ LLM_PROVIDER=cohere ou mode auto si seule clé Cohere)
  - OpenAI-compatible : OPENAI_API_KEY (+ OPENAI_BASE_URL optionnel)
  - LLM_PROVIDER=auto|cohere|openai pour forcer le fournisseur quand plusieurs clés sont présentes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.Rag_classique.llm_client import LLMClient, MockLLMClient

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
        logger.debug("Variables chargées depuis %s", env_path)


_load_dotenv()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_llm_client() -> LLMClient:
    """
    Ordre :
      1. LLM_MOCK=1 → mock
      2. LLM_PROVIDER=cohere|openai (ou auto) + clé correspondante
      3. auto : COHERE_API_KEY seule ou avant OpenAI → Cohere ; sinon OpenAI si clé
      4. sinon mock + avertissement
    """
    if os.environ.get("LLM_MOCK", "").strip().lower() in ("1", "true", "yes"):
        logger.info("LLM : mode mock (LLM_MOCK activé).")
        return MockLLMClient()

    provider = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    if provider not in ("auto", "cohere", "openai"):
        logger.warning("LLM_PROVIDER inconnu %r — traité comme auto.", provider)
        provider = "auto"

    cohere_key = os.environ.get("COHERE_API_KEY", "").strip()
    openai_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )

    temperature = _float_env("LLM_TEMPERATURE", 0.2)
    max_tokens = _int_env("LLM_MAX_TOKENS", 2048)
    timeout = _float_env("LLM_TIMEOUT", 240.0)
    max_retries = _int_env("LLM_MAX_RETRIES", 3)

    use_cohere = False
    use_openai = False

    if provider == "cohere":
        use_cohere = bool(cohere_key)
        if not use_cohere:
            logger.warning("LLM_PROVIDER=cohere mais COHERE_API_KEY manquant — mock.")
            return MockLLMClient()
    elif provider == "openai":
        use_openai = bool(openai_key)
        if not use_openai:
            logger.warning("LLM_PROVIDER=openai mais OPENAI_API_KEY / LLM_API_KEY manquant — mock.")
            return MockLLMClient()
    else:
        # auto : préférer Cohere si les deux clés (explicite via doc), sinon première disponible
        if cohere_key:
            use_cohere = True
        elif openai_key:
            use_openai = True

    if use_cohere:
        from app.Rag_classique.llm_cohere import CohereChatClient  # noqa: PLC0415

        # Noms à jour : https://docs.cohere.com/docs/models — plus d’usage de « command-r-plus » sans date (retiré).
        model = (
            os.environ.get("COHERE_MODEL", "").strip()
            or os.environ.get("LLM_MODEL", "").strip()
            or "command-r-08-2024"
        )
        base_url = os.environ.get("COHERE_BASE_URL", "").strip() or None
        return CohereChatClient(
            api_key=cohere_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if use_openai:
        from app.Rag_classique.llm_openai import OpenAIChatClient  # noqa: PLC0415

        model = (
            os.environ.get("LLM_MODEL", "").strip()
            or os.environ.get("OPENAI_MODEL", "").strip()
            or "gpt-4o-mini"
        )
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return OpenAIChatClient(
            api_key=openai_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    logger.warning(
        "Aucune clé LLM (COHERE_API_KEY ou OPENAI_API_KEY / LLM_API_KEY) — mock. "
        "Voir env.example et définissez LLM_PROVIDER si besoin."
    )
    return MockLLMClient()
