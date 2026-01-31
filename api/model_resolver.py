"""Model resolution utilities for Chat2API.

This module centralizes how inbound model slugs are normalized into the
request models expected by the upstream ChatGPT service. The mapping is now
rule-driven so new models (such as GPT-5) can be supported without modifying
callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from api.models import model_proxy


@dataclass(frozen=True)
class ModelRule:
    """Definition of how to map an inbound model string to a request model."""

    match: str
    target: str


class ModelResolver:
    """Resolve inbound model names to request and response slugs."""

    DEFAULT_MODEL = "gpt-3.5-turbo-0125"

    # Order matters: more specific matches must come before broader families.
    MODEL_RULES: List[ModelRule] = [
        ModelRule("gpt-5.2-gizmo", "gpt-5.2"),
        ModelRule("gpt-5.2-codex", "gpt-5.2-codex"),
        ModelRule("gpt-5.2-pro", "gpt-5.2-pro"),
        ModelRule("gpt-5.2", "gpt-5.2"),
        ModelRule("gpt-5.1", "gpt-5.1"),
        ModelRule("gpt-5-mini", "gpt-5-mini"),
        ModelRule("gpt-5-nano", "gpt-5-nano"),
        ModelRule("gpt-5", "gpt-5"),
        ModelRule("gpt-4.5o", "gpt-4.5o"),
        ModelRule("gpt-4o-canmore", "gpt-4o-canmore"),
        ModelRule("gpt-4o-mini", "gpt-4o-mini"),
        ModelRule("gpt-4o", "gpt-4o"),
        ModelRule("gpt-4-mobile", "gpt-4-mobile"),
        ModelRule("gpt-4-gizmo", "gpt-4o"),
        ModelRule("gpt-4", "gpt-4"),
        ModelRule("o1-preview", "o1-preview"),
        ModelRule("o1-mini", "o1-mini"),
        ModelRule("o1", "o1"),
        ModelRule("gpt-3.5", "text-davinci-002-render-sha"),
        ModelRule("auto", "auto"),
    ]

    @classmethod
    def resolve(cls, origin_model: str | None) -> Tuple[str, str]:
        """Resolve origin model into response and request models.

        Returns a tuple of (response_model, request_model).
        """

        normalized = origin_model or cls.DEFAULT_MODEL
        response_model = model_proxy.get(normalized, normalized)
        request_model = cls._match_request_model(normalized)
        return response_model, request_model

    @classmethod
    def _match_request_model(cls, origin_model: str) -> str:
        lower_model = origin_model.lower()
        for rule in cls.MODEL_RULES:
            if rule.match in lower_model:
                return rule.target
        return "auto"