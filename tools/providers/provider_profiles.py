"""Provider profile resolution for live model calls.

This module keeps provider credentials and fallback policy outside proposal and
graph extraction logic. It intentionally supports legacy OPENAI_* variables so
existing workflows keep working while allowing named primary/fallback profiles.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


SUPPORTED_API_MODES = {"responses", "chat_completions"}
FALLBACK_ERROR_MARKERS = (
    "model_not_found",
    "no available channel",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "retryable",
    "http error 408",
    "http error 409",
    "http error 429",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "transport failure",
)


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    base_url: str
    api_key: str
    api_mode: str
    weak_model: str
    strong_model: str
    user_agent: str
    max_retries: int
    retry_base_seconds: float
    retry_max_seconds: float


@dataclass(frozen=True)
class ProviderProfileBundle:
    provider: str
    api_mode: str
    weak_model: str
    strong_model: str
    primary: ProviderProfile | None
    fallback: ProviderProfile | None
    fallback_enabled: bool

    @property
    def primary_profile_id(self) -> str | None:
        return self.primary.profile_id if self.primary else None

    @property
    def fallback_profile_id(self) -> str | None:
        return self.fallback.profile_id if self.fallback else None

    def manifest_fields(self) -> dict[str, object]:
        return {
            "provider_profile_id": self.primary_profile_id,
            "provider_fallback_profile_id": self.fallback_profile_id,
            "provider_fallback_enabled": self.fallback_enabled,
        }


def env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _profile_prefix(profile_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", profile_id).strip("_").upper()
    if not cleaned:
        raise ValueError("Provider profile id cannot be empty.")
    return f"PSML_PROVIDER_{cleaned}"


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in {None, ""}:
            return str(value)
    return default


def load_named_profile(profile_id: str, *, fallback_from_legacy: bool = False) -> ProviderProfile:
    prefix = _profile_prefix(profile_id)
    provider = _env_first(f"{prefix}_PROVIDER", default="openai")
    api_mode = _env_first(f"{prefix}_API_MODE", default=os.environ.get("OPENAI_API_MODE", "responses") if fallback_from_legacy else "responses")
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode for provider profile {profile_id}: {api_mode}")
    api_key = _env_first(f"{prefix}_API_KEY", default=os.environ.get("OPENAI_API_KEY", "") if fallback_from_legacy else "")
    api_key_env = os.environ.get(f"{prefix}_API_KEY_ENV")
    if api_key_env:
        api_key = os.environ.get(api_key_env, api_key)
    base_url = _env_first(f"{prefix}_BASE_URL", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") if fallback_from_legacy else "https://api.openai.com/v1")
    return ProviderProfile(
        profile_id=profile_id,
        provider=provider,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        api_mode=api_mode,
        weak_model=_env_first(f"{prefix}_WEAK_MODEL", default=os.environ.get("OPENAI_MODEL_WEAK", "") if fallback_from_legacy else ""),
        strong_model=_env_first(f"{prefix}_STRONG_MODEL", default=os.environ.get("OPENAI_MODEL_STRONG", "") if fallback_from_legacy else ""),
        user_agent=_env_first(f"{prefix}_USER_AGENT", default=os.environ.get("OPENAI_USER_AGENT", "curl/8.19.0")),
        max_retries=int(_env_first(f"{prefix}_MAX_RETRIES", default=os.environ.get("OPENAI_MAX_RETRIES", "2"))),
        retry_base_seconds=float(_env_first(f"{prefix}_RETRY_BASE_SECONDS", default=os.environ.get("OPENAI_RETRY_BASE_SECONDS", "1.0"))),
        retry_max_seconds=float(_env_first(f"{prefix}_RETRY_MAX_SECONDS", default=os.environ.get("OPENAI_RETRY_MAX_SECONDS", "12.0"))),
    )


def resolve_provider_profile_bundle(
    *,
    provider: str,
    api_mode: str | None,
    weak_model: str | None,
    strong_model: str | None,
    provider_profile: str | None = None,
    fallback_provider_profile: str | None = None,
) -> ProviderProfileBundle:
    if provider != "openai":
        resolved_api_mode = api_mode or os.environ.get("OPENAI_API_MODE") or "responses"
        return ProviderProfileBundle(
            provider=provider,
            api_mode=resolved_api_mode,
            weak_model=weak_model or os.environ.get("OPENAI_MODEL_WEAK") or "mock-weak-model",
            strong_model=strong_model or os.environ.get("OPENAI_MODEL_STRONG") or "mock-strong-model",
            primary=None,
            fallback=None,
            fallback_enabled=False,
        )

    primary_id = provider_profile or os.environ.get("PSML_PROVIDER_DEFAULT") or os.environ.get("OPENAI_PROVIDER_PROFILE") or "legacy_openai"
    fallback_id = fallback_provider_profile or os.environ.get("PSML_PROVIDER_FALLBACK") or os.environ.get("OPENAI_FALLBACK_PROVIDER_PROFILE") or ""
    primary = load_named_profile(primary_id, fallback_from_legacy=(primary_id == "legacy_openai"))
    fallback = load_named_profile(fallback_id, fallback_from_legacy=False) if fallback_id else None
    resolved_api_mode = api_mode or primary.api_mode or os.environ.get("OPENAI_API_MODE") or "responses"
    if resolved_api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {resolved_api_mode}")
    resolved_weak_model = weak_model or primary.weak_model or os.environ.get("OPENAI_MODEL_WEAK") or "mock-weak-model"
    resolved_strong_model = strong_model or primary.strong_model or os.environ.get("OPENAI_MODEL_STRONG") or "mock-strong-model"
    return ProviderProfileBundle(
        provider=provider,
        api_mode=resolved_api_mode,
        weak_model=resolved_weak_model,
        strong_model=resolved_strong_model,
        primary=primary,
        fallback=fallback,
        fallback_enabled=fallback is not None and bool(fallback.api_key),
    )


def should_fallback_provider_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in FALLBACK_ERROR_MARKERS)
