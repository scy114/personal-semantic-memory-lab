import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.providers.provider_profiles import (
    load_named_profile,
    resolve_provider_profile_bundle,
    should_fallback_provider_error,
)
from tools.proposals.proposal_runner import (
    FallbackModelProvider,
    ModelProvider,
    PromptPolicy,
    ProviderResult,
)


class ProviderProfileTests(unittest.TestCase):
    def test_legacy_openai_profile_reads_existing_openai_env(self):
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "OPENAI_API_MODE": "chat_completions",
            "OPENAI_MODEL_WEAK": "weak-model",
            "OPENAI_MODEL_STRONG": "strong-model",
        }
        with patch.dict(os.environ, env, clear=True):
            bundle = resolve_provider_profile_bundle(
                provider="openai",
                api_mode=None,
                weak_model=None,
                strong_model=None,
            )
        self.assertEqual(bundle.primary_profile_id, "legacy_openai")
        self.assertEqual(bundle.api_mode, "chat_completions")
        self.assertEqual(bundle.weak_model, "weak-model")
        self.assertEqual(bundle.strong_model, "strong-model")
        self.assertFalse(bundle.fallback_enabled)

    def test_named_profiles_support_key_indirection_and_fallback(self):
        env = {
            "REAL_PRIMARY_KEY": "primary-key",
            "REAL_FALLBACK_KEY": "fallback-key",
            "PSML_PROVIDER_DEFAULT": "openai_primary",
            "PSML_PROVIDER_FALLBACK": "claude_secondary",
            "PSML_PROVIDER_OPENAI_PRIMARY_API_KEY_ENV": "REAL_PRIMARY_KEY",
            "PSML_PROVIDER_OPENAI_PRIMARY_BASE_URL": "https://openai.test/v1",
            "PSML_PROVIDER_OPENAI_PRIMARY_API_MODE": "responses",
            "PSML_PROVIDER_OPENAI_PRIMARY_WEAK_MODEL": "gpt-weak",
            "PSML_PROVIDER_OPENAI_PRIMARY_STRONG_MODEL": "gpt-strong",
            "PSML_PROVIDER_CLAUDE_SECONDARY_API_KEY_ENV": "REAL_FALLBACK_KEY",
            "PSML_PROVIDER_CLAUDE_SECONDARY_BASE_URL": "https://claude-compatible.test/v1",
            "PSML_PROVIDER_CLAUDE_SECONDARY_API_MODE": "chat_completions",
            "PSML_PROVIDER_CLAUDE_SECONDARY_WEAK_MODEL": "claude-weak",
            "PSML_PROVIDER_CLAUDE_SECONDARY_STRONG_MODEL": "claude-strong",
        }
        with patch.dict(os.environ, env, clear=True):
            primary = load_named_profile("openai_primary")
            bundle = resolve_provider_profile_bundle(
                provider="openai",
                api_mode=None,
                weak_model=None,
                strong_model=None,
            )
        self.assertEqual(primary.api_key, "primary-key")
        self.assertEqual(bundle.primary_profile_id, "openai_primary")
        self.assertEqual(bundle.fallback_profile_id, "claude_secondary")
        self.assertTrue(bundle.fallback_enabled)
        self.assertEqual(bundle.fallback.weak_model, "claude-weak")

    def test_fallback_error_classifier_does_not_treat_schema_errors_as_provider_errors(self):
        self.assertTrue(should_fallback_provider_error(RuntimeError("model_not_found: unavailable")))
        self.assertTrue(should_fallback_provider_error(RuntimeError("HTTP error 429: rate limit")))
        self.assertTrue(should_fallback_provider_error(RuntimeError("OpenAI-compatible API retryable transport failure")))
        self.assertFalse(should_fallback_provider_error(ValueError("schema_validation_failed: missing quote")))
        self.assertFalse(should_fallback_provider_error(ValueError("source_text_quote_not_exact")))

    def test_fallback_provider_switches_only_on_provider_failure(self):
        class FailingProvider(ModelProvider):
            def generate(self, *, prompt, model_id, input_packet):
                raise RuntimeError("model_not_found: primary unavailable")

        class GoodProvider(ModelProvider):
            def generate(self, *, prompt, model_id, input_packet):
                return ProviderResult(
                    output_text="{}",
                    model_id=model_id,
                    provider="openai",
                    estimated_input_tokens=1,
                    estimated_output_tokens=1,
                    cache_hit=None,
                    latency_ms=1,
                    provider_profile_id="claude_secondary",
                )

        env = {
            "PSML_PROVIDER_DEFAULT": "openai_primary",
            "PSML_PROVIDER_FALLBACK": "claude_secondary",
            "PSML_PROVIDER_OPENAI_PRIMARY_API_KEY": "primary-key",
            "PSML_PROVIDER_OPENAI_PRIMARY_WEAK_MODEL": "gpt-weak",
            "PSML_PROVIDER_OPENAI_PRIMARY_STRONG_MODEL": "gpt-strong",
            "PSML_PROVIDER_CLAUDE_SECONDARY_API_KEY": "fallback-key",
            "PSML_PROVIDER_CLAUDE_SECONDARY_WEAK_MODEL": "claude-weak",
            "PSML_PROVIDER_CLAUDE_SECONDARY_STRONG_MODEL": "claude-strong",
        }
        with patch.dict(os.environ, env, clear=True):
            bundle = resolve_provider_profile_bundle(
                provider="openai",
                api_mode=None,
                weak_model=None,
                strong_model=None,
            )
        provider = FallbackModelProvider(FailingProvider(), GoodProvider(), bundle)
        result = provider.generate(
            prompt=PromptPolicy("test", Path(__file__), "prompt", "hash"),
            model_id="gpt-strong",
            input_packet={"text": "hello"},
        )
        self.assertEqual(result.model_id, "claude-strong")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_from_profile_id, "openai_primary")
        self.assertEqual(result.provider_profile_id, "claude_secondary")


if __name__ == "__main__":
    unittest.main()
