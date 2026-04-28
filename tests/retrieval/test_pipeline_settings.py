"""Regression tests for retrieval-phase runtime settings exposure."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from Chunking.config.settings import PROJECT_ROOT, PipelineSettings


class PipelineSettingsRetrievalExposureTests(unittest.TestCase):
    """Protect retrieval, guardrail, and retrieval-quality settings parsing."""

    def test_pipeline_settings_reads_new_retrieval_guardrail_and_metric_flags(self) -> None:
        """Ensure the active appsettings contract is exposed through PipelineSettings."""
        settings = PipelineSettings()

        self.assertEqual(settings.retrieval_candidate_pool_size, 12)
        self.assertTrue(settings.retrieval_query_normalization_enabled)
        self.assertTrue(
            settings.retrieval_query_normalization_strip_formatting_instructions
        )
        self.assertTrue(
            settings.retrieval_query_normalization_extract_formatting_directives
        )
        self.assertTrue(settings.retrieval_context_include_article_number)
        self.assertTrue(settings.retrieval_context_include_article_title)
        self.assertTrue(settings.retrieval_context_include_parent_structure)
        self.assertTrue(settings.guardrails_portuguese_coverage_enabled)
        self.assertTrue(settings.guardrails_portuguese_jailbreak_pattern_checks_enabled)
        self.assertTrue(settings.guardrails_pre_request_jailbreak_pattern_checks_enabled)
        self.assertTrue(settings.guardrails_post_response_sensitive_data_checks_enabled)
        self.assertTrue(settings.metrics_retrieval_quality_enabled)
        self.assertTrue(settings.metrics_track_candidate_pool_size)
        self.assertTrue(settings.metrics_track_selected_context_size)
        self.assertTrue(settings.metrics_track_context_truncation)
        self.assertTrue(settings.metrics_track_structural_richness)
        self.assertTrue(settings.retrieval_routing_enabled)
        self.assertEqual(settings.retrieval_routing_broad_candidate_pool_size, 16)
        self.assertEqual(settings.retrieval_evidence_strong_min_score, 0.75)
        self.assertTrue(settings.retrieval_response_policy_cautious_answer_enabled)
        self.assertTrue(settings.retrieval_response_policy_clarification_enabled)
        self.assertFalse(settings.benchmark_enabled)
        self.assertEqual(settings.response_generation_provider, "openai")
        self.assertEqual(
            settings.response_generation_openai_api_key_env_var,
            "OPENAI_API_KEY",
        )
        self.assertEqual(
            settings.benchmark_questions_path,
            PROJECT_ROOT / "benchmark" / "questions.jsonl",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_endpoint_url,
            "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_env_var,
            "EXTERNAL_GPT4O_API_KEY",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_question_field_name,
            "message",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_name,
            "x-api-key",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_prefix,
            "",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_channel_id,
            "cmnp58zt6205piw01mp9it1sf",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_thread_id,
            "Ri0MAVWCLcJuIN19DYw0q",
        )
        self.assertEqual(settings.response_generation_external_gpt4o_user_info, "{}")
        self.assertFalse(settings.embedding_comparison_enabled)
        self.assertEqual(
            settings.embedding_comparison_candidate_models,
            ["all-MiniLM-L6-v2", "Qwen/Qwen3-VL-Embedding-8B"],
        )
        self.assertFalse(settings.embedding_visualization_benchmark_overlay_enabled)

    def test_pipeline_settings_keeps_coherent_defaults_when_optional_blocks_are_missing(
        self,
    ) -> None:
        """Ensure omitted retrieval-correction settings fall back to coherent defaults."""
        minimal_settings = {
            "retrieval": {
                "enabled": True,
                "top_k": 5,
                "context": {
                    "max_chunks": 2,
                    "max_characters": 3000,
                },
            },
            "guardrails": {
                "enabled": True,
                "pre_request": {},
                "post_response": {},
            },
            "metrics": {
                "enabled": True,
            },
        }

        with patch(
            "Chunking.config.settings._load_appsettings",
            return_value=minimal_settings,
        ):
            settings = PipelineSettings()

        self.assertEqual(settings.retrieval_top_k, 5)
        self.assertEqual(settings.retrieval_candidate_pool_size, 8)
        self.assertFalse(settings.retrieval_query_normalization_enabled)
        self.assertTrue(
            settings.retrieval_query_normalization_strip_formatting_instructions
        )
        self.assertTrue(
            settings.retrieval_query_normalization_extract_formatting_directives
        )
        self.assertFalse(settings.retrieval_context_include_article_number)
        self.assertFalse(settings.retrieval_context_include_article_title)
        self.assertFalse(settings.retrieval_context_include_parent_structure)
        self.assertFalse(settings.guardrails_portuguese_coverage_enabled)
        self.assertFalse(settings.guardrails_portuguese_jailbreak_pattern_checks_enabled)
        self.assertFalse(settings.guardrails_pre_request_jailbreak_pattern_checks_enabled)
        self.assertFalse(settings.guardrails_post_response_sensitive_data_checks_enabled)
        self.assertFalse(settings.metrics_retrieval_quality_enabled)
        self.assertFalse(settings.metrics_track_candidate_pool_size)
        self.assertFalse(settings.metrics_track_selected_context_size)
        self.assertFalse(settings.metrics_track_context_truncation)
        self.assertFalse(settings.metrics_track_structural_richness)
        self.assertFalse(settings.retrieval_routing_enabled)
        self.assertTrue(settings.retrieval_routing_article_scoping_enabled)
        self.assertTrue(settings.retrieval_routing_document_inference_enabled)
        self.assertEqual(settings.retrieval_routing_document_inference_min_score, 3.0)
        self.assertEqual(settings.retrieval_routing_document_inference_min_margin, 2.0)
        self.assertEqual(settings.retrieval_routing_scoped_candidate_pool_size, 64)
        self.assertEqual(settings.retrieval_evidence_weak_min_score, 0.45)
        self.assertTrue(settings.retrieval_response_policy_cautious_answer_enabled)
        self.assertTrue(settings.retrieval_response_policy_clarification_enabled)
        self.assertFalse(settings.benchmark_enabled)
        self.assertEqual(
            settings.benchmark_output_root,
            PROJECT_ROOT / "data" / "benchmark_runs",
        )
        self.assertEqual(settings.response_generation_external_gpt4o_endpoint_url, "")
        self.assertEqual(settings.response_generation_external_gpt4o_timeout_seconds, 30.0)
        self.assertEqual(settings.response_generation_external_gpt4o_max_retries, 2)
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_name,
            "Authorization",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_prefix,
            "Bearer",
        )
        self.assertEqual(settings.response_generation_external_gpt4o_channel_id, "")
        self.assertEqual(settings.response_generation_external_gpt4o_thread_id, "")
        self.assertEqual(settings.response_generation_external_gpt4o_user_info, "{}")
        self.assertEqual(
            settings.response_generation_openai_api_key_env_var,
            "OPENAI_API_KEY",
        )
        self.assertFalse(settings.embedding_comparison_enabled)
        self.assertEqual(
            settings.embedding_visualization_benchmark_overlay_output_path,
            PROJECT_ROOT / "data" / "embeddings" / "benchmark_overlay.jsonl",
        )

    def test_pipeline_settings_reads_routing_benchmark_provider_and_comparison_settings(
        self,
    ) -> None:
        """Ensure retrieval-correction settings are parsed from appsettings."""
        configured_settings = {
            "embedding": {
                "comparison": {
                    "enabled": True,
                    "candidate_models": [
                        "all-MiniLM-L6-v2",
                        "Qwen/Qwen3-VL-Embedding-8B",
                    ],
                    "output_root": "artifacts/embedding-comparison",
                },
                "visualization": {
                    "benchmark_overlay": {
                        "enabled": True,
                        "output_path": "artifacts/benchmark-overlay.jsonl",
                    },
                },
            },
            "retrieval": {
                "routing": {
                    "enabled": True,
                    "article_scoping_enabled": False,
                    "document_scoping_enabled": True,
                    "comparative_retrieval_enabled": False,
                    "weak_evidence_retry_enabled": False,
                    "document_inference_enabled": False,
                    "document_inference_min_score": 4.5,
                    "document_inference_min_margin": 1.5,
                    "scoped_candidate_pool_size": 6,
                    "broad_candidate_pool_size": 18,
                },
                "evidence_quality": {
                    "strong_min_score": 0.82,
                    "weak_min_score": 0.51,
                    "ambiguity_score_margin": 0.07,
                    "conflict_score_margin": 0.14,
                },
                "response_policy": {
                    "cautious_answer_enabled": False,
                    "clarification_enabled": False,
                },
            },
            "response_generation": {
                "openai": {
                    "api_key_env_var": "DIRECT_OPENAI_API_KEY",
                },
                "external_gpt4o": {
                    "endpoint_url": "https://example.test/gpt4o",
                    "auth_env_var": "GPT4O_PROXY_TOKEN",
                    "question_field_name": "prompt",
                    "context_field_name": "evidence",
                    "instructions_field_name": "format",
                    "metadata_field_name": "trace",
                    "channel_id": "channel-1",
                    "thread_id": "thread-1",
                    "user_info": '{"kind": "test"}',
                    "user_id": "user-1",
                    "user_name": "Test User",
                    "auth_header_name": "x-api-key",
                    "auth_header_prefix": "",
                    "timeout_seconds": 45,
                    "max_retries": 4,
                },
            },
            "benchmark": {
                "enabled": True,
                "datasets": {
                    "questions_path": "benchmark/custom_questions.jsonl",
                    "guardrails_path": "benchmark/custom_guardrails.jsonl",
                },
                "outputs": {
                    "output_root": "artifacts/benchmark-runs",
                },
            },
        }

        with patch(
            "Chunking.config.settings._load_appsettings",
            return_value=configured_settings,
        ):
            settings = PipelineSettings()

        self.assertTrue(settings.retrieval_routing_enabled)
        self.assertFalse(settings.retrieval_routing_article_scoping_enabled)
        self.assertTrue(settings.retrieval_routing_document_scoping_enabled)
        self.assertFalse(settings.retrieval_routing_comparative_retrieval_enabled)
        self.assertFalse(settings.retrieval_routing_weak_evidence_retry_enabled)
        self.assertFalse(settings.retrieval_routing_document_inference_enabled)
        self.assertEqual(settings.retrieval_routing_document_inference_min_score, 4.5)
        self.assertEqual(settings.retrieval_routing_document_inference_min_margin, 1.5)
        self.assertEqual(settings.retrieval_routing_scoped_candidate_pool_size, 6)
        self.assertEqual(settings.retrieval_routing_broad_candidate_pool_size, 18)
        self.assertEqual(settings.retrieval_evidence_strong_min_score, 0.82)
        self.assertEqual(settings.retrieval_evidence_weak_min_score, 0.51)
        self.assertEqual(settings.retrieval_evidence_ambiguity_score_margin, 0.07)
        self.assertEqual(settings.retrieval_evidence_conflict_score_margin, 0.14)
        self.assertFalse(settings.retrieval_response_policy_cautious_answer_enabled)
        self.assertFalse(settings.retrieval_response_policy_clarification_enabled)
        self.assertTrue(settings.benchmark_enabled)
        self.assertEqual(
            settings.benchmark_questions_path,
            PROJECT_ROOT / "benchmark" / "custom_questions.jsonl",
        )
        self.assertEqual(
            settings.benchmark_guardrails_path,
            PROJECT_ROOT / "benchmark" / "custom_guardrails.jsonl",
        )
        self.assertEqual(
            settings.benchmark_output_root,
            PROJECT_ROOT / "artifacts" / "benchmark-runs",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_endpoint_url,
            "https://example.test/gpt4o",
        )
        self.assertEqual(
            settings.response_generation_openai_api_key_env_var,
            "DIRECT_OPENAI_API_KEY",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_env_var,
            "GPT4O_PROXY_TOKEN",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_question_field_name,
            "prompt",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_context_field_name,
            "evidence",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_instructions_field_name,
            "format",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_metadata_field_name,
            "trace",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_name,
            "x-api-key",
        )
        self.assertEqual(
            settings.response_generation_external_gpt4o_auth_header_prefix,
            "",
        )
        self.assertEqual(settings.response_generation_external_gpt4o_channel_id, "channel-1")
        self.assertEqual(settings.response_generation_external_gpt4o_thread_id, "thread-1")
        self.assertEqual(
            settings.response_generation_external_gpt4o_user_info,
            '{"kind": "test"}',
        )
        self.assertEqual(settings.response_generation_external_gpt4o_user_id, "user-1")
        self.assertEqual(settings.response_generation_external_gpt4o_user_name, "Test User")
        self.assertEqual(settings.response_generation_external_gpt4o_timeout_seconds, 45.0)
        self.assertEqual(settings.response_generation_external_gpt4o_max_retries, 4)
        self.assertTrue(settings.embedding_comparison_enabled)
        self.assertEqual(
            settings.embedding_comparison_output_root,
            PROJECT_ROOT / "artifacts" / "embedding-comparison",
        )
        self.assertTrue(settings.embedding_visualization_benchmark_overlay_enabled)
        self.assertEqual(
            settings.embedding_visualization_benchmark_overlay_output_path,
            PROJECT_ROOT / "artifacts" / "benchmark-overlay.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
