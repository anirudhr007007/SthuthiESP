import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_provider


def make_settings(**overrides):
    values = {
        "ai_provider": "offline",
        "ai_timeout": 3.0,
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "test-local-model",
        "gemini_model": "test-gemini",
        "gemini_api_key": "",
        "openai_model": "test-openai",
        "openai_api_key": "",
        "ai_base_url": "",
        "ai_api_key": "",
        "ai_model": "",
        "college_name": "Configured college",
        "cloud_min_interval": 0.0,
        "cloud_max_calls_per_hour": 0,
        "provider_probe_timeout": 1.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def fact(
    question="What can you do?",
    answer="I can answer this verified question.",
    source="test source",
    **extra,
):
    value = {
        "verified": True,
        "question": question,
        "answer": answer,
        "source": source,
    }
    value.update(extra)
    return value


class KnowledgeTests(unittest.TestCase):
    def test_loader_requires_explicit_complete_verified_facts(self):
        entries = [
            fact(answer="kept"),
            {"verified": False, "question": "No", "answer": "no", "source": "x"},
            {"question": "Missing flag", "answer": "no", "source": "x"},
            fact(answer="", source="source"),
            fact(answer="TODO: fill this", source="source"),
            fact(answer="unverified", source=""),
            "not an object",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(json.dumps({"facts": entries}), encoding="utf-8")
            loaded = ai_provider.load_knowledge(path)

        self.assertEqual([entry["answer"] for entry in loaded], ["kept"])

    def test_context_excludes_unverified_and_placeholder_entries(self):
        context = ai_provider._knowledge_context(
            [
                fact(question="Verified question", answer="Verified answer"),
                {"verified": False, "question": "Hidden", "answer": "Hidden", "source": "x"},
                fact(question="TODO", answer="placeholder", source="source"),
            ]
        )
        self.assertIn("Verified answer", context)
        self.assertNotIn("Hidden", context)
        self.assertNotIn("placeholder", context)

    def test_project_knowledge_file_has_verified_capabilities_and_templates(self):
        knowledge_path = (
            Path(__file__).resolve().parents[1] / "data" / "college_knowledge.json"
        )
        data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        self.assertTrue(data["facts"])
        self.assertTrue(all(item.get("verified") is True for item in data["facts"]))
        self.assertTrue(data["college_fact_templates"])
        self.assertTrue(all(item.get("verified") is False for item in data["college_fact_templates"]))

    def test_user_faq_answers_hod_without_calling_openai(self):
        knowledge_path = (
            Path(__file__).resolve().parents[1] / "data" / "college_knowledge.json"
        )
        facts = ai_provider.load_knowledge(knowledge_path)
        settings = make_settings(ai_provider="openai", openai_api_key="test-key")
        with patch.object(ai_provider, "_ask_openai") as ask:
            answer = ai_provider.answer_question(
                "Who is the HOD of Civil Engineering at MCE?", settings, facts
            )
        self.assertIn("Dr. Sumana Jayaprakash", answer)
        ask.assert_not_called()


class MatchingAndRoutingTests(unittest.TestCase):
    def test_exact_and_alias_faq_answers_do_not_call_provider(self):
        facts = [fact(aliases=["Tell me about this verified feature"])]
        settings = make_settings(ai_provider="ollama")
        with patch.object(ai_provider, "_ask_ollama") as ask:
            answer = ai_provider.answer_question(
                "Tell me about this verified feature", settings, facts
            )
        self.assertEqual(answer, "I can answer this verified question.")
        ask.assert_not_called()

    def test_generic_what_is_and_prefix_greeting_do_not_collide(self):
        facts = [
            fact(question="What is the library?", answer="Library answer."),
            fact(question="What is the hostel?", answer="Hostel answer."),
            fact(question="Hello", answer="Greeting answer.", aliases=["hi"]),
        ]
        settings = make_settings(ai_provider="offline")
        self.assertIsNone(ai_provider.match_verified_fact("What is", facts))
        self.assertEqual(
            ai_provider.match_verified_fact("hello what is the hostel", facts),
            facts[1],
        )
        self.assertEqual(
            ai_provider.answer_question("hello", settings, facts), "Greeting answer."
        )
        self.assertEqual(
            ai_provider.answer_question("hello what is the hostel", settings, facts),
            "Hostel answer.",
        )

    def test_answer_result_reports_the_local_route(self):
        facts = [fact(aliases=["Tell me about this verified feature"])]
        settings = make_settings(ai_provider="ollama")

        verified = ai_provider.answer_question_result(
            "Tell me about this verified feature", settings, facts
        )
        college = ai_provider.answer_question_result(
            "What are the admission fees?", make_settings(ai_provider="offline"), []
        )
        offline = ai_provider.answer_question_result(
            "What is photosynthesis?", make_settings(), []
        )

        self.assertEqual(verified.source, "verified_faq")
        self.assertEqual(college.source, "college_fallback")
        self.assertEqual(offline.source, "offline_fallback")
        self.assertIn("verified local answer", offline.text)

    def test_answer_result_reports_provider_and_failure_routes(self):
        settings = make_settings(ai_provider="ollama")
        with patch.object(ai_provider, "_ask_ollama", return_value="model answer"):
            answered = ai_provider.answer_question_result(
                "What is photosynthesis?", settings, []
            )
        self.assertEqual(answered.source, "provider")
        self.assertEqual(answered.provider, "ollama")
        self.assertEqual(answered.text, "model answer")

        with patch.object(
            ai_provider,
            "_ask_ollama",
            side_effect=ai_provider.AIProviderError("unavailable", code="network"),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                failed = ai_provider.answer_question_result(
                    "What is photosynthesis?", settings, []
                )
        self.assertEqual(failed.source, "provider_fallback")
        self.assertEqual(failed.provider, "ollama")
        self.assertNotIn("What is photosynthesis?", output.getvalue())

    def test_unknown_college_and_general_questions_use_configured_provider(self):
        settings = make_settings(ai_provider="ollama")
        with patch.object(ai_provider, "_ask_ollama", return_value="model answer") as ask:
            answer = ai_provider.answer_question("What are the admission fees?", settings, [])
        self.assertEqual(answer, "model answer")
        ask.assert_called_once()

        with patch.object(ai_provider, "_ask_ollama", return_value="model answer") as ask:
            answer = ai_provider.answer_question("What is photosynthesis?", settings, [])
        self.assertEqual(answer, "model answer")
        ask.assert_called_once()

    def test_ambiguous_fuzzy_match_fails_closed(self):
        facts = [
            fact(question="Tell me about registered visitor faces", answer="faces"),
            fact(question="Tell me about registered visitor badges", answer="badges"),
        ]
        self.assertIsNone(ai_provider.match_verified_fact("registered visitor", facts))

    def test_short_department_acronym_does_not_pick_another_hod(self):
        facts = [
            fact(question="Who is the HOD of Computer Science and Engineering at MCE?", answer="CSE"),
            fact(
                question="Who is the HOD of CSE Artificial Intelligence and Machine Learning at MCE?",
                answer="CSE AI",
            ),
        ]
        self.assertIsNone(ai_provider.match_verified_fact("Who is the HOD of CSE at MCE?", facts))


class AdapterTests(unittest.TestCase):
    def test_ollama_request_has_bounded_num_predict(self):
        settings = make_settings()
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"message": {"content": "local answer"}},
        ) as post:
            answer = ai_provider._ask_ollama("general question", settings, "system")
        self.assertEqual(answer, "local answer")
        payload = post.call_args.args[1]
        self.assertEqual(payload["options"]["num_predict"], ai_provider.MAX_OUTPUT_TOKENS)

    def test_gemini_key_is_header_not_url(self):
        settings = make_settings(gemini_api_key="secret-key", gemini_model="gemini-test")
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
        ) as post:
            answer = ai_provider._ask_gemini("general question", settings, "system")
        self.assertEqual(answer, "answer")
        endpoint, _, headers, _ = post.call_args.args
        self.assertNotIn("secret-key", endpoint)
        self.assertEqual(headers, {"x-goog-api-key": "secret-key"})

    def test_openai_client_is_no_retry_and_bounded(self):
        class FakeResponses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(output_text="openai answer")

        class FakeClient:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.responses = FakeResponses()
                self.instances.append(self)

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeClient
        settings = make_settings(openai_api_key="secret-key")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            answer = ai_provider._ask_openai("general question", settings, "system")
        self.assertEqual(answer, "openai answer")
        self.assertEqual(FakeClient.instances[0].kwargs["max_retries"], 0)
        self.assertEqual(
            FakeClient.instances[0].responses.kwargs["max_output_tokens"],
            ai_provider.MAX_OUTPUT_TOKENS,
        )

    def test_openai_uses_dependency_free_http_when_sdk_is_missing(self):
        settings = make_settings(openai_api_key="secret-key")
        with patch.dict(sys.modules, {"openai": None}):
            with patch.object(
                ai_provider,
                "_post_json",
                return_value={"output_text": "http answer"},
            ) as post:
                answer = ai_provider._ask_openai("general question", settings, "system")
        self.assertEqual(answer, "http answer")
        endpoint, payload, headers, _ = post.call_args.args
        self.assertEqual(endpoint, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["max_output_tokens"], ai_provider.MAX_OUTPUT_TOKENS)
        self.assertEqual(headers, {"Authorization": "Bearer secret-key"})

    def test_compatible_adapter_supports_local_http_and_bearer_header(self):
        settings = make_settings(
            ai_base_url="http://127.0.0.1:1234/v1",
            ai_api_key="local-secret",
            ai_model="local-compatible",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"choices": [{"message": {"content": "compatible answer"}}]},
        ) as post:
            answer = ai_provider._ask_compatible("general question", settings, "system")
        self.assertEqual(answer, "compatible answer")
        endpoint, _, headers, _ = post.call_args.args
        self.assertEqual(endpoint, "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(headers, {"Authorization": "Bearer local-secret"})

    def test_groq_adapter_uses_fixed_endpoint_and_bearer_header(self):
        settings = make_settings(
            groq_api_key="groq-secret",
            groq_model="llama-3.3-70b-versatile",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"choices": [{"message": {"content": "groq answer"}}]},
        ) as post:
            answer = ai_provider._ask_groq("general question", settings, "system")
        self.assertEqual(answer, "groq answer")
        endpoint, _, headers, _ = post.call_args.args
        self.assertEqual(endpoint, "https://api.groq.com/openai/v1/chat/completions")
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer groq-secret")
        self.assertIn("User-Agent", headers)
        self.assertTrue(headers["User-Agent"].startswith("espsthuthi2/"))


class TransportSafetyTests(unittest.TestCase):
    def test_remote_plain_http_and_url_credentials_are_rejected(self):
        for endpoint in (
            "http://example.test/api",
            "https://user:password@example.test/api",
            "https://example.test/api?key=secret",
            "https://example.test/api#fragment",
            "https://example.test:99999/api",
            "https://:443/api",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ai_provider.AIProviderError):
                    ai_provider._validate_endpoint(endpoint)

    def test_loopback_http_and_explicit_ports_are_accepted(self):
        for endpoint in (
            "http://127.0.0.1:11434",
            "http://localhost:1234/v1",
            "https://example.test:8443/v1",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(
                    ai_provider._validate_endpoint(endpoint), endpoint
                )

    def test_post_json_bounds_response_and_does_not_follow_redirects(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size):
                self.size = size
                return json.dumps({"ok": True}).encode("utf-8")

        fake_response = FakeResponse()
        with patch.object(ai_provider, "urlopen", return_value=fake_response) as open_url:
            self.assertEqual(
                ai_provider._post_json(
                    "http://127.0.0.1:11434/api/chat", {"question": "x"}, {}, 1.0
                ),
                {"ok": True},
            )
        self.assertEqual(fake_response.size, ai_provider.MAX_RESPONSE_BYTES + 1)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 1.0)

        class HugeResponse(FakeResponse):
            def read(self, size):
                return b"x" * (ai_provider.MAX_RESPONSE_BYTES + 1)

        with patch.object(ai_provider, "urlopen", return_value=HugeResponse()):
            with self.assertRaises(ai_provider.AIProviderError):
                ai_provider._post_json("http://127.0.0.1:11434/api/chat", {}, {}, 1.0)

    def test_provider_failure_log_does_not_include_secret_or_question(self):
        settings = make_settings(ai_provider="gemini", gemini_api_key="secret-key")
        question = "What is my very private question?"
        with patch.object(
            ai_provider,
            "_ask_gemini",
            side_effect=ai_provider.AIProviderError("remote body with secret-key", code="http"),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                answer = ai_provider.answer_question(question, settings, [])
        self.assertIn("safe offline answer", output.getvalue())
        self.assertNotIn("secret-key", output.getvalue())
        self.assertNotIn(question, output.getvalue())
        self.assertIn("verified local answer", answer)

    def test_malformed_provider_response_is_safe(self):
        settings = make_settings(ai_provider="ollama")
        with patch.object(
            ai_provider,
            "_ask_ollama",
            side_effect=ai_provider.AIProviderError("empty", code="response"),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                answer = ai_provider.answer_question("What is photosynthesis?", settings, [])
        self.assertIn("safe offline answer", output.getvalue())
        self.assertIn("verified local answer", answer)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.principal = fact(
            question="Who is the current principal of MCE?",
            answer="Dr. Example is the principal.",
            aliases=["Who is the principal?"],
            keywords=["principal", "MCE"],
        )
        self.library = fact(
            question="Does MCE have a library?",
            answer="Yes, MCE has a library.",
            keywords=["library"],
        )
        self.hostel = fact(
            question="Does MCE have hostels?",
            answer="Yes, MCE has hostels.",
            keywords=["hostel"],
        )

    def test_exact_alias_is_retrieved_first(self):
        selected = ai_provider.select_relevant_facts(
            "Who is the principal?", [self.library, self.hostel, self.principal]
        )
        self.assertEqual(selected[0]["question"], self.principal["question"])

    def test_limit_is_respected(self):
        selected = ai_provider.select_relevant_facts(
            "library hostel", [self.library, self.hostel], limit=1
        )
        self.assertEqual(len(selected), 1)

    def test_character_budget_skips_oversized_follow_up_facts(self):
        large = fact(
            question="Hostel details",
            answer="x" * 500,
            source="test source",
        )
        selected = ai_provider.select_relevant_facts(
            "library hostel", [self.library, large], limit=2, max_chars=120
        )
        self.assertEqual([item["question"] for item in selected], ["Does MCE have a library?"])

    def test_unrelated_question_selects_nothing(self):
        self.assertEqual(
            ai_provider.select_relevant_facts(
                "What is photosynthesis?", [self.library, self.hostel]
            ),
            [],
        )

    def test_empty_question_keeps_stable_ordering(self):
        selected = ai_provider.select_relevant_facts(
            "", [self.library, self.hostel], limit=2
        )
        self.assertEqual(
            [item["question"] for item in selected],
            [self.library["question"], self.hostel["question"]],
        )


class RateLimiterTests(unittest.TestCase):
    def test_minimum_interval_blocks_second_attempt(self):
        clock = [0.0]
        limiter = ai_provider._RateLimiter(clock=lambda: clock[0])
        limiter.acquire(min_interval=10.0, max_per_hour=0)
        with self.assertRaises(ai_provider.AIProviderError) as caught:
            limiter.acquire(min_interval=10.0, max_per_hour=0, now=5.0)
        self.assertEqual(caught.exception.code, "rate_limit")
        limiter.acquire(min_interval=10.0, max_per_hour=0, now=10.0)

    def test_hourly_cap_rolls_over_after_the_window(self):
        limiter = ai_provider._RateLimiter(clock=lambda: 0.0)
        limiter.acquire(min_interval=0.0, max_per_hour=2, now=0.0)
        limiter.acquire(min_interval=0.0, max_per_hour=2, now=1.0)
        with self.assertRaises(ai_provider.AIProviderError) as caught:
            limiter.acquire(min_interval=0.0, max_per_hour=2, now=2.0)
        self.assertEqual(caught.exception.code, "rate_limit")
        limiter.acquire(
            min_interval=0.0,
            max_per_hour=2,
            now=ai_provider.RATE_LIMIT_WINDOW_SECONDS + 2.0,
        )

    def test_reset_clears_all_state(self):
        limiter = ai_provider._RateLimiter(clock=lambda: 0.0)
        limiter.acquire(min_interval=3600.0, max_per_hour=1)
        limiter.reset()
        limiter.acquire(min_interval=3600.0, max_per_hour=1)

    def test_no_limits_configured_is_a_noop(self):
        limiter = ai_provider._RateLimiter(clock=lambda: 0.0)
        for _ in range(50):
            limiter.acquire(min_interval=0.0, max_per_hour=0)


class ProviderProbeTests(unittest.TestCase):
    def test_offline_provider_is_reported_available(self):
        status = ai_provider.probe_provider(make_settings(ai_provider="offline"))
        self.assertTrue(status.available)
        self.assertIn("offline", status.detail.lower())

    def test_missing_cloud_credentials_fail_without_echoing_secrets(self):
        for provider, attribute in (
            ("gemini", "gemini_api_key"),
            ("openai", "openai_api_key"),
            ("groq", "groq_api_key"),
        ):
            with self.subTest(provider=provider):
                status = ai_provider.probe_provider(
                    make_settings(ai_provider=provider, **{attribute: ""})
                )
                self.assertFalse(status.available)
                self.assertIn(attribute.upper(), status.detail)

    def test_ollama_success_and_missing_model(self):
        settings = make_settings(ai_provider="ollama", ollama_model="test-local-model")
        with patch.object(
            ai_provider,
            "_get_json",
            return_value={"models": [{"name": "test-local-model:latest"}]},
        ):
            status = ai_provider.probe_provider(settings)
        self.assertTrue(status.available)

        with patch.object(
            ai_provider,
            "_get_json",
            return_value={"models": [{"name": "another-model:latest"}]},
        ):
            status = ai_provider.probe_provider(settings)
        self.assertFalse(status.available)
        self.assertIn("not pulled", status.detail)

    def test_ollama_unreachable_is_reported_not_raised(self):
        settings = make_settings(ai_provider="ollama")
        with patch.object(
            ai_provider,
            "_get_json",
            side_effect=ai_provider.AIProviderError("connection refused", code="network"),
        ):
            status = ai_provider.probe_provider(settings)
        self.assertFalse(status.available)
        self.assertIn("not reachable", status.detail)

    def test_groq_provider_requires_credentials(self):
        missing_key = ai_provider.probe_provider(
            make_settings(ai_provider="groq", groq_api_key="", groq_model="test-model")
        )
        self.assertFalse(missing_key.available)
        self.assertIn("GROQ_API_KEY", missing_key.detail)

        missing_model = ai_provider.probe_provider(
            make_settings(ai_provider="groq", groq_api_key="test-key", groq_model="")
        )
        self.assertFalse(missing_model.available)
        self.assertIn("GROQ_MODEL", missing_model.detail)

        status = ai_provider.probe_provider(
            make_settings(ai_provider="groq", groq_api_key="test-key", groq_model="test-model")
        )
        self.assertTrue(status.available)
        self.assertIn("credentials configured", status.detail)

    def test_compatible_provider_requires_url_and_model(self):
        missing_url = ai_provider.probe_provider(
            make_settings(ai_provider="compatible", ai_base_url="", ai_model="m")
        )
        self.assertFalse(missing_url.available)

        missing_model = ai_provider.probe_provider(
            make_settings(
                ai_provider="compatible",
                ai_base_url="http://127.0.0.1:1234/v1",
                ai_model="",
            )
        )
        self.assertFalse(missing_model.available)
        self.assertIn("AI_MODEL", missing_model.detail)

        ready = ai_provider.probe_provider(
            make_settings(
                ai_provider="compatible",
                ai_base_url="http://127.0.0.1:1234/v1",
                ai_model="local-model",
            )
        )
        self.assertTrue(ready.available)

    def test_unknown_provider_is_rejected(self):
        status = ai_provider.probe_provider(make_settings(ai_provider="mystery"))
        self.assertFalse(status.available)
        self.assertIn("unknown", status.detail)


class CloudLimitTests(unittest.TestCase):
    def setUp(self):
        ai_provider.reset_rate_limits()

    def tearDown(self):
        ai_provider.reset_rate_limits()

    def test_remote_provider_calls_are_rate_limited_with_safe_fallback(self):
        settings = make_settings(
            ai_provider="gemini",
            gemini_api_key="test-key",
            cloud_min_interval=3600.0,
            cloud_max_calls_per_hour=10,
        )
        with patch.object(
            ai_provider, "_ask_gemini", return_value="model answer"
        ) as ask:
            first = ai_provider.answer_question(
                "What is photosynthesis?", settings, []
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                second = ai_provider.answer_question(
                    "What is photosynthesis?", settings, []
                )

        self.assertEqual(first, "model answer")
        self.assertNotEqual(second, "model answer")
        self.assertIn("rate_limit", output.getvalue())
        self.assertEqual(ask.call_count, 1)

    def test_local_ollama_is_not_subject_to_cloud_limits(self):
        settings = make_settings(
            ai_provider="ollama",
            cloud_min_interval=3600.0,
            cloud_max_calls_per_hour=1,
        )
        with patch.object(
            ai_provider, "_ask_ollama", return_value="local answer"
        ) as ask:
            first = ai_provider.answer_question(
                "What is photosynthesis?", settings, []
            )
            second = ai_provider.answer_question(
                "What is photosynthesis?", settings, []
            )
        self.assertEqual(first, "local answer")
        self.assertEqual(second, "local answer")
        self.assertEqual(ask.call_count, 2)

    def test_rate_limit_classification_is_correct(self):
        self.assertTrue(
            ai_provider._provider_needs_cloud_limit(
                "gemini", make_settings(ai_provider="gemini")
            )
        )
        self.assertFalse(
            ai_provider._provider_needs_cloud_limit(
                "compatible",
                make_settings(
                    ai_provider="compatible", ai_base_url="http://127.0.0.1:1234"
                ),
            )
        )
        self.assertTrue(
            ai_provider._provider_needs_cloud_limit(
                "compatible",
                make_settings(
                    ai_provider="compatible", ai_base_url="https://example.test"
                ),
            )
        )


class BoundedAnswerTests(unittest.TestCase):
    def test_short_answer_is_trimmed(self):
        self.assertEqual(ai_provider._bounded_answer("  hello  "), "hello")

    def test_empty_or_non_string_answer_is_rejected(self):
        for value in ("", "   ", None, 42):
            with self.subTest(value=value):
                with self.assertRaises(ai_provider.AIProviderError):
                    ai_provider._bounded_answer(value)

    def test_long_answer_is_truncated_on_a_word_boundary(self):
        answer = ai_provider._bounded_answer("word " * 1000)
        self.assertLessEqual(len(answer), ai_provider.MAX_ANSWER_CHARS)
        self.assertTrue(answer.endswith("..."))
        self.assertNotIn("  ", answer)


class CollegeDetectionTests(unittest.TestCase):
    def test_college_questions_are_detected(self):
        for question in (
            "What are the admission fees?",
            "Who is the principal?",
            "What are the library timings?",
            "How do I contact the college office?",
        ):
            with self.subTest(question=question):
                self.assertTrue(ai_provider.is_college_specific_question(question))

    def test_general_science_questions_are_not_college_specific(self):
        for question in (
            "What is photosynthesis?",
            "Who wrote Hamlet?",
            "How does a transformer work?",
            "",
        ):
            with self.subTest(question=question):
                self.assertFalse(ai_provider.is_college_specific_question(question))

    def test_configured_college_name_is_detected(self):
        self.assertTrue(
            ai_provider.is_college_specific_question(
                "Tell me about Malnad College", "Malnad College of Engineering"
            )
        )


class NvidiaProviderTests(unittest.TestCase):
    def test_nvidia_adapter_defaults_to_hosted_endpoint_and_bearer_header(self):
        settings = make_settings(
            ai_provider="nvidia",
            ai_api_key="nvapi-secret",
            ai_model="z-ai/glm-5.3-flash",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"choices": [{"message": {"content": "nvidia answer"}}]},
        ) as post:
            answer = ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(answer, "nvidia answer")
        endpoint, payload, headers, _ = post.call_args.args
        self.assertEqual(endpoint, "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(payload["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(payload["max_tokens"], ai_provider.MAX_OUTPUT_TOKENS)
        self.assertEqual(headers["Authorization"], "Bearer nvapi-secret")
        self.assertIn("User-Agent", headers)
        self.assertTrue(headers["User-Agent"].startswith("espsthuthi2/"))

    def test_nvidia_adapter_honours_explicit_base_url(self):
        settings = make_settings(
            ai_provider="nvidia",
            ai_base_url="http://127.0.0.1:8000/v1",
            ai_api_key="local-secret",
            ai_model="local-nim",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"choices": [{"message": {"content": "nim answer"}}]},
        ) as post:
            answer = ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(answer, "nim answer")
        endpoint, _, headers, _ = post.call_args.args
        self.assertEqual(endpoint, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer local-secret")

    def test_nvidia_adapter_accepts_content_part_lists(self):
        settings = make_settings(
            ai_provider="nvidia",
            ai_api_key="k",
            ai_model="m",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={
                "choices": [{"message": {"content": [{"text": "first"}, {"text": "second"}]}}]
            },
        ):
            answer = ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(answer, "first second")

    def test_nvidia_adapter_disables_thinking_and_retries_plain_on_http_error(self):
        settings = make_settings(
            ai_provider="nvidia",
            ai_api_key="nvapi-secret",
            ai_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        )
        with patch.object(
            ai_provider,
            "_post_json",
            return_value={"choices": [{"message": {"content": "short answer"}}]},
        ) as post:
            answer = ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(answer, "short answer")
        _, payload, _, _ = post.call_args.args
        self.assertEqual(
            payload["chat_template_kwargs"], {"thinking": False}
        )

        # An HTTP rejection (for example a template that does not accept the
        # kwarg) triggers exactly one plain retry without it.
        responses = [
            ai_provider.AIProviderError("provider returned an HTTP error", code="http"),
            {"choices": [{"message": {"content": "plain answer"}}]},
        ]
        with patch.object(ai_provider, "_post_json", side_effect=responses) as post:
            answer = ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(answer, "plain answer")
        self.assertEqual(post.call_count, 2)
        _, retried_payload, _, _ = post.call_args.args
        self.assertNotIn("chat_template_kwargs", retried_payload)

        # Timeouts and network failures are never retried.
        with patch.object(
            ai_provider,
            "_post_json",
            side_effect=ai_provider.AIProviderError(
                "provider request failed", code="network"
            ),
        ) as post:
            with self.assertRaises(ai_provider.AIProviderError):
                ai_provider._ask_nvidia("general question", settings, "system")
        self.assertEqual(post.call_count, 1)

    def test_nvidia_adapter_requires_key_and_model(self):
        missing_key = make_settings(
            ai_provider="nvidia", ai_api_key="", ai_model="m"
        )
        with self.assertRaises(ai_provider.AIProviderError) as ctx:
            ai_provider._ask_nvidia("q", missing_key, "system")
        self.assertIn("AI_API_KEY", str(ctx.exception))

        missing_model = make_settings(
            ai_provider="nvidia", ai_api_key="k", ai_model=""
        )
        with self.assertRaises(ai_provider.AIProviderError) as ctx:
            ai_provider._ask_nvidia("q", missing_model, "system")
        self.assertIn("AI_MODEL", str(ctx.exception))

    def test_nvidia_provider_probe(self):
        missing_key = ai_provider.probe_provider(
            make_settings(ai_provider="nvidia", ai_api_key="", ai_model="m")
        )
        self.assertFalse(missing_key.available)
        self.assertIn("AI_API_KEY", missing_key.detail)

        missing_model = ai_provider.probe_provider(
            make_settings(ai_provider="nvidia", ai_api_key="nvapi-secret", ai_model="")
        )
        self.assertFalse(missing_model.available)
        self.assertIn("AI_MODEL", missing_model.detail)

        ready = ai_provider.probe_provider(
            make_settings(ai_provider="nvidia", ai_api_key="nvapi-secret", ai_model="m")
        )
        self.assertTrue(ready.available)
        self.assertIn("network not tested", ready.detail)

        with patch.object(
            ai_provider,
            "_get_json",
            return_value={"data": [{"id": "m"}]},
        ) as get:
            verified = ai_provider.probe_provider(
                make_settings(
                    ai_provider="nvidia", ai_api_key="nvapi-secret", ai_model="m"
                ),
                network=True,
            )
        self.assertTrue(verified.available)
        self.assertIn("/models", get.call_args.args[0])
        self.assertEqual(
            get.call_args.args[1]["Authorization"], "Bearer nvapi-secret"
        )

    def test_nvidia_is_rate_limited_like_other_cloud_providers(self):
        ai_provider.reset_rate_limits()
        settings = make_settings(
            ai_provider="nvidia",
            ai_api_key="nvapi-secret",
            ai_model="m",
            cloud_min_interval=3600.0,
            cloud_max_calls_per_hour=10,
        )
        try:
            with patch.object(
                ai_provider, "_ask_nvidia", return_value="model answer"
            ) as ask:
                first = ai_provider.answer_question(
                    "What is photosynthesis?", settings, []
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    second = ai_provider.answer_question(
                        "What is photosynthesis?", settings, []
                    )
            self.assertEqual(first, "model answer")
            self.assertEqual(ask.call_count, 1)
            self.assertNotEqual(second, "model answer")
            self.assertTrue(second.startswith("I do not have"))
        finally:
            ai_provider.reset_rate_limits()


if __name__ == "__main__":
    unittest.main()
