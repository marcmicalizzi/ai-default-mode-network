import copy
import unittest

from dmn.responses import responses_to_chat


class ResponsesConversionTest(unittest.TestCase):
    def test_instructions_reasoning_calls_outputs_and_assistant_merging(self):
        body = {"model": "gemma", "instructions": "original system", "temperature": .7,
            "input": [{"role": "user", "content": "question"},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "thought"}]},
                {"type": "message", "role": "assistant", "id": "answer", "status": "completed",
                 "content": [{"type": "output_text", "text": "answer"}]},
                {"type": "function_call", "call_id": "call-1", "name": "clock", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call-1", "output": [{"type": "input_text", "text": "12:00"}]},
                {"type": "message", "role": "assistant", "content": "finished"}],
            "tools": [{"type": "function", "name": "clock", "parameters": {}}],
            "reasoning": {"effort": "high"}, "max_output_tokens": 50}
        before = copy.deepcopy(body)
        result = responses_to_chat(body)
        self.assertEqual(body, before)
        self.assertEqual(result["messages"], [
            {"role": "system", "content": "original system"},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "reasoning_content": "thought", "content": [{"type": "text", "text": "answer"}],
             "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "clock", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": [{"type": "text", "text": "12:00"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "finished"}]}])
        self.assertEqual(result["temperature"], .7)
        self.assertEqual(result["tools"][0]["function"]["strict"], True)
        self.assertEqual(result["max_tokens"], 50)
        self.assertEqual(result["reasoning_effort"], "high")
        self.assertNotIn("input", result)
        self.assertNotIn("instructions", result)

    def test_text_input_and_refusal_preserve_source_representation(self):
        self.assertEqual(responses_to_chat({"input": "text"})["messages"], [{"role": "user", "content": "text"}])
        result = responses_to_chat({"input": [{"role": "assistant", "type": "message", "id": "x", "status": "completed",
            "content": [{"type": "refusal", "refusal": "not provided"}]}]})
        self.assertEqual(result["messages"][0], {"role": "assistant", "id": "x", "content": [{"type": "refusal", "refusal": "not provided"}]})

    def test_unsupported_context_is_never_silently_removed(self):
        for body in ({"input": "x", "previous_response_id": "old"},
                     {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "http://example.invalid/image"}]}]},
                     {"input": [{"type": "reasoning", "summary": [{"text": "summary only"}]}]},
                     {"input": "x", "tools": [{"type": "web_search"}]}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                responses_to_chat(body)
