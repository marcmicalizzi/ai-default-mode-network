"""Check supported Responses conversion against the native count route, no inference.

This supplements the pinned source conversion tests. Equal token counts are
independent evidence, not a proof that two arbitrary token sequences are equal.
"""
import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn.initial_context import SOURCE_BUILD, local_request, validate_request
from dmn.responses import responses_to_chat
from dmn.storage import write_durable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:9933")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    props = local_request(args.server_url, "/props")
    assert props["build_info"] == SOURCE_BUILD
    cases = {
        "string_input": {"instructions": "Keep the original system.", "input": "A cloud over a garden."},
        "unicode_and_content_arrays": {"input": [
            {"role": "system", "content": "Preserve Unicode and line breaks."},
            {"role": "user", "content": [{"type": "input_text", "text": "Café\n雪"},
                                         {"type": "input_text", "text": "Second part."}]},
            {"role": "assistant", "type": "message", "id": "answer-1", "status": "completed", "content": "Recorded."}]},
        "reasoning_calls_outputs_merging": {"input": [
            {"role": "user", "content": "What time does this clock report?"},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Read the clock."}]},
            {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
            {"type": "function_call", "call_id": "clock-1", "name": "clock", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "clock-1", "output": [{"type": "input_text", "text": "12:00"}]},
            {"role": "assistant", "type": "message", "content": "The clock says 12:00."}],
            "tools": [{"type": "function", "name": "clock", "description": "Read a clock.",
                       "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}],
            "reasoning": {"effort": "high"}, "max_output_tokens": 50},
        "refusal_and_empty_tools": {"input": [{"role": "user", "content": "A deliberately missing value."},
            {"role": "assistant", "type": "message", "content": [{"type": "refusal", "refusal": "The value is unavailable."}]}], "tools": []},
    }
    results = []
    for name, body in cases.items():
        body["model"] = props.get("model_alias") or "diagnostic-model"
        converted = responses_to_chat(body)
        validate_request(converted)
        def checked(route, payload):
            try:
                return local_request(args.server_url, route, payload), None
            except HTTPError as exc:
                return None, {"status": exc.code, "body": json.loads(exc.read())}
        rendered, render_error = checked("/apply-template", converted)
        counted, count_error = checked("/v1/responses/input_tokens", body)
        if render_error or count_error:
            # The pinned converter preserves refusals, but its final chat parser
            # rejects that content type. Verify both paths reject it equally.
            results.append({"case": name, "render_error": render_error, "native_error": count_error,
                "matched": name == "refusal_and_empty_tools" and render_error == count_error})
        else:
            tokens = local_request(args.server_url, "/tokenize", {"content": rendered["prompt"], "add_special": True, "parse_special": True})["tokens"]
            results.append({"case": name, "converted_tokens": len(tokens), "native_responses_tokens": counted["input_tokens"],
                            "matched": counted["input_tokens"] == len(tokens)})
    report = {"verified": all(r["matched"] for r in results), "source_build": SOURCE_BUILD,
              "inference_performed": False, "cases": results,
              "limit": "Native count parity plus separately tested pinned conversion; counts alone do not prove prompt equality."}
    write_durable(args.report, report)
    print(json.dumps(report, indent=2))
    assert report["verified"]


if __name__ == "__main__":
    main()
