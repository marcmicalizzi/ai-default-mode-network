"""Text-only port of llama-server b10502's Responses-to-chat conversion.

Pinned reference: tools/server/server-chat.cpp at 0adcc3bb5. Keep content arrays,
reasoning boundaries, assistant merging, call IDs and tool strict defaults.
Multimodal and unsupported items fail instead of silently dropping information.
The final prompt is still rendered/tokenized by the actual native server.
"""
from copy import deepcopy


def responses_to_chat(body):
    if "input" not in body or body.get("previous_response_id"):
        raise ValueError("Responses import needs complete input, without previous_response_id")
    result = deepcopy(body)
    items = result.pop("input")
    messages = []
    if "instructions" in result:
        instructions = result.pop("instructions")
        if not isinstance(instructions, str):
            raise ValueError("Responses instructions must be text")
        messages.append({"role": "system", "content": instructions})
    if isinstance(items, str):
        messages.append({"role": "user", "content": items})
    elif isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Responses input items must be objects")
            merge = bool(messages and messages[-1].get("role") == "assistant")
            content = item.get("content")
            if isinstance(content, str):
                content = [{"type": "input_text", "text": content}]
                item["content"] = content
            role, kind = item.get("role"), item.get("type")
            if isinstance(content, list) and role in {"user", "system", "developer"}:
                parts = []
                for part in content:
                    if part.get("type") != "input_text" or not isinstance(part.get("text"), str):
                        raise ValueError("multimodal Responses input is not supported")
                    parts.append({"type": "text", "text": part["text"]})
                item.pop("type", None)
                item.pop("status", None)
                messages.append({**item, "content": parts})
            elif role == "assistant" and kind == "message":
                parts = []
                for part in content or []:
                    if part.get("type") in {"output_text", "input_text"} and isinstance(part.get("text"), str):
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                        parts.append({"type": "refusal", "refusal": part["refusal"]})
                    else:
                        raise ValueError("unsupported Responses assistant content")
                if merge:
                    previous = messages[-1]
                    if not isinstance(previous.get("content"), list):
                        previous["content"] = []
                    previous["content"].extend(parts)
                else:
                    item.pop("type", None)
                    item.pop("status", None)
                    messages.append({**item, "content": parts})
            elif kind == "function_call" and all(isinstance(item.get(k), str) for k in ("arguments", "call_id", "name")):
                call = {"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}
                if merge:
                    messages[-1].setdefault("tool_calls", []).append(call)
                else:
                    messages.append({"role": "assistant", "tool_calls": [call]})
            elif kind == "function_call_output" and isinstance(item.get("call_id"), str):
                output = item.get("output")
                if isinstance(output, list):
                    for part in output:
                        if part.get("type") != "input_text" or not isinstance(part.get("text"), str):
                            raise ValueError("Responses tool output must be text")
                        part["type"] = "text"
                elif not isinstance(output, str):
                    raise ValueError("Responses tool output must be text")
                messages.append({"role": "tool", "content": output, "tool_call_id": item["call_id"]})
            elif kind == "reasoning" and isinstance(item.get("summary"), list):
                if not isinstance(content, list) or not content or not isinstance(content[0].get("text"), str):
                    raise ValueError("source llama-server requires actual reasoning content, not just a summary")
                if merge:
                    messages[-1]["reasoning_content"] = content[0]["text"]
                else:
                    messages.append({"role": "assistant", "content": [], "reasoning_content": content[0]["text"]})
            else:
                raise ValueError("unsupported Responses input item")
    else:
        raise ValueError("Responses input must be text or a list")
    result["messages"] = messages
    if "tools" in result:
        tools = result.pop("tools")
        if not isinstance(tools, list):
            raise ValueError("Responses tools must be an array")
        converted = []
        for tool in tools:
            if tool.get("type") != "function":
                raise ValueError("non-function Responses tools need a separate adapter")
            tool.pop("type")
            tool.setdefault("strict", True)
            converted.append({"type": "function", "function": tool})
        if converted:
            result["tools"] = converted
    if "max_output_tokens" in result:
        result["max_tokens"] = result.pop("max_output_tokens")
    if "reasoning" in result:
        reasoning = result.pop("reasoning")
        if "effort" in reasoning:
            result["reasoning_effort"] = reasoning["effort"]
    return result
