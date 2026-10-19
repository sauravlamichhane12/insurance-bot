from __future__ import annotations

import argparse
import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import get_rag_service  # noqa: E402

DEFAULT_QUESTION = (
    "If I am on Plan 2M and I get admitted to the hospital, then also need a private duty nurse "
    "and a non-surgical physician visit on the same confinement, which items are as charged and "
    "which ones have daily caps?"
)


@dataclass
class TokenUsageEntry:
    input_tokens: int
    output_tokens: int
    total_tokens: int


def _message_role(message: Any) -> str:
    role = getattr(message, "role", None)
    if isinstance(role, str) and role:
        return role.lower()
    if isinstance(message, dict):
        dict_role = message.get("role")
        if isinstance(dict_role, str) and dict_role:
            return dict_role.lower()
    return ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return " ".join(part.strip() for part in parts if part.strip())
    if isinstance(message, dict):
        dict_content = message.get("content")
        if isinstance(dict_content, str):
            return dict_content.strip()
    return ""


def _usage_dict(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)
    if usage is None and isinstance(message, dict):
        usage = message.get("usage_metadata")
    if not isinstance(usage, dict):
        return {}
    normalized = {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    if not any(normalized.values()):
        return {}
    return normalized


def extract_token_usage_entries(agent_result: Any) -> list[TokenUsageEntry]:
    if hasattr(agent_result, "usage_metadata"):
        usage = _usage_dict(agent_result)
        return [
            TokenUsageEntry(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
            )
        ] if usage else []

    messages = agent_result.get("messages") if isinstance(agent_result, dict) else getattr(agent_result, "messages", None)
    messages = messages or []
    entries: list[TokenUsageEntry] = []
    for message in messages:
        usage = _usage_dict(message)
        if usage:
            entries.append(
                TokenUsageEntry(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                )
            )
    return entries


def summarize_token_usage(entries: list[TokenUsageEntry]) -> dict[str, int]:
    return {
        "llm_input_tokens": sum(entry.input_tokens for entry in entries),
        "llm_output_tokens": sum(entry.output_tokens for entry in entries),
        "llm_total_tokens": sum(entry.total_tokens for entry in entries),
    }


def extract_final_response_text(agent_result: Any) -> str:
    if hasattr(agent_result, "content") and isinstance(getattr(agent_result, "content"), str):
        text = getattr(agent_result, "content").strip()
        if text:
            return text

    messages = agent_result.get("messages") if isinstance(agent_result, dict) else getattr(agent_result, "messages", None)
    messages = messages or []
    for message in reversed(messages):
        if _message_role(message) in {"assistant", "ai", "aimessage", "aimessagechunk", "apimessage", "bot"}:
            text = _message_text(message)
            if text:
                return text
    if isinstance(agent_result, dict):
        fallback_output = agent_result.get("output")
        if isinstance(fallback_output, str):
            return fallback_output.strip()
    return ""


def _compact_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)].rstrip()}..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-query frontend-style system token test on a fresh thread."
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to send through the real app service path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = get_rag_service()

    stream_thread_id = f"frontend-stream-{secrets.token_urlsafe(6)}"
    usage_thread_id = f"frontend-usage-{secrets.token_urlsafe(6)}"
    route = service._route_query(args.question)

    stream_chunks: list[str] = []
    stream_metadata: list[dict[str, object]] = []
    for token, metadata in service.stream_answer(args.question, history=None, thread_id=stream_thread_id):
        stream_chunks.append(str(token))
        if isinstance(metadata, dict):
            stream_metadata.append(metadata)

    if route == "direct":
        context = service._retrieve_direct_context(args.question, max_results=2)
        direct_messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the provided MediLink document context. "
                    "If the answer is not present, say so. "
                    "Be concise and start with the answer."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {args.question}",
            },
        ]
        result = service.chat_model.invoke(direct_messages)
    else:
        # stream_answer() matches app behavior, but it does not expose usage_metadata.
        # Use a second fresh thread with the same message preparation path to get exact token counts.
        result = service.agent.invoke(
            {"messages": service._prepare_messages(args.question, None, thread_id=usage_thread_id)},
            config={"configurable": {"thread_id": usage_thread_id}},
        )

    entries = extract_token_usage_entries(result)
    summary = summarize_token_usage(entries)

    report = {
        "question": args.question,
        "stream_thread_id": stream_thread_id,
        "usage_thread_id": usage_thread_id,
        "stream": {
            "chunk_count": len(stream_chunks),
            "answer": "".join(stream_chunks),
            "metadata_nodes": sorted(
                {
                    metadata.get("langgraph_node")
                    for metadata in stream_metadata
                    if metadata.get("langgraph_node")
                }
            ),
            "metadata_has_usage": any(
                "usage_metadata" in metadata or "input_tokens" in metadata or "output_tokens" in metadata
                for metadata in stream_metadata
            ),
        },
        "usage_accounting": {
            "answer": extract_final_response_text(result),
            "turns_counted": len(entries),
            "llm_input_tokens": summary["llm_input_tokens"],
            "llm_output_tokens": summary["llm_output_tokens"],
            "llm_total_tokens": summary["llm_total_tokens"],
            "route": route,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 72)
    print("SINGLE QUERY SYSTEM TEST")
    print("=" * 72)
    print(f"Question: {report['question']}")
    print(f"Stream thread id: {stream_thread_id}")
    print(f"Usage thread id:  {usage_thread_id}")
    print(f"Route:            {route}")
    print("-" * 72)
    print("[STREAM PATH]")
    print(f"- chunk count:      {report['stream']['chunk_count']}")
    print(f"- metadata nodes:   {', '.join(report['stream']['metadata_nodes']) or 'none'}")
    print(f"- metadata has usage: {report['stream']['metadata_has_usage']}")
    print(f"- answer:\n{report['stream']['answer']}\n")
    print("[TOKEN ACCOUNTING]")
    print(f"- turns counted:    {report['usage_accounting']['turns_counted']}")
    print(f"- llm input tokens: {report['usage_accounting']['llm_input_tokens']}")
    print(f"- llm output tokens:{report['usage_accounting']['llm_output_tokens']}")
    print(f"- llm total tokens: {report['usage_accounting']['llm_total_tokens']}")
    print(f"- answer:\n{report['usage_accounting']['answer']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
