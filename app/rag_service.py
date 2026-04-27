from __future__ import annotations

import csv
import io
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bcrypt
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langgraph.checkpoint.memory import InMemorySaver
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Pinecone
from pypdf import PdfReader

from .db import CHECKPOINT_DB_PATH, ensure_dirs, fetch_one

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - depends on optional package availability
    SqliteSaver = None


ROOT = Path(__file__).resolve().parent.parent


def _trim(value: str | None, fallback: str = "") -> str:
    if value is None:
      return fallback
    text = value.strip()
    return text or fallback


def _safe_page_number(page_number: Any) -> int | None:
    try:
        page_num = int(page_number)
        return page_num if page_num > 0 else None
    except Exception:
        return None


def _normalize_chat_role(role: str | None) -> str | None:
    if role in {"user", "userMessage"}:
        return "user"
    if role in {"assistant", "apiMessage", "bot"}:
        return "assistant"
    return None


def _is_assistant_message(message: Any) -> bool:
    role = getattr(message, "role", None) or getattr(message, "type", None)
    if isinstance(role, str) and role.lower() in {"assistant", "ai", "aimessage", "aimessagechunk", "apimessage", "bot"}:
        return True
    if isinstance(message, dict):
        dict_role = message.get("role") or message.get("type")
        if isinstance(dict_role, str):
            return dict_role.lower() in {"assistant", "ai", "aimessage", "aimessagechunk", "apimessage", "bot"}
    return False


def _extract_text_blocks(message: Any) -> str:
    if isinstance(message, str):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        if parts:
            return "".join(parts)

    blocks = getattr(message, "content_blocks", None) or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        else:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if block_type == "text" and isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _chunk_text(text: str, size: int = 120) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if size <= 0:
        return [cleaned]

    chunks: list[str] = []
    current = ""
    for paragraph in cleaned.splitlines(keepends=True):
        if len(current) + len(paragraph) <= size:
            current += paragraph
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= size:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start : start + size])
            start += size
    if current:
        chunks.append(current)
    return chunks


def _estimate_tokens(text: str) -> int:
    """Estimate token count using the standard 1 token ~= 4 chars heuristic for English."""
    return math.ceil(len(text) / 4)


def _build_chat_messages(question: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    trimmed_history = (history or [])[-4:]
    for item in trimmed_history:
        role = _normalize_chat_role(item.get("role"))
        content = (item.get("content") or "").strip()
        if role and content:
            if len(content) > 600:
                content = f"{content[:597].rstrip()}..."
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _is_tabular(line: str) -> bool:
    """Determine if a line likely belongs to a table based on spacing or numeric density."""
    cleaned = line.strip()
    if not cleaned:
        return False

    # Universal Columnar Spacing: Multiple gaps of 3+ spaces indicate table columns
    # regardless of whether the content is numeric or textual.
    column_gaps = len(re.findall(r"\s{3,}", line))
    if column_gaps >= 2:
        return True

    tokens = cleaned.split()
    if not tokens:
        return False

    # High density of numeric tokens (Financial tables often have mixed text/numbers)
    numbers = sum(1 for t in tokens if any(c.isdigit() for c in t))
    return (numbers / len(tokens)) > 0.4 and len(tokens) > 2


def _is_list_item(line: str) -> bool:
    """Determine if a line starts with a common list marker (numeric, alpha, or bullet)."""
    cleaned = line.strip()
    # Universal Patterns: 
    # 1., 1), (1), a., A., (a), i., (iv), -, *, •, ▪
    markers = [
        r"^\d+[\.\)]",      # 1., 2)
        r"^[a-zA-Z][\.\)]",  # a., B)
        r"^[•\-\*▪]",        # bullets
        r"^\([a-zA-Z\d]+\)", # (1), (a), (iv)
        r"^[ivxlc]+[\.\)]",  # i., ii., iv)
    ]
    return any(re.match(p, cleaned, re.IGNORECASE) for p in markers)


def _split_semantic_units(text: str) -> list[str]:
    units: list[str] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        cleaned_line = _normalize_whitespace(line)
        if not cleaned_line:
            i += 1
            continue

        # If it's a list item or tabular, we might want to group it with context
        is_list = _is_list_item(line)
        is_tab = _is_tabular(line)

        # Track if the previous line was a list item to avoid merging consecutive ones
        prev_line_was_list = i > 0 and _is_list_item(lines[i-1])

        # Avoid splitting on periods that are part of a list marker (e.g., '1.', 'a.')
        if is_list:
            sentences = [cleaned_line]
        else:
            sentences = re.split(r"(?<=[.!?])\s+", cleaned_line)
        for sentence in sentences:
            cleaned_sentence = _normalize_whitespace(sentence)
            if cleaned_sentence:
                # Merge logic: keep list item with its non-list parent (e.g. "Coverage:")
                if is_list and units and not units[-1].endswith((".", "!", "?")) and not prev_line_was_list:
                    units[-1] = f"{units[-1]} {cleaned_sentence}"
                else:
                    units.append(cleaned_sentence)
        i += 1

    if units:
        return units
    cleaned_text = _normalize_whitespace(text)
    return [cleaned_text] if cleaned_text else []


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        numerator += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0 or right_norm <= 0:
        return 1.0
    cosine_similarity = numerator / (math.sqrt(left_norm) * math.sqrt(right_norm))
    cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)
    return 1.0 - cosine_similarity


def _truncate_title(text: str, limit: int = 80) -> str:
    cleaned = _normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned or "Semantic Chunk"
    return f"{cleaned[: limit - 3].rstrip()}..."


def _csv_row_to_text(headers: list[str], row: list[str]) -> str:
    pairs: list[str] = []
    for index, value in enumerate(row):
        cell = _normalize_whitespace(value)
        if not cell:
            continue
        header = headers[index] if index < len(headers) else f"column_{index + 1}"
        pairs.append(f"{header}: {cell}")
    return "; ".join(pairs)


def _normalize_metadata_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"
    return cleaned or "field"


def _tokenize_for_match(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (text or "").lower())


def _extract_query_keywords(query: str) -> list[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "basis", "benefit", "benefits", "by", "compare",
        "covered", "coverage", "daily", "difference", "does", "for", "get", "have",
        "how", "i", "if", "in", "include", "included", "is", "limit", "limits", "m",
        "me", "of", "on", "or", "plan", "same", "show", "the", "then", "to", "under",
        "visit", "what", "which", "with", "year",
    }
    return [
        token for token in _tokenize_for_match(query)
        if token not in stopwords and len(token) > 2
    ]


def _extract_plan_terms(query: str) -> set[str]:
    normalized = (query or "").lower()
    matches = set(re.findall(r"\bplan\s*([0-9]+(?:\.[0-9]+)?m)\b", normalized))
    compact_matches = set(re.findall(r"\b([0-9]+(?:\.[0-9]+)?m)\b", normalized))
    return {f"plan {value}" for value in matches | compact_matches}


def _is_csv_metadata_row(metadata: dict[str, Any]) -> bool:
    source = (
        metadata.get("mime_type")
        or metadata.get("source")
        or metadata.get("original_name")
        or ""
    )
    if str(source).lower() == "text/csv":
        return True
    if str(source).lower().endswith(".csv"):
        return True
    return "benefit_name" in metadata or "sub_benefit" in metadata


def _score_csv_metadata_match(query: str, metadata: dict[str, Any], content: str) -> tuple[float, int]:
    query_lower = query.lower()
    keywords = _extract_query_keywords(query)
    plan_terms = _extract_plan_terms(query)

    benefit_text = " ".join(
        _normalize_whitespace(str(metadata.get(key, "")))
        for key in ("benefit_name", "sub_benefit", "benefit_type")
    ).lower()
    auxiliary_text = " ".join(
        _normalize_whitespace(str(metadata.get(key, "")))
        for key in ("limit_type", "limit_cycle")
    ).lower()
    full_text = _normalize_whitespace(content).lower()

    benefit_tokens = set(_tokenize_for_match(benefit_text))
    auxiliary_tokens = set(_tokenize_for_match(auxiliary_text))
    full_tokens = set(_tokenize_for_match(full_text))

    score = 0.0
    benefit_hits = 0
    for token in keywords:
        if token in benefit_tokens:
            score += 4.0
            benefit_hits += 1
        elif token in auxiliary_tokens:
            score += 2.0
        elif token in full_tokens:
            score += 1.0

    if benefit_hits >= 2:
        score += 4.0

    for phrase in (
        _normalize_whitespace(str(metadata.get("benefit_name", ""))).lower(),
        _normalize_whitespace(str(metadata.get("sub_benefit", ""))).lower(),
    ):
        if phrase and len(phrase) > 3 and phrase in query_lower:
            score += 8.0

    requested_plan_keys = {_normalize_metadata_key(plan) for plan in plan_terms}
    if requested_plan_keys:
        available_plans = {
            key for key, value in metadata.items()
            if key.startswith("plan_") and str(value).strip()
        }
        matched_plan_keys = requested_plan_keys & available_plans
        score += len(matched_plan_keys) * 3.0

    row_number = _safe_page_number(metadata.get("row_number") or metadata.get("page_number")) or 10**9
    return score, row_number


def _format_csv_metadata_row(query: str, metadata: dict[str, Any], fallback_text: str) -> str:
    requested_plan_keys = {_normalize_metadata_key(plan) for plan in _extract_plan_terms(query)}
    parts: list[str] = []

    for key, label in (
        ("benefit_type", "Benefit Type"),
        ("benefit_name", "Benefit Name"),
        ("sub_benefit", "Sub Benefit"),
        ("limit_type", "Limit Type"),
        ("limit_cycle", "Limit Cycle"),
    ):
        value = _normalize_whitespace(str(metadata.get(key, "")))
        if not value or (key == "sub_benefit" and value.lower() == "na"):
            continue
        parts.append(f"{label}: {value}")

    plan_fields = [
        (key, value)
        for key, value in metadata.items()
        if key.startswith("plan_") and not key.endswith("_norm") and str(value).strip()
    ]
    for key, value in sorted(plan_fields):
        if requested_plan_keys and key not in requested_plan_keys:
            continue
        label = key.replace("_", " ").title().replace("Plan ", "Plan ")
        parts.append(f"{label}: {value}")

    return "; ".join(parts) if parts else fallback_text


def _is_grouped_csv_query(query: str) -> bool:
    normalized = (query or "").lower()
    markers = (
        "what is included",
        "what's included",
        "what includes",
        "included in",
        "includes under",
        "included under",
        "sub benefits",
        "sub-benefits",
        "complete list",
        "line items",
    )
    return any(marker in normalized for marker in markers)


def _group_csv_rows_by_benefit_name(
    query: str,
    scored_csv_rows: list[tuple[tuple[float, int], Document, float]],
    fallback_limit: int,
) -> tuple[list[tuple[Document, float]], bool, str | None]:
    if not scored_csv_rows:
        return [], False, None

    best_doc = scored_csv_rows[0][1]
    best_meta = best_doc.metadata or {}
    target_benefit_name = _normalize_whitespace(str(best_meta.get("benefit_name", "")))
    if not target_benefit_name:
        return [(doc, score) for _match, doc, score in scored_csv_rows[:fallback_limit]], False, None

    top_csv_score = scored_csv_rows[0][0][0]
    repeated_benefit_rows = [
        (doc, score)
        for (match_score, _row_number), doc, score in scored_csv_rows
        if (
            _normalize_whitespace(str((doc.metadata or {}).get("benefit_name", ""))) == target_benefit_name
            and match_score >= max(2.0, top_csv_score - 4.0)
        )
    ]

    should_group = _is_grouped_csv_query(query) or len(repeated_benefit_rows) >= 3
    if not should_group:
        selected_results = [
            (doc, score)
            for (_match_score, _row_number), doc, score in scored_csv_rows
        ]
        selected_results.sort(
            key=lambda item: (
                -_score_csv_metadata_match(query, item[0].metadata or {}, item[0].page_content)[0],
                -item[1],
                _safe_page_number(item[0].metadata.get("row_number") or item[0].metadata.get("page_number")) or 10**9,
            )
        )
        selected_results = selected_results[:fallback_limit]
        return selected_results, False, None

    grouped_rows = [
        (doc, score)
        for (_match, doc, score) in scored_csv_rows
        if _normalize_whitespace(str((doc.metadata or {}).get("benefit_name", ""))) == target_benefit_name
    ]
    if not grouped_rows:
        return [(doc, score) for _match, doc, score in scored_csv_rows[:fallback_limit]], False, None

    return grouped_rows, True, target_benefit_name


def _read_csv_rows(file_path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with file_path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.reader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV encoding is not supported")


@dataclass
class RAGConfig:
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    bedrock_evaluator_model_id: str
    bedrock_embedding_model_id: str
    semantic_max_chars: int = 1200


class RAGService:
    def __init__(self, config: RAGConfig):
        self.config = config
        ensure_dirs()
        self.pinecone = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pinecone.Index(config.pinecone_index_name)
        self.checkpoint_conn: sqlite3.Connection | None = None
        self.memory_backend = "memory"
        if SqliteSaver is not None:
            self.checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
            self.checkpointer = SqliteSaver(self.checkpoint_conn)
            self.memory_backend = "sqlite"
        else:
            self.checkpointer = InMemorySaver()
        print(
            f"RAG memory backend: {self.memory_backend}"
            + (f" ({CHECKPOINT_DB_PATH})" if self.memory_backend == "sqlite" else " (non-persistent fallback)")
        )
        self.embeddings = BedrockEmbeddings(
            model_id=config.bedrock_embedding_model_id,
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=config.pinecone_namespace,
        )
        self.chat_model = ChatBedrockConverse(
            model_id=config.bedrock_evaluator_model_id,
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        self.agent = self._build_agent()

    def _build_agent(self):
        @tool
        def retrieve_document_context(query: str, document_id: str | None = None, max_results: int = 4) -> str:
            """Retrieve the most relevant context using a Hybrid (Semantic + Keyword) search strategy."""
            # 1. PREPARE FILTERS (Hard Filtering)
            metadata_filter: dict[str, Any] = {}
            if document_id:
                metadata_filter["document_id"] = {"$eq": str(document_id)}

            plan_terms = _extract_plan_terms(query)
            if plan_terms:
                plan_filters = []
                for plan in plan_terms:
                    key = _normalize_metadata_key(plan)
                    plan_filters.append({key: {"$exists": True}})
                
                if len(plan_filters) == 1:
                    metadata_filter.update(plan_filters[0])
                elif len(plan_filters) > 1:
                    metadata_filter["$or"] = plan_filters

            # 2. SEMANTIC SEARCH (Dense Retrieval)
            # We fetch more candidates than needed (up to 50) so we can rerank them using keyword matches
            is_group_query = _is_grouped_csv_query(query)
            candidate_count = 20 if is_group_query else 15

            
            results = self.vector_store.similarity_search_with_score(
                query,
                k=candidate_count,
                filter=metadata_filter if metadata_filter else None,
            )
            
            if not results:
                return "No relevant passages were found in the uploaded documents."

            # 3. HYBRID RERANKING (Applying "Sparse" Keyword Logic)
            # We split results into CSV and Non-CSV to apply different scoring logic
            csv_matches = []
            non_csv_matches = []
            
            for doc, vector_score in results:
                if _is_csv_metadata_row(doc.metadata or {}):
                    # Keyword boost: exact matches in metadata get a significant score multiplier
                    keyword_score, row_num = _score_csv_metadata_match(query, doc.metadata or {}, doc.page_content)
                    # Combined Score = (Semantic Similarity * 10) + Keyword Match Score
                    hybrid_score = (vector_score * 10.0) + keyword_score
                    csv_matches.append((hybrid_score, row_num, doc, vector_score, keyword_score))
                else:
                    non_csv_matches.append((doc, vector_score))

            # Sort CSV rows by our Hybrid score
            csv_matches.sort(key=lambda x: (-x[0], x[1])) # High score first, then by row number
            
            # 4. GROUPING & SELECTION (Structural Retrieval)
            selected_results: list[tuple[Document, float]]
            retrieval_scope = "partial"
            grouped_benefit_name: str | None = None

            if csv_matches:
                # Re-package for grouping: requires ((keyword_score, row_num), doc, vector_score)
                scored_for_grouping = [
                    ((m[4], m[1]), m[2], m[3])
                    for m in csv_matches
                ]
                
                selected_results, used_grouping, grouped_benefit_name = _group_csv_rows_by_benefit_name(
                    query,
                    scored_for_grouping,
                    max_results,
                )
                retrieval_scope = "complete" if used_grouping else "partial"
                
                # If we still have room, add non-CSV semantic results
                if not used_grouping and len(selected_results) < max_results:
                    selected_results.extend(non_csv_matches[: max_results - len(selected_results)])
            else:
                selected_results = results[:max_results]

            lines: list[str] = []
            if grouped_benefit_name:
                lines.append(f"Retrieved benefit group: {grouped_benefit_name}")
                lines.append(f"Coverage scope: {retrieval_scope}")
            
            for rank, (doc, _score) in enumerate(selected_results, start=1):
                meta = doc.metadata or {}
                source = meta.get("original_name") or meta.get("source") or "document"
                page = _safe_page_number(meta.get("page_number") or meta.get("page"))
                header = f"{rank}. {source}"
                if page is not None:
                    header += f" row/page {page}"
                
                content = doc.page_content.strip()
                if _is_csv_metadata_row(meta):
                    content = _format_csv_metadata_row(query, meta, content)
                    
                lines.append(f"{header}: {content}")
                
            return "\n\n".join(lines)

        return create_agent(
            model=self.chat_model,
            tools=[retrieve_document_context],
            checkpointer=self.checkpointer,
            system_prompt=("""
    Answer based only on retrieved context. Do not use outside knowledge or fill gaps with inference.

    - Direct match: Answer clearly and precisely.
    - No match: If the specific term is not found, say it is not explicitly mentioned in the documents.
    - Unnamed item: Do not infer coverage from a related category. Do not attempt to categorize items using outside knowledge.

    Lists: Present only retrieved line items. If completeness is uncertain, say "the retrieved rows show." Do not add catch-all items.

    Comparisons: Use only what is explicitly shown. Preserve distinctions (plan limit, inner limit, daily cap, per-visit, "As Charged," etc.). Do not convert plan names to amounts unless stated.

    Never: Narrate your process, expose tool usage, or say "let me check."
""")
    )

    def _extract_pages(self, file_path: Path) -> list[Document]:
        reader = PdfReader(str(file_path))
        documents: list[Document] = []
        total_pages = len(reader.pages)
        
        # Define universal header patterns for "Markdownification"
        header_patterns = [
            (r"^\s*(Chapter|Section|Article|Part)\s+[A-Z\d\.]+", 1), # Level 1
            (r"^\s*[IVXLC\d]+\.[A-Z\d\.]+\s+[A-Z]", 2),              # Level 2
            (r"^\s*[A-Z][\s\w]{2,50}:$", 3),                         # Level 3
        ]

        for page_number, page in enumerate(reader.pages, start=1):
            # Use 'layout' mode to preserve horizontal alignment in tables
            try:
                raw_text = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                # Fallback if layout mode fails
                raw_text = page.extract_text() or ""
                
            lines = raw_text.splitlines()
            markdown_lines = []
            for line in lines:
                cleaned = line.strip()
                if not cleaned:
                    continue
                
                # Check for header matches to prepend # markers
                is_header = False
                for pattern, level in header_patterns:
                    if re.match(pattern, cleaned, re.IGNORECASE):
                        markdown_lines.append(f"{'#' * level} {cleaned}")
                        is_header = True
                        break
                
                if not is_header:
                    markdown_lines.append(line)
            
            text = "\n".join(markdown_lines).strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_number,
                        "total_pages": total_pages,
                        "source": file_path.name
                    },
                )
            )
        return documents

    def _extract_csv_rows(self, file_path: Path) -> list[Document]:
        documents: list[Document] = []
        rows = _read_csv_rows(file_path)

        if not rows:
            return documents

        # Find the header row (sometimes CSVs have leading empty rows or title rows)
        header_index = 0
        for i, row in enumerate(rows):
            if any(cell.strip() for cell in row):
                header_index = i
                break

        headers = [_normalize_whitespace(cell) or f"column_{index + 1}" for index, cell in enumerate(rows[header_index])]
        data_rows = rows[header_index + 1:]
        carry_forward = [""] * len(headers)
        for row_number, row in enumerate(data_rows, start=header_index + 2):
            normalized_row = list(row)
            while len(normalized_row) < len(headers):
                normalized_row.append("")

            for index, value in enumerate(normalized_row):
                cell = _normalize_whitespace(value)
                if cell:
                    normalized_row[index] = cell
                    carry_forward[index] = cell
                elif index < min(3, len(carry_forward)):
                    normalized_row[index] = carry_forward[index]

            content = _csv_row_to_text(headers, normalized_row)
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "row_number": row_number,
                        "source": file_path.name,
                        **{
                            _normalize_metadata_key(headers[index]): normalized_row[index]
                            for index in range(len(headers))
                            if normalized_row[index]
                        }
                    },
                )
            )
        return documents

    def _extract_source_documents(self, file_path: Path) -> list[Document]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pages(file_path)
        if suffix == ".csv":
            return self._extract_csv_rows(file_path)
        raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")

    def load_and_split_document(self, file_path: str | Path, base_metadata: dict[str, Any]) -> list[Document]:
        path = Path(file_path)
        source_documents = self._extract_source_documents(path)
        if not source_documents:
            raise ValueError("No extractable text was found in the uploaded document")
        if path.suffix.lower() == ".csv":
            return [Document(page_content=doc.page_content, metadata={**base_metadata, **(doc.metadata or {})}) for doc in source_documents]
        return self._build_semantic_chunks(source_documents, base_metadata)

    def _build_semantic_chunks(self, page_documents: list[Document], base_metadata: dict[str, Any]) -> list[Document]:
        # Tier 1: Markdown Header Splitting (Structural Hierarchy)
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
        # Combine all pages into one text for proper header hierarchy tracking
        full_text = "\n\n".join(doc.page_content for doc in page_documents)
        md_sections = md_splitter.split_text(full_text)
        
        # Tier 2 & 3: Semantic and Recursive (Meaning & Safety)
        # We increase chunk size slightly to ensure table rows stay together
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600, 
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        final_chunks: list[Document] = []
        chunk_index = 0
        
        for section in md_sections:
            section_meta = dict(base_metadata)
            section_meta.update(section.metadata)
            
            header_path = " > ".join(
                section.metadata.get(f"Header {i}") 
                for i in range(1, 4) 
                if section.metadata.get(f"Header {i}")
            )
            
            # Use a simpler but more robust splitting strategy for table-heavy docs
            # Layout mode already provides good visual structure
            sub_chunks = recursive_splitter.split_text(section.page_content)
            
            for sub_part in sub_chunks:
                metadata = dict(section_meta)
                metadata["chunk_index"] = chunk_index
                metadata["chunk_strategy"] = "layout_aware_recursive"
                metadata["section_title"] = header_path or _truncate_title(sub_part)
                
                final_chunks.append(Document(page_content=sub_part, metadata=metadata))
                chunk_index += 1
                    
        return final_chunks

    def _build_model_semantic_parts(self, text: str) -> list[str]:
        units = _split_semantic_units(text)
        if not units:
            return []
        if len(units) == 1:
            return units

        # Identify which units look tabular or list-like for later bias
        is_tabular_unit = [_is_tabular(u) for u in units]
        is_list_unit = [_is_list_item(u) for u in units]

        embeddings = self.embeddings.embed_documents(units)
        distances: list[float] = []
        for index in range(len(embeddings) - 1):
            distances.append(_cosine_distance(embeddings[index], embeddings[index + 1]))

        if distances:
            average_distance = sum(distances) / len(distances)
            variance = sum((distance - average_distance) ** 2 for distance in distances) / len(distances)
            # Use a slightly more conservative threshold for insurance documents
            breakpoint_threshold = average_distance + math.sqrt(variance) * 0.7
        else:
            breakpoint_threshold = 1.0

        parts: list[str] = []
        current_units: list[str] = []
        current_length = 0

        for index, unit in enumerate(units):
            unit_length = len(unit)
            next_length = current_length + unit_length + (1 if current_units else 0)
            should_split_for_size = bool(current_units) and next_length > self.config.semantic_max_chars

            should_split_for_distance = False
            if current_units and index > 0 and index - 1 < len(distances):
                distance = distances[index - 1]

                # Tabular Bias: If both units are tabular, increase threshold (stay together)
                effective_threshold = breakpoint_threshold
                if is_tabular_unit[index] and is_tabular_unit[index - 1]:
                    effective_threshold *= 1.5
                # List Bias: If current is a list item, stay with previous
                elif is_list_unit[index]:
                    effective_threshold *= 1.3

                should_split_for_distance = distance > effective_threshold

            if should_split_for_size or should_split_for_distance:
                parts.append(" ".join(current_units))
                current_units = []
                current_length = 0

            current_units.append(unit)
            current_length += unit_length + (1 if current_units else 0)

        if current_units:
            parts.append(" ".join(current_units))

        return [_normalize_whitespace(part) for part in parts if _normalize_whitespace(part)]

    def index_document(self, document_row: dict[str, Any]) -> dict[str, Any]:
        chunks = self.load_and_split_document(
            document_row["stored_path"],
            {
                "document_id": str(document_row["id"]),
            },
        )

        ids = [f"doc-{document_row['id']}-chunk-{index}" for index, _ in enumerate(chunks)]
        self.vector_store.add_documents(chunks, ids=ids)
        return {"chunk_count": len(chunks), "ids": ids}

    def delete_document_vectors(self, document_id: int | str) -> None:
        self.index.delete(
            namespace=self.config.pinecone_namespace,
            filter={"document_id": {"$eq": str(document_id)}},
        )

    def _prepare_messages(self, question: str, history: list[dict[str, str]] | None, thread_id: str | None = None) -> list[dict[str, str]]:
        """
        Formats history into messages. If thread_id is present, the checkpointer already 
        handles persistence, so we only pass the new question if possible, or format the 
        history once if it's the first time.
        """
        # If we have a thread ID and the checkpointer is active, LangGraph handles history.
        # We only need to provide the new message.
        if thread_id and self.memory_backend == "sqlite":
            return [{"role": "user", "content": question}]

        if not history:
            return [{"role": "user", "content": question}]

        messages: list[dict[str, str]] = []
        # Fallback for stateless calls: take last 10 messages max to prevent initial bloat
        trimmed_history = history[-10:]
        
        for item in trimmed_history:
            role = _normalize_chat_role(item.get("role"))
            content = (item.get("content") or "").strip()
            if role and content:
                if len(content) > 1000:
                    content = f"{content[:997].rstrip()}..."
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})
        return messages

    def stream_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        thread_id: str | None = None,
    ):
        messages = self._prepare_messages(question, history, thread_id=thread_id)
        emitted_text = False
        config: dict[str, Any] | None = None
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        try:
            stream = self.agent.stream({"messages": messages}, config=config, stream_mode="messages")
            for token, metadata in stream:
                if not _is_assistant_message(token):
                    continue
                text = _extract_text_blocks(token)
                if not text:
                    continue
                # FILTER: Skip internal summarization leak
                if "SESSION INTENT" in text or "NEXT STEPS" in text:
                    continue
                emitted_text = True
                yield text, metadata
            return
        except Exception:
            if emitted_text:
                raise

        answer = self.answer(question, history, thread_id=thread_id)
        for chunk in _chunk_text(answer, size=120):
            yield chunk, {}

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        thread_id: str | None = None,
    ) -> str:
        messages = self._prepare_messages(question, history, thread_id=thread_id)
        config: dict[str, Any] | None = None
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": messages}, config=config)

        if isinstance(result, dict):
            response_messages = result.get("messages") or []
            for message in reversed(response_messages):
                if not _is_assistant_message(message):
                    continue
                content = _extract_text_blocks(message).strip()
                if content:
                    return content
            content = result.get("output")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return _extract_text_blocks(result).strip() if result is not None else ""
