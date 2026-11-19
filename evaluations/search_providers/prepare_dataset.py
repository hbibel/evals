"""Prepare replay scenarios from Bifrost logs and retained LibreChat history."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import polars as pl
from bson import decode
from bson.errors import InvalidBSON

from evals_lib.load_bifrost import (
    DEFAULT_DB,
    SEARCH_TOOL,
    DatasetError,
    Request,
    extract_rows,
    read_extract,
)

DEFAULT_OUT = Path("data/search_provider_scenarios.jsonl")
ARCHIVE_MAGIC = b"\x6d\xe2\x99\x81"
BLOCK_END = b"\xff" * 4
ROOT = "00000000-0000-0000-0000-000000000000"
Document = dict[str, Any]
MessageKey = tuple[str, str]


def archive_block(stream: gzip.GzipFile) -> Iterator[bytes]:
    while True:
        prefix = stream.read(4)
        if prefix == BLOCK_END:
            return
        size = int.from_bytes(prefix, "little")
        if len(prefix) != 4 or not 5 <= size <= 16 * 1024 * 1024:
            raise DatasetError("invalid BSON length in LibreChat archive")
        body = stream.read(size - 4)
        if len(body) != size - 4 or body[-1] != 0:
            raise DatasetError("invalid BSON record in LibreChat archive")
        yield prefix + body


def read_librechat_archive(
    path: Path, collections: dict[str, list[Document]]
) -> set[str]:
    databases: set[str] = set()
    with gzip.open(path, "rb") as stream:
        if stream.read(4) != ARCHIVE_MAGIC:
            raise DatasetError("expected a gzipped mongodump archive")
        # The prelude is an archive header and collection metadata block.
        for _ in archive_block(stream):
            pass
        while stream.peek(1):
            block = archive_block(stream)
            header = next(block, None)
            if header is None:
                raise DatasetError(
                    "missing namespace header in LibreChat archive"
                )
            namespace = decode(header)
            target = collections.get(namespace["collection"])
            for raw in block:
                if target is not None:
                    databases.add(namespace["db"])
                    target.append(decode(raw))
    return databases


def read_librechat(
    path: Path, conversations_path: Path | None = None
) -> tuple[list[Document], list[Document]]:
    messages: list[Document] = []
    conversations: list[Document] = []
    if conversations_path is None:
        databases = read_librechat_archive(
            path, {"messages": messages, "conversations": conversations}
        )
    else:
        databases = read_librechat_archive(path, {"messages": messages})
        databases |= read_librechat_archive(
            conversations_path, {"conversations": conversations}
        )
    if len(databases) > 1:
        raise DatasetError("LibreChat collections span multiple databases")
    if not messages or not conversations:
        raise DatasetError(
            "LibreChat archive needs messages and conversations"
        )
    return messages, conversations


def search_turns(
    requests: list[Request],
) -> tuple[dict[str, Request], pl.DataFrame]:
    turns: dict[str, Request] = {}
    rows: list[tuple[str, str]] = []
    for request in requests:
        call_ids = []
        for call in request.search_calls:
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise DatasetError(
                    f"row {request.id}: web_search call without an ID"
                )
            call_ids.append(call_id)
        if not call_ids:
            continue
        if len(set(call_ids)) != len(call_ids):
            raise DatasetError(f"row {request.id}: duplicate search call ID")
        scenario_id = hashlib.sha1(
            "\n".join(sorted(call_ids)).encode("utf-8")
        ).hexdigest()
        if scenario_id not in turns:
            turns[scenario_id] = request
            rows.extend((scenario_id, call_id) for call_id in call_ids)
    return turns, pl.DataFrame(
        rows,
        schema={"scenario_id": pl.String, "call_id": pl.String},
        orient="row",
    )


def librechat_searches(
    messages: list[Document],
) -> tuple[pl.DataFrame, dict[tuple[str, str, str], Any]]:
    rows = []
    results = {}
    for message in messages:
        if message["isCreatedByUser"]:
            continue
        conversation_id, message_id = message_key(message)
        for block in message.get("content") or []:
            call = block.get("tool_call")
            if not call or call.get("name") != SEARCH_TOOL:
                continue
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id:
                rows.append((call_id, conversation_id, message_id))
                results[conversation_id, message_id, call_id] = call.get(
                    "output"
                )
    return pl.DataFrame(
        rows,
        schema={
            "call_id": pl.String,
            "conversation_id": pl.String,
            "message_id": pl.String,
        },
        orient="row",
    ), results


def match_turns(
    expected: pl.DataFrame, observed: pl.DataFrame
) -> pl.DataFrame:
    target = pl.struct("conversation_id", "message_id")
    matched = (
        expected.join(observed, on="call_id", how="inner")
        .group_by("scenario_id")
        .agg(
            pl.len().alias("joined"),
            pl.col("call_id").n_unique().alias("matched"),
            target.n_unique().alias("targets"),
            target.first().alias("target"),
        )
    )
    return (
        expected.group_by("scenario_id")
        .len(name="expected")
        .join(matched, on="scenario_id", how="left")
        .with_columns(
            pl.when(pl.col("joined").is_null())
            .then(pl.lit("unmatched"))
            .when(
                (pl.col("targets") != 1)
                | (pl.col("joined") != pl.col("matched"))
            )
            .then(pl.lit("ambiguous"))
            .when(pl.col("matched") != pl.col("expected"))
            .then(pl.lit("partial"))
            .otherwise(pl.lit("retained"))
            .alias("status")
        )
    )


def message_key(message: Document) -> MessageKey:
    return message["conversationId"], message["messageId"]


def message_chain(
    messages: dict[MessageKey, Document], key: MessageKey
) -> list[Document]:
    chain = []
    visited: set[MessageKey] = set()
    while key[1] != ROOT:
        message = messages.get(key)
        if message is None or key in visited:
            return []
        visited.add(key)
        chain.append(message)
        parent = message.get("parentMessageId")
        if parent is None:
            break
        key = key[0], parent
    return chain[::-1]


def original_branch(
    messages: dict[MessageKey, Document],
    leaves: list[MessageKey],
    matched: MessageKey,
    last_response: str | None,
) -> tuple[list[Document], str]:
    if last_response:
        chain = message_chain(messages, (matched[0], last_response))
        if any(message_key(message) == matched for message in chain):
            return chain, "resolved"
    candidates = []
    for leaf in leaves:
        chain = message_chain(messages, leaf)
        if any(message_key(message) == matched for message in chain):
            candidates.append(chain)
    if len(candidates) == 1:
        return candidates[0], "resolved"
    prefix = message_chain(messages, matched)
    return prefix, "ambiguous_branch" if prefix else "missing_branch"


def openai_messages(message: Document) -> Iterator[Document]:
    if message["isCreatedByUser"]:
        yield {"role": "user", "content": message.get("text", "")}
        return

    for block in message.get("content") or []:
        if call := block.get("tool_call"):
            arguments = call.get("args")
            yield {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
            if call.get("output") is not None:
                yield {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": call["output"],
                }
        elif block.get("type") == "text" and block.get("text"):
            yield {"role": "assistant", "content": block["text"]}
    if not message.get("content") and message.get("text"):
        yield {"role": "assistant", "content": message["text"]}


def incomplete(message: Document) -> bool:
    return bool(message.get("unfinished") or message.get("error"))


def prepare(
    requests: list[Request],
    messages: list[Document],
    conversations: list[Document],
) -> tuple[list[Document], Counter[str]]:
    turns, expected = search_turns(requests)
    observed, results = librechat_searches(messages)
    matches = match_turns(expected, observed)
    coverage = Counter(matches["status"].to_list())
    by_key = {message_key(message): message for message in messages}
    if len(by_key) != len(messages):
        raise DatasetError("duplicate LibreChat message ID in a conversation")
    parents = {
        (message["conversationId"], message.get("parentMessageId"))
        for message in messages
    }
    leaves = defaultdict(list)
    for key in by_key.keys() - parents:
        leaves[key[0]].append(key)
    last_responses = {
        convo["conversationId"]: convo.get("lastResponseMessageId")
        for convo in conversations
    }

    records = []
    for match in matches.filter(pl.col("status") == "retained").iter_rows(
        named=True
    ):
        scenario_id = match["scenario_id"]
        request = turns[scenario_id]
        target = match["target"]
        conversation_id = target["conversation_id"]
        assistant_id = target["message_id"]
        chain, status = original_branch(
            by_key,
            leaves[conversation_id],
            (conversation_id, assistant_id),
            last_responses.get(conversation_id),
        )
        converted = [list(openai_messages(message)) for message in chain]
        original = [part for parts in converted for part in parts]
        if status == "resolved":
            terminal = converted[-1]
            if incomplete(chain[-1]):
                status = "unfinished"
            elif (
                terminal
                and terminal[-1]["role"] == "assistant"
                and terminal[-1].get("content", "").strip()
            ):
                status = "complete"
            else:
                status = "missing_answer"
        record = {
            "id": scenario_id,
            "created_at": request.timestamp,
            "provider": request.provider,
            "model": request.model,
            "history": request.messages_with_output(),
            "conversationId": conversation_id,
            "assistantMessageId": assistant_id,
            "originalSearchResults": [
                {
                    "tool_call_id": call["id"],
                    "output": results[
                        conversation_id, assistant_id, call["id"]
                    ],
                }
                for call in request.search_calls
            ],
            "originalAnswerStatus": status,
            "originalConversationHasIncompleteMessages": any(
                map(incomplete, chain)
            ),
        }
        if original:
            record["originalConversation"] = original
        records.append(record)
    return sorted(
        records, key=lambda row: (row["created_at"], row["id"])
    ), coverage


def write_dataset(path: Path, records: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete_on_close=False
    ) as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.close()
        Path(output.name).replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    librechat = parser.add_mutually_exclusive_group(required=True)
    librechat.add_argument("--librechat", type=Path, help="combined archive")
    librechat.add_argument(
        "--librechat-collections",
        type=Path,
        nargs=2,
        metavar=("MESSAGES", "CONVERSATIONS"),
        help="separate gzipped mongodump archives",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="maximum scenarios to write")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit requires a positive integer")
    librechat_paths = args.librechat_collections or [args.librechat]
    try:
        with TemporaryDirectory(prefix="bifrost-") as directory:
            extract = Path(directory) / "logs.jsonl"
            extract_rows(args.db, extract)
            requests = read_extract(extract)
        records, coverage = prepare(
            requests, *read_librechat(*librechat_paths)
        )
        excluded = ", ".join(
            f"{coverage[reason]} {reason}"
            for reason in ("unmatched", "partial", "ambiguous")
        )
        print(
            f"Excluded: {sum(coverage.values()) - len(records)} ({excluded})"
        )
        if not records:
            raise DatasetError("no replay scenarios matched both sources")
        if args.limit is not None and len(records) > args.limit:
            print(
                f"Selecting {args.limit} of {len(records)} eligible scenarios"
            )
            # Scenario IDs are hashes; sampling by ID avoids favoring early turns.
            ids = set(sorted(record["id"] for record in records)[: args.limit])
            records = [record for record in records if record["id"] in ids]
        write_dataset(args.out, records)
    except (DatasetError, InvalidBSON, KeyError, OSError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    conversations = {row["conversationId"] for row in records}
    calls = sum(len(row["originalSearchResults"]) for row in records)
    answers = Counter(row["originalAnswerStatus"] for row in records)
    incomplete_conversations = {
        row["conversationId"]
        for row in records
        if row["originalConversationHasIncompleteMessages"]
    }
    print(
        f"Retained: {len(records)} scenarios / {len(conversations)} "
        f"conversations / {calls} calls"
    )
    print(
        "Original answers: "
        + ", ".join(
            f"{count} {status}" for status, count in sorted(answers.items())
        )
    )
    print(
        f"Conversations with unfinished/error messages: {len(incomplete_conversations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
