from evals_lib.load_bifrost import Request
from evaluations.search_providers.prepare_dataset import prepare


def request(*call_ids: str) -> Request:
    return Request(
        id=call_ids[0],
        timestamp="2026-11-13",
        provider="test",
        model="test",
        messages=[{"role": "system", "content": "Exact replay context"}],
        output={
            "role": "assistant",
            "tool_calls": [
                {"id": call_id, "function": {"name": "web_search"}}
                for call_id in call_ids
            ],
        },
    )


def assistant(
    message_id: str,
    parent_id: str | None,
    *call_ids: str,
    unfinished: bool = False,
) -> dict:
    return {
        "messageId": message_id,
        "parentMessageId": parent_id,
        "conversationId": "conversation",
        "isCreatedByUser": False,
        "unfinished": unfinished,
        "content": [
            {
                "type": "tool_call",
                "tool_call": {
                    "id": call_id,
                    "name": "web_search",
                    "args": {"query": "test"},
                    "output": f"result for {call_id}",
                },
            }
            for call_id in call_ids
        ]
        + [{"type": "text", "text": f"answer from {message_id}"}],
    }


def test_keeps_only_turns_with_all_calls_in_one_message() -> None:
    retained = request("second", "first")
    records, coverage = prepare(
        [
            retained,
            request("first", "second"),
            request("partial", "missing"),
            request("split-a", "split-b"),
            request("duplicated"),
            request("unmatched"),
        ],
        [
            assistant("search", None, "first", "second", "partial", "split-a"),
            assistant("other", None, "split-b", "duplicated", "duplicated"),
        ],
        [
            {
                "conversationId": "conversation",
                "lastResponseMessageId": "search",
            }
        ],
    )

    assert coverage == {
        "retained": 1,
        "partial": 1,
        "ambiguous": 2,
        "unmatched": 1,
    }
    (record,) = records
    assert record["history"] == retained.messages_with_output()
    assert record["originalSearchResults"] == [
        {"tool_call_id": "second", "output": "result for second"},
        {"tool_call_id": "first", "output": "result for first"},
    ]


def test_uses_only_an_ending_on_the_matched_branch() -> None:
    messages = [
        assistant("root", None, unfinished=True),
        assistant("search", "root", "call"),
        assistant("other", "root"),
    ]
    conversation = {
        "conversationId": "conversation",
        "lastResponseMessageId": "other",
    }
    (record,), _ = prepare([request("call")], messages, [conversation])
    assert record["originalAnswerStatus"] == "complete"
    assert record["originalConversationHasIncompleteMessages"]
    assert (
        record["originalConversation"][-1]["content"] == "answer from search"
    )

    messages.extend([assistant("a", "search"), assistant("b", "search")])
    (record,), _ = prepare([request("call")], messages, [conversation])
    assert record["originalAnswerStatus"] == "ambiguous_branch"
    assert (
        record["originalConversation"][-1]["content"] == "answer from search"
    )

    conversation["lastResponseMessageId"] = "a"
    messages[-2]["unfinished"] = True
    (record,), _ = prepare([request("call")], messages, [conversation])
    assert record["originalAnswerStatus"] == "unfinished"
    assert record["originalConversation"][-1]["content"] == "answer from a"
